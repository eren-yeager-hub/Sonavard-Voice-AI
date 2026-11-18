"""
Sonavard Voice AI — Hybrid Acoustic Distress & Parallel Multilingual Keyword Detector.

Combines:
1. Microphone-Agnostic DSP (High-pass + Soft-Knee AGC + Adaptive Noise-Floor VAD)
2. Dual AI Models:
   - 3-Channel Log-Mel 2D CNN (Keras 3)
   - Prosodic & Acoustic Feature Ensemble (HistGB + ExtraTrees + L2-MLP)
3. Fast-Attack / Slow-Decay Temporal Confidence Tracker (prevents accidental 1-frame spikes
   while reacting within ~1-2 seconds on real screams/crying/gasping)
4. Parallel Multilingual Keyword Radar (Instant transcript checking + optional backend STT)
5. Audio File Sliding-Window Lab Analyzer
"""

from __future__ import annotations

import io
import json
import os
import re
import threading
import time
from collections import deque
from difflib import SequenceMatcher
from typing import Callable, Optional

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(max(1, (os.cpu_count() or 4) // 2)))

import joblib
import librosa
import numpy as np
import soundfile as sf
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import config
from engine.audio_dsp import (
    SAMPLE_RATE,
    TARGET_SAMPLES,
    AdaptiveVAD,
    preprocess_audio,
    resample_audio,
    sanitize_audio,
)
from engine.features import extract_acoustic_features, extract_spectrogram_tensor

try:
    import sounddevice as sd
except Exception:
    sd = None

try:
    import speech_recognition as sr_lib
except Exception:
    sr_lib = None


class SpecAugmentLayer(layers.Layer):
    """Inference no-op matching training SpecAugmentLayer for Keras 3 deserialization."""

    def __init__(self, freq_mask_max: int = 8, time_mask_max: int = 10, **kwargs):
        super().__init__(**kwargs)
        self.freq_mask_max = freq_mask_max
        self.time_mask_max = time_mask_max

    def call(self, x, training=None):
        return x

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"freq_mask_max": self.freq_mask_max, "time_mask_max": self.time_mask_max})
        return cfg


class DistressDetectorEngine:
    """Thread-safe hybrid neural + prosodic + multilingual keyword detection engine."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._model_lock = threading.Lock()

        self.cnn_model: Optional[keras.Model] = None
        self.acoustic_model = None
        self.metadata: dict = {}

        self.cnn_weight: float = 0.50
        self.acoustic_weight: float = 0.50
        self.sensitivity: str = "balanced"  # "high_sensitivity" | "balanced" | "strict"
        self.thresholds: dict[str, float] = {
            "high_sensitivity": 0.45,
            "balanced": 0.55,
            "strict": 0.67,
        }

        self.vad = AdaptiveVAD(initial_noise_floor=0.004)
        self.ema_score: float = 0.0
        self.consecutive_hits: int = 0
        self.score_history: deque[dict] = deque(maxlen=90)

        self.keywords: list[str] = list(config.DEFAULT_KEYWORDS)
        self.language: str = "en-US"
        self.enable_server_stt: bool = False

        # Optional server-side microphone capture state
        self._server_stream_running: bool = False
        self._server_thread: Optional[threading.Thread] = None

        self.load_models()

    def load_models(self) -> bool:
        """Load trained Keras 3 CNN, Scikit-Learn Acoustic Ensemble, and metadata."""
        with self._model_lock:
            loaded_any = False
            if config.METADATA_PATH.is_file():
                try:
                    self.metadata = json.loads(config.METADATA_PATH.read_text(encoding="utf-8"))
                    self.cnn_weight = float(self.metadata.get("cnn_weight", 0.5))
                    self.acoustic_weight = float(self.metadata.get("acoustic_weight", 0.5))
                    if "thresholds" in self.metadata:
                        self.thresholds.update(self.metadata["thresholds"])
                except Exception:
                    self.metadata = {}

            if config.CNN_MODEL_PATH.is_file():
                try:
                    self.cnn_model = keras.models.load_model(
                        str(config.CNN_MODEL_PATH),
                        custom_objects={"SpecAugmentLayer": SpecAugmentLayer},
                        compile=False,
                    )
                    # Warmup single forward pass so first live chunk has zero JIT latency
                    dummy = np.zeros((1, 64, 63, 3), dtype=np.float32)
                    self.cnn_model(dummy, training=False)
                    loaded_any = True
                except Exception as exc:
                    print(f"[Sonavard] Warning: failed loading CNN model: {exc}")
                    self.cnn_model = None

            if config.ENSEMBLE_MODEL_PATH.is_file():
                try:
                    self.acoustic_model = joblib.load(str(config.ENSEMBLE_MODEL_PATH))
                    loaded_any = True
                except Exception as exc:
                    print(f"[Sonavard] Warning: failed loading Acoustic Ensemble: {exc}")
                    self.acoustic_model = None

            return loaded_any

    @property
    def is_ready(self) -> bool:
        return self.cnn_model is not None or self.acoustic_model is not None

    @property
    def current_threshold(self) -> float:
        return float(self.thresholds.get(self.sensitivity, self.thresholds["balanced"]))

    def reset_temporal_state(self) -> None:
        with self._lock:
            self.ema_score = 0.0
            self.consecutive_hits = 0

    def update_settings(
        self,
        sensitivity: Optional[str] = None,
        keywords: Optional[list[str]] = None,
        language: Optional[str] = None,
        enable_server_stt: Optional[bool] = None,
    ) -> dict:
        with self._lock:
            if sensitivity in self.thresholds:
                self.sensitivity = sensitivity
            if keywords is not None:
                cleaned = [k.strip().lower()[:60] for k in keywords if isinstance(k, str) and k.strip()]
                if cleaned:
                    self.keywords = list(dict.fromkeys(cleaned))[:60]
            if language and isinstance(language, str):
                self.language = language.strip()[:16]
            if enable_server_stt is not None:
                self.enable_server_stt = bool(enable_server_stt)
        return self.get_info()

    def get_info(self) -> dict:
        with self._lock:
            return {
                "ready": self.is_ready,
                "cnn_loaded": self.cnn_model is not None,
                "acoustic_loaded": self.acoustic_model is not None,
                "model_version": self.metadata.get("model_version", "untrained"),
                "cnn_weight": self.cnn_weight,
                "acoustic_weight": self.acoustic_weight,
                "sensitivity": self.sensitivity,
                "threshold": self.current_threshold,
                "thresholds": self.thresholds,
                "noise_floor": round(self.vad.noise_floor, 5),
                "vad_calibrated": self.vad.calibrated,
                "keywords": list(self.keywords),
                "language": self.language,
                "enable_server_stt": self.enable_server_stt,
                "overall_oof_metrics": self.metadata.get("overall_oof_metrics", {}),
                "mic_benchmarks": self.metadata.get("mic_benchmarks", {}),
            }

    def list_server_audio_devices(self) -> list[dict]:
        """List available server-side Windows/OS microphone devices safely."""
        if sd is None:
            return []
        devices = []
        try:
            default_in = sd.default.device[0] if isinstance(sd.default.device, (tuple, list)) else sd.default.device
            for idx, dev in enumerate(sd.query_devices()):
                if int(dev.get("max_input_channels", 0)) > 0:
                    devices.append(
                        {
                            "id": idx,
                            "name": str(dev.get("name", f"Device {idx}")),
                            "channels": int(dev.get("max_input_channels", 1)),
                            "default_samplerate": int(dev.get("default_samplerate", 16000)),
                            "is_default": idx == default_in,
                        }
                    )
        except Exception:
            return []
        return devices

    def calibrate_noise_floor(self, ambient_audio: np.ndarray, sr: int = SAMPLE_RATE) -> dict:
        floor = self.vad.calibrate(ambient_audio, sr=sr)
        return {
            "noise_floor": round(floor, 5),
            "noise_floor_db": round(20.0 * float(np.log10(max(floor, 1e-6))), 1),
            "calibrated": True,
        }

    def _predict_preprocessed(self, proc_audio: np.ndarray) -> tuple[float, float, float]:
        """Run both models on a preprocessed 2.0s 16kHz float32 waveform."""
        spec = extract_spectrogram_tensor(proc_audio, SAMPLE_RATE)[None, ...]
        ac_feat = extract_acoustic_features(proc_audio, SAMPLE_RATE)[None, ...]

        with self._model_lock:
            p_cnn = None
            p_ac = None
            if self.cnn_model is not None:
                p_cnn = float(self.cnn_model(spec, training=False).numpy()[0, 0])
            if self.acoustic_model is not None:
                p_ac = float(self.acoustic_model.predict_proba(ac_feat)[0, 1])

        if p_cnn is not None and p_ac is not None:
            p_fused = self.cnn_weight * p_cnn + self.acoustic_weight * p_ac
        elif p_cnn is not None:
            p_ac = p_cnn
            p_fused = p_cnn
        elif p_ac is not None:
            p_cnn = p_ac
            p_fused = p_ac
        else:
            p_cnn = p_ac = p_fused = 0.0

        return float(p_cnn), float(p_ac), float(p_fused)

    def analyze_live_chunk(
        self,
        raw_audio: np.ndarray,
        sr: int = SAMPLE_RATE,
        run_server_stt: bool = False,
    ) -> dict:
        """
        Analyze a live 2.0s sliding window chunk from browser mic OR server mic.
        Works cleanly on quiet earphone mics, Bluetooth HFP, USB headsets, and laptop mics.
        """
        t_start = time.perf_counter()
        raw_audio = sanitize_audio(raw_audio)
        vad_res = self.vad.evaluate(raw_audio, sr=sr)

        if not vad_res.is_active:
            with self._lock:
                # Smoothly decay EMA when silent
                self.ema_score = max(0.0, self.ema_score * 0.65)
                self.consecutive_hits = 0
                ema = self.ema_score
                thr = self.current_threshold

            result = {
                "timestamp": round(time.time(), 2),
                "active": False,
                "vad_reason": vad_res.reason,
                "raw_rms": round(vad_res.raw_rms, 5),
                "raw_peak": round(vad_res.raw_peak, 4),
                "noise_floor": round(vad_res.noise_floor, 5),
                "snr_db": round(vad_res.snr_db, 1),
                "agc_gain_db": 0.0,
                "cnn_score": 0.0,
                "acoustic_score": 0.0,
                "fused_score": 0.0,
                "smoothed_score": round(ema, 3),
                "threshold": round(thr, 3),
                "threat_level": "idle",
                "distress_detected": False,
                "keyword_match": None,
                "transcript": "",
                "latency_ms": round((time.perf_counter() - t_start) * 1000.0, 1),
            }
            self.score_history.append(result)
            return result

        # Preprocess & Run Dual Models
        proc, diag = preprocess_audio(raw_audio, sr=sr, target_samples=TARGET_SAMPLES)
        p_cnn, p_ac, p_fused = self._predict_preprocessed(proc)

        # Fast-attack, slow-decay temporal tracking:
        # Reacts fast to sudden loud screams, resists 1-frame accidental thumps
        with self._lock:
            thr = self.current_threshold
            alpha = 0.68 if p_fused > self.ema_score else 0.35
            self.ema_score = alpha * p_fused + (1.0 - alpha) * self.ema_score

            # High-confidence single scream (>= thr + 0.16) triggers immediately;
            # moderate distress requires 2 consecutive elevated windows to prevent false alarms
            if p_fused >= min(0.92, thr + 0.16) and self.ema_score >= thr:
                self.consecutive_hits += 2
            elif self.ema_score >= thr:
                self.consecutive_hits += 1
            else:
                self.consecutive_hits = max(0, self.consecutive_hits - 1)

            required_hits = 1 if self.sensitivity == "high_sensitivity" else 2
            distress_detected = self.consecutive_hits >= required_hits and self.ema_score >= thr
            ema = self.ema_score

        # Optional server-side STT fallback (only when explicitly enabled & online)
        transcript = ""
        kw_match = None
        if run_server_stt and self.enable_server_stt and sr_lib is not None:
            transcript = self._run_server_stt(proc)
            if transcript:
                kw_match = self.match_keywords(transcript)

        if distress_detected or kw_match:
            threat_level = "critical"
        elif ema >= thr * 0.72:
            threat_level = "elevated"
        else:
            threat_level = "monitoring"

        result = {
            "timestamp": round(time.time(), 2),
            "active": True,
            "vad_reason": vad_res.reason,
            "raw_rms": round(vad_res.raw_rms, 5),
            "raw_peak": round(vad_res.raw_peak, 4),
            "noise_floor": round(vad_res.noise_floor, 5),
            "snr_db": round(vad_res.snr_db, 1),
            "agc_gain_db": round(diag["agc_gain_db"], 1),
            "cnn_score": round(p_cnn, 3),
            "acoustic_score": round(p_ac, 3),
            "fused_score": round(p_fused, 3),
            "smoothed_score": round(ema, 3),
            "threshold": round(thr, 3),
            "threat_level": threat_level,
            "distress_detected": bool(distress_detected),
            "keyword_match": kw_match,
            "transcript": transcript,
            "latency_ms": round((time.perf_counter() - t_start) * 1000.0, 1),
        }
        self.score_history.append(result)
        return result

    def match_keywords(self, transcript: str) -> Optional[dict]:
        """
        Multilingual distress keyword matcher with exact phrase + fuzzy word matching.
        Handles English, Hindi/Hinglish, Spanish, French, and custom keywords.
        """
        if not transcript or not isinstance(transcript, str):
            return None
        clean = re.sub(r"[^\w\s'-]", " ", transcript.lower())
        clean = re.sub(r"\s+", " ", clean).strip()
        if not clean:
            return None

        words = clean.split()
        with self._lock:
            kw_list = list(self.keywords)

        # 1. Exact phrase or whole-word match
        for kw in kw_list:
            kw_clean = re.sub(r"\s+", " ", kw.lower()).strip()
            if not kw_clean:
                continue
            pattern = r"(?<!\w)" + re.escape(kw_clean) + r"(?!\w)"
            if re.search(pattern, clean):
                return {"keyword": kw, "match_type": "exact", "transcript": transcript[:200]}

        # 2. Fuzzy match for phonetic variants (e.g. "bachaao", "madat", "emergancy") on words >= 5 chars
        for kw in kw_list:
            kw_clean = kw.lower().strip()
            if " " in kw_clean or len(kw_clean) < 5:
                continue
            for w in words:
                if abs(len(w) - len(kw_clean)) <= 2 and SequenceMatcher(None, w, kw_clean).ratio() >= 0.84:
                    return {"keyword": kw, "match_type": "fuzzy", "matched_token": w, "transcript": transcript[:200]}

        return None

    def _run_server_stt(self, proc_audio: np.ndarray) -> str:
        """Best-effort Google Web Speech API fallback (gracefully silent if offline)."""
        if sr_lib is None:
            return ""
        try:
            recognizer = sr_lib.Recognizer()
            pcm16 = np.clip(proc_audio * 32767.0, -32768, 32767).astype(np.int16)
            audio_data = sr_lib.AudioData(pcm16.tobytes(), SAMPLE_RATE, 2)
            return recognizer.recognize_google(audio_data, language=self.language)
        except Exception:
            return ""

    def analyze_audio_file(self, data: bytes, filename: str) -> dict:
        """
        Decode an uploaded audio file in memory and run sliding-window distress analysis.
        Returns per-window timeline, peak distress score, and diagnosis.
        """
        t_start = time.perf_counter()
        bio = io.BytesIO(data)
        try:
            y, sr = sf.read(bio, dtype="float32")
        except Exception:
            bio.seek(0)
            y, sr = librosa.load(bio, sr=SAMPLE_RATE, mono=True)

        y = resample_audio(y, int(sr), SAMPLE_RATE)
        duration = round(y.size / float(SAMPLE_RATE), 2)

        # Cap analysis to first 60 seconds to prevent DoS on huge files
        max_samples = SAMPLE_RATE * 60
        if y.size > max_samples:
            y = y[:max_samples]

        hop = SAMPLE_RATE  # 1.0s step
        windows: list[dict] = []

        if y.size <= TARGET_SAMPLES:
            starts = [0]
        else:
            starts = list(range(0, y.size - TARGET_SAMPLES + 1, hop))
            if starts[-1] + TARGET_SAMPLES < y.size:
                starts.append(y.size - TARGET_SAMPLES)

        thr = self.current_threshold
        peak_fused = 0.0
        peak_cnn = 0.0
        peak_ac = 0.0

        for s in starts:
            seg = y[s : s + TARGET_SAMPLES]
            proc, diag = preprocess_audio(seg, SAMPLE_RATE, TARGET_SAMPLES)
            p_cnn, p_ac, p_fused = self._predict_preprocessed(proc)
            peak_fused = max(peak_fused, p_fused)
            peak_cnn = max(peak_cnn, p_cnn)
            peak_ac = max(peak_ac, p_ac)
            windows.append(
                {
                    "start_sec": round(s / float(SAMPLE_RATE), 2),
                    "end_sec": round(min(y.size, s + TARGET_SAMPLES) / float(SAMPLE_RATE), 2),
                    "raw_rms": round(diag["raw_rms"], 4),
                    "agc_gain_db": round(diag["agc_gain_db"], 1),
                    "cnn_score": round(p_cnn, 3),
                    "acoustic_score": round(p_ac, 3),
                    "fused_score": round(p_fused, 3),
                    "flagged": p_fused >= thr,
                }
            )

        flagged_count = sum(1 for w in windows if w["flagged"])
        verdict = "DISTRESS DETECTED" if peak_fused >= thr else "NORMAL / SAFE"

        return {
            "filename": filename,
            "duration_sec": duration,
            "windows_analyzed": len(windows),
            "flagged_windows": flagged_count,
            "peak_cnn_score": round(peak_cnn, 3),
            "peak_acoustic_score": round(peak_ac, 3),
            "peak_fused_score": round(peak_fused, 3),
            "threshold": round(thr, 3),
            "verdict": verdict,
            "is_emergency": peak_fused >= thr,
            "timeline": windows,
            "latency_ms": round((time.perf_counter() - t_start) * 1000.0, 1),
        }

    # ------------------------------------------------------------------
    # Optional Server-Side Continuous Ring-Buffer Mic Capture
    # ------------------------------------------------------------------
    def start_server_capture(
        self,
        device_id: Optional[int],
        on_result: Callable[[dict, np.ndarray], None],
        on_error: Callable[[str], None],
    ) -> bool:
        if sd is None:
            on_error("Server sounddevice library is unavailable.")
            return False

        self.stop_server_capture()
        self.reset_temporal_state()
        self._server_stream_running = True

        def _worker() -> None:
            try:
                dev_info = sd.query_devices(device_id, "input") if device_id is not None else sd.query_devices(kind="input")
                native_sr = int(dev_info.get("default_samplerate", SAMPLE_RATE))
            except Exception:
                native_sr = SAMPLE_RATE

            # Use device's native sample rate (prevents Bluetooth HFP 8kHz/48kHz open failures!)
            # and resample cleanly to 16kHz in DSP.
            use_sr = native_sr if native_sr in {8000, 16000, 22050, 32000, 44100, 48000} else SAMPLE_RATE
            buf_samples = int(use_sr * 2.0)
            ring = np.zeros(buf_samples, dtype=np.float32)
            ring_lock = threading.Lock()
            samples_received = 0

            def _audio_cb(indata, frames, time_info, status):
                nonlocal samples_received
                mono = indata[:, 0].astype(np.float32)
                n = mono.size
                with ring_lock:
                    if n >= buf_samples:
                        ring[:] = mono[-buf_samples:]
                    else:
                        ring[:-n] = ring[n:]
                        ring[-n:] = mono
                    samples_received += n

            try:
                with sd.InputStream(
                    device=device_id,
                    channels=1,
                    samplerate=use_sr,
                    dtype="float32",
                    blocksize=int(use_sr * 0.1),
                    callback=_audio_cb,
                ):
                    while self._server_stream_running:
                        time.sleep(0.95)
                        if not self._server_stream_running:
                            break
                        with ring_lock:
                            if samples_received < int(use_sr * 1.2):
                                continue
                            snap = ring.copy()
                        res = self.analyze_live_chunk(snap, sr=use_sr, run_server_stt=self.enable_server_stt)
                        snap_16k = resample_audio(snap, use_sr, SAMPLE_RATE)
                        on_result(res, snap_16k)
            except Exception as exc:
                on_error(f"Server mic capture error ({type(exc).__name__}): {exc}")
            finally:
                self._server_stream_running = False

        self._server_thread = threading.Thread(target=_worker, daemon=True)
        self._server_thread.start()
        return True

    def stop_server_capture(self) -> None:
        self._server_stream_running = False
        if self._server_thread and self._server_thread.is_alive():
            self._server_thread.join(timeout=2.0)
        self._server_thread = None
