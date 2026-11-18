"""
Microphone-agnostic Digital Signal Processing (DSP), Adaptive Noise-Floor VAD,
Automatic Gain Control (AGC), and Multi-Microphone Domain Randomization Simulator.

Designed specifically so quiet earphone mics, Bluetooth HFP headsets (300-3400Hz),
USB mics, and laptop microphones all map into the same canonical acoustic space.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.signal import butter, iirpeak, lfilter, resample_poly, sosfiltfilt

SAMPLE_RATE = 16000
WINDOW_SECONDS = 2.0
TARGET_SAMPLES = int(SAMPLE_RATE * WINDOW_SECONDS)


def sanitize_audio(audio: np.ndarray) -> np.ndarray:
    """Ensure finite 1D float32 mono waveform with zero DC offset."""
    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim > 1:
        arr = np.mean(arr, axis=-1, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if arr.size == 0:
        return np.zeros(TARGET_SAMPLES, dtype=np.float32)
    arr = arr - float(np.mean(arr))
    return arr.astype(np.float32, copy=False)


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """High-quality polyphase resampling to target_sr."""
    arr = sanitize_audio(audio)
    if orig_sr == target_sr or orig_sr <= 0:
        return arr
    g = math.gcd(int(orig_sr), int(target_sr))
    up = int(target_sr) // g
    down = int(orig_sr) // g
    resampled = resample_poly(arr, up, down).astype(np.float32)
    return sanitize_audio(resampled)


def highpass_filter(audio: np.ndarray, sr: int = SAMPLE_RATE, cutoff_hz: float = 85.0) -> np.ndarray:
    """4th-order Butterworth high-pass filter to remove hum, breath rumble, and cable bumps."""
    if audio.size < 32:
        return audio
    nyq = 0.5 * sr
    norm_cutoff = min(max(cutoff_hz / nyq, 1e-4), 0.99)
    sos = butter(4, norm_cutoff, btype="highpass", output="sos")
    try:
        return sosfiltfilt(sos, audio).astype(np.float32)
    except ValueError:
        return audio.astype(np.float32)


def bandpass_filter(
    audio: np.ndarray,
    low_hz: float,
    high_hz: float,
    sr: int = SAMPLE_RATE,
    order: int = 4,
) -> np.ndarray:
    """Zero-phase Butterworth bandpass filter for microphone channel simulation."""
    if audio.size < 32:
        return audio
    nyq = 0.5 * sr
    low = min(max(low_hz / nyq, 1e-4), 0.95)
    high = min(max(high_hz / nyq, low + 0.02), 0.99)
    sos = butter(order, [low, high], btype="bandpass", output="sos")
    try:
        return sosfiltfilt(sos, audio).astype(np.float32)
    except ValueError:
        return audio.astype(np.float32)


def soft_agc(
    audio: np.ndarray,
    target_rms: float = 0.125,
    max_gain: float = 55.0,
    noise_gate_rms: float = 0.0015,
) -> tuple[np.ndarray, float]:
    """
    Soft-knee Automatic Gain Control (AGC).
    Brings quiet earphone mics (RMS ~0.008) and loud studio mics (RMS ~0.25)
    into the same canonical loudness envelope without amplifying dead silence.
    Returns (normalized_audio, applied_gain_db).
    """
    rms = float(np.sqrt(np.mean(audio * audio) + 1e-12))
    if rms < noise_gate_rms:
        return audio.astype(np.float32), 0.0

    gain = min(target_rms / rms, max_gain)
    # Prevent attenuating below 0.25x so extremely loud clips keep clarity
    gain = max(gain, 0.25)
    boosted = audio * gain
    # Smooth soft limiter preserves transient dynamics while bounding peaks to [-0.95, 0.95]
    limited = 0.95 * np.tanh(boosted / 0.95)
    gain_db = 20.0 * math.log10(max(gain, 1e-6))
    return limited.astype(np.float32), float(gain_db)


def fit_length(audio: np.ndarray, target_samples: int = TARGET_SAMPLES, select_loudest: bool = True) -> np.ndarray:
    """
    Standardize waveform length to exactly target_samples.
    If longer, selects the highest-energy window (so screams aren't clipped off).
    If shorter, centers it with gentle zero padding.
    """
    n = audio.size
    if n == target_samples:
        return audio
    if n > target_samples:
        if select_loudest:
            hop = max(target_samples // 8, 512)
            best_start = 0
            best_energy = -1.0
            for start in range(0, n - target_samples + 1, hop):
                seg = audio[start : start + target_samples]
                e = float(np.mean(seg * seg))
                if e > best_energy:
                    best_energy = e
                    best_start = start
            # Also check the very last window
            tail = audio[n - target_samples :]
            if float(np.mean(tail * tail)) > best_energy:
                return tail.astype(np.float32)
            return audio[best_start : best_start + target_samples].astype(np.float32)
        return audio[:target_samples].astype(np.float32)

    pad_total = target_samples - n
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    return np.pad(audio, (pad_left, pad_right), mode="constant").astype(np.float32)


def preprocess_audio(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    target_samples: int = TARGET_SAMPLES,
) -> tuple[np.ndarray, dict[str, float]]:
    """
    Canonical microphone-agnostic preprocessing shared by BOTH training and live inference.
    Returns (processed_waveform, diagnostics_dict).
    """
    raw = resample_audio(audio, sr, SAMPLE_RATE) if sr != SAMPLE_RATE else sanitize_audio(raw_audio := audio)
    filtered = highpass_filter(raw, SAMPLE_RATE, cutoff_hz=85.0)

    raw_rms = float(np.sqrt(np.mean(filtered * filtered) + 1e-12))
    raw_peak = float(np.max(np.abs(filtered))) if filtered.size else 0.0

    sized = fit_length(filtered, target_samples=target_samples, select_loudest=True)
    normalized, agc_gain_db = soft_agc(sized)

    post_rms = float(np.sqrt(np.mean(normalized * normalized) + 1e-12))
    diagnostics = {
        "raw_rms": raw_rms,
        "raw_peak": raw_peak,
        "post_rms": post_rms,
        "agc_gain_db": agc_gain_db,
    }
    return normalized, diagnostics


@dataclass
class VADResult:
    is_active: bool
    raw_rms: float
    raw_peak: float
    noise_floor: float
    snr_db: float
    spectral_flatness: float
    reason: str


class AdaptiveVAD:
    """
    Adaptive Noise-Floor Voice & Event Activity Detector.
    Replaces hardcoded RMS/peak thresholds so quiet earphone mics work reliably
    while stationary room noise (AC/fan/hiss) stays suppressed.
    """

    def __init__(self, initial_noise_floor: float = 0.0025) -> None:
        self.noise_floor: float = max(0.0008, float(initial_noise_floor))
        self.calibrated: bool = False

    def calibrate(self, ambient_audio: np.ndarray, sr: int = SAMPLE_RATE) -> float:
        """Calibrate noise floor from ~1.5-2s of ambient room silence on user's current mic."""
        clean = highpass_filter(resample_audio(ambient_audio, sr, SAMPLE_RATE))
        # Compute 100ms frame RMS and take 65th percentile as noise floor
        frame_len = int(SAMPLE_RATE * 0.1)
        if clean.size >= frame_len:
            n_frames = clean.size // frame_len
            frames = clean[: n_frames * frame_len].reshape(n_frames, frame_len)
            frame_rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)
            measured = float(np.percentile(frame_rms, 65))
        else:
            measured = float(np.sqrt(np.mean(clean * clean) + 1e-12))

        self.noise_floor = float(np.clip(measured, 0.0008, 0.08))
        self.calibrated = True
        return self.noise_floor

    def evaluate(self, raw_audio: np.ndarray, sr: int = SAMPLE_RATE) -> VADResult:
        clean = highpass_filter(resample_audio(raw_audio, sr, SAMPLE_RATE))
        rms = float(np.sqrt(np.mean(clean * clean) + 1e-12))
        peak = float(np.max(np.abs(clean))) if clean.size else 0.0

        # Frame-level peak energy catches short screams/gasps inside a 2s window
        frame_len = int(SAMPLE_RATE * 0.1)
        if clean.size >= frame_len:
            n_frames = clean.size // frame_len
            frames = clean[: n_frames * frame_len].reshape(n_frames, frame_len)
            frame_rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)
            active_rms = float(np.percentile(frame_rms, 85))
            quiet_rms = float(np.percentile(frame_rms, 20))
        else:
            active_rms = rms
            quiet_rms = rms

        # Quick FFT spectral flatness (white hiss -> ~0.6+, human voice/scream -> <0.35)
        spec = np.abs(np.fft.rfft(clean[: min(clean.size, 16384)])) + 1e-10
        geo_mean = float(np.exp(np.mean(np.log(spec))))
        arith_mean = float(np.mean(spec))
        flatness = geo_mean / max(arith_mean, 1e-10)

        snr_db = 20.0 * math.log10(max(active_rms, 1e-6) / max(self.noise_floor, 1e-6))

        # Absolute minimal electrical floor (handles very quiet Bluetooth/earphone mics!)
        min_active_rms = max(0.0018, self.noise_floor * 1.75)

        if active_rms < 0.0012 and peak < 0.008:
            # Update noise floor gently during silence
            self._adapt_floor(quiet_rms)
            return VADResult(False, rms, peak, self.noise_floor, snr_db, flatness, "silence")

        if active_rms < min_active_rms and snr_db < 5.0:
            self._adapt_floor(quiet_rms)
            return VADResult(False, rms, peak, self.noise_floor, snr_db, flatness, "ambient_noise")

        # Pure stationary hiss with low SNR is ignored
        if flatness > 0.55 and snr_db < 9.0:
            self._adapt_floor(quiet_rms)
            return VADResult(False, rms, peak, self.noise_floor, snr_db, flatness, "stationary_hiss")

        # Slowly track lower quiet frames even during speech
        if quiet_rms < self.noise_floor * 1.4:
            self._adapt_floor(quiet_rms, rate=0.05)

        return VADResult(True, rms, peak, self.noise_floor, snr_db, flatness, "active_audio")

    def _adapt_floor(self, measured_rms: float, rate: float = 0.12) -> None:
        clamped = float(np.clip(measured_rms, 0.0008, 0.06))
        self.noise_floor = (1.0 - rate) * self.noise_floor + rate * clamped


# =====================================================================
# MULTI-MICROPHONE DOMAIN RANDOMIZATION (FOR TRAINING & BENCHMARKING)
# =====================================================================

def _apply_peaking_eq(audio: np.ndarray, freq_hz: float, q: float, gain_db: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Apply a simple peaking resonance filter simulating earphone mic capsule acoustics."""
    w0 = min(max(freq_hz / (0.5 * sr), 0.02), 0.92)
    b, a = iirpeak(w0, Q=q)
    resonant = lfilter(b, a, audio)
    mix = 10.0 ** (gain_db / 20.0) - 1.0
    return (audio + 0.35 * mix * resonant).astype(np.float32)


def _apply_spectral_tilt(audio: np.ndarray, tilt: float) -> np.ndarray:
    """Apply first-order pre-emphasis / de-emphasis tilt (-0.85 to +0.85)."""
    if abs(tilt) < 0.02:
        return audio
    if tilt > 0:
        return lfilter([1.0, -tilt], [1.0], audio).astype(np.float32)
    return lfilter([1.0], [1.0, tilt], audio).astype(np.float32)


def _apply_simple_reverb(audio: np.ndarray, rng: np.random.Generator, decay_ms: float = 120.0) -> np.ndarray:
    """Fast synthetic room reverb via exponentially decaying noise impulse response."""
    ir_len = int(SAMPLE_RATE * (decay_ms / 1000.0))
    if ir_len < 64:
        return audio
    t = np.linspace(0.0, 1.0, ir_len, dtype=np.float32)
    ir = rng.standard_normal(ir_len).astype(np.float32) * np.exp(-6.0 * t)
    ir[0] = 1.0
    ir /= np.sqrt(np.sum(ir * ir) + 1e-8)
    wet = np.convolve(audio, ir, mode="full")[: audio.size]
    wet_mix = float(rng.uniform(0.15, 0.42))
    return ((1.0 - wet_mix) * audio + wet_mix * wet).astype(np.float32)


def simulate_microphone_profile(
    audio: np.ndarray,
    profile: str,
    rng: np.random.Generator,
    ambient_pool: Optional[list[np.ndarray]] = None,
) -> np.ndarray:
    """
    Simulate real-world microphone hardware & environments BEFORE canonical preprocessing.
    Profiles:
      - 'clean': native audio with mild gain jitter
      - 'bluetooth_hfp': narrowband 300-3400Hz + nasal mic resonance + low gain + codec noise
      - 'wired_earbud': high-pass 200-350Hz + low-pass 4500-6200Hz + tinny tilt + low gain
      - 'low_gain_distant': very quiet gain (0.04x-0.15x) + room reverb + background noise
      - 'noisy_street': bandpass + real ambient noise mixed at 6-16 dB SNR
    """
    y = sanitize_audio(audio).copy()

    if profile == "clean":
        y *= float(rng.uniform(0.4, 1.15))
        return y

    if profile == "bluetooth_hfp":
        low = float(rng.uniform(260.0, 360.0))
        high = float(rng.uniform(3100.0, 3600.0))
        y = bandpass_filter(y, low, high)
        y = _apply_peaking_eq(y, freq_hz=float(rng.uniform(1300.0, 2400.0)), q=2.5, gain_db=float(rng.uniform(3.0, 7.5)))
        # Subtle nonlinear codec / compression artifact
        y = np.tanh(y * float(rng.uniform(1.1, 2.0))) / 1.5
        # Typical quiet Bluetooth headset input gain
        y *= float(rng.uniform(0.05, 0.25))
        # Electronic headset self-noise
        rms = float(np.sqrt(np.mean(y * y) + 1e-10))
        snr = float(rng.uniform(14.0, 26.0))
        noise_std = rms / (10.0 ** (snr / 20.0))
        y += rng.normal(0.0, noise_std, size=y.size).astype(np.float32)
        return y

    if profile == "wired_earbud":
        low = float(rng.uniform(160.0, 320.0))
        high = float(rng.uniform(4200.0, 6200.0))
        y = bandpass_filter(y, low, high)
        y = _apply_spectral_tilt(y, float(rng.uniform(-0.45, 0.55)))
        y = _apply_peaking_eq(y, freq_hz=float(rng.uniform(1800.0, 3200.0)), q=2.0, gain_db=float(rng.uniform(2.0, 6.0)))
        y *= float(rng.uniform(0.05, 0.35))
        if ambient_pool and rng.random() < 0.65:
            y = mix_ambient_noise(y, ambient_pool[int(rng.integers(0, len(ambient_pool)))], float(rng.uniform(10.0, 22.0)), rng)
        return y

    if profile == "low_gain_distant":
        low = float(rng.uniform(120.0, 260.0))
        high = float(rng.uniform(2800.0, 4800.0))
        y = bandpass_filter(y, low, high)
        y = _apply_simple_reverb(y, rng, decay_ms=float(rng.uniform(80.0, 200.0)))
        y *= float(rng.uniform(0.03, 0.15))
        if ambient_pool:
            y = mix_ambient_noise(y, ambient_pool[int(rng.integers(0, len(ambient_pool)))], float(rng.uniform(8.0, 18.0)), rng)
        else:
            rms = float(np.sqrt(np.mean(y * y) + 1e-10))
            y += rng.normal(0.0, rms * 0.08, size=y.size).astype(np.float32)
        return y

    if profile == "noisy_street":
        y = bandpass_filter(y, float(rng.uniform(140.0, 300.0)), float(rng.uniform(3500.0, 6000.0)))
        y *= float(rng.uniform(0.08, 0.5))
        if ambient_pool:
            y = mix_ambient_noise(y, ambient_pool[int(rng.integers(0, len(ambient_pool)))], float(rng.uniform(5.0, 15.0)), rng)
        else:
            rms = float(np.sqrt(np.mean(y * y) + 1e-10))
            y += rng.normal(0.0, rms * 0.18, size=y.size).astype(np.float32)
        return y

    return y


def mix_ambient_noise(
    signal: np.ndarray,
    noise: np.ndarray,
    snr_db: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Mix an ambient noise clip into signal at a specified SNR in dB."""
    n = signal.size
    if noise.size < n:
        reps = int(math.ceil(n / max(noise.size, 1)))
        noise_seg = np.tile(noise, reps)[:n]
    else:
        start = int(rng.integers(0, max(1, noise.size - n + 1)))
        noise_seg = noise[start : start + n]

    sig_rms = float(np.sqrt(np.mean(signal * signal) + 1e-10))
    noise_rms = float(np.sqrt(np.mean(noise_seg * noise_seg) + 1e-10))
    target_noise_rms = sig_rms / (10.0 ** (snr_db / 20.0))
    scaled_noise = noise_seg * (target_noise_rms / noise_rms)
    return (signal + scaled_noise).astype(np.float32)


def perturb_speed_and_shift(audio: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Fast resampling-based pitch/speed perturbation (0.88x-1.12x) + circular time shift."""
    y = audio.copy()
    if rng.random() < 0.7:
        factor = float(rng.uniform(0.88, 1.12))
        new_len = max(1000, int(y.size / factor))
        indices = np.linspace(0, y.size - 1, new_len, dtype=np.float32)
        y = np.interp(indices, np.arange(y.size, dtype=np.float32), y).astype(np.float32)
        y = fit_length(y, TARGET_SAMPLES, select_loudest=False)
    if rng.random() < 0.7:
        shift = int(rng.integers(-y.size // 4, y.size // 4))
        y = np.roll(y, shift)
    return y.astype(np.float32)
