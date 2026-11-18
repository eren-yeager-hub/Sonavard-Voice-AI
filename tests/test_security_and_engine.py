"""
Automated verification tests for Aegis Voice AI:
- Microphone-agnostic DSP & Adaptive VAD (quiet earphone & Bluetooth HFP invariance)
- Dual Feature Extractor shapes & finiteness
- Multilingual Keyword Radar (exact + fuzzy matching)
- Flask API Security (CSP, CSRF, upload magic-byte rejection, no secret leakage)
- Alert safety (cancel window + dry-run mode)
"""

from __future__ import annotations

import base64
import io
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.audio_dsp import (
    SAMPLE_RATE,
    TARGET_SAMPLES,
    AdaptiveVAD,
    preprocess_audio,
    simulate_microphone_profile,
)
from engine.features import extract_acoustic_features, extract_spectrogram_tensor
import app as aegis_app


def _make_sine_scream(duration: float = 2.0, amp: float = 0.25) -> np.ndarray:
    t = np.arange(int(SAMPLE_RATE * duration), dtype=np.float32) / SAMPLE_RATE
    # Frequency-modulated vocal-like burst (700Hz -> 1400Hz)
    freq = 900.0 + 350.0 * np.sin(2 * np.pi * 4.5 * t)
    y = amp * np.sin(2 * np.pi * freq * t)
    return y.astype(np.float32)


def test_dsp_and_earphone_vad() -> None:
    rng = np.random.default_rng(123)
    vad = AdaptiveVAD(initial_noise_floor=0.004)

    # 1. Pure silence stays inactive
    silence = rng.normal(0.0, 0.0005, size=TARGET_SAMPLES).astype(np.float32)
    res_sil = vad.evaluate(silence)
    assert not res_sil.is_active, f"Silence should be inactive, got {res_sil}"

    # 2. Quiet earphone / Bluetooth HFP signal (RMS ~0.01) MUST pass VAD and get soft-AGC normalized
    loud_src = _make_sine_scream(2.0, amp=0.3)
    bt_quiet = simulate_microphone_profile(loud_src, "bluetooth_hfp", rng)
    # Normalize to realistic quiet Bluetooth earphone level (RMS ~0.012)
    bt_quiet = bt_quiet * (0.012 / max(float(np.sqrt(np.mean(bt_quiet ** 2))), 1e-6))
    res_bt = vad.evaluate(bt_quiet)
    assert res_bt.is_active, f"Quiet Bluetooth earphone signal was wrongly dropped! {res_bt}"

    # 2b. After calibrating to a quiet room, even an ultra-quiet (0.004 RMS) earphone voice passes!
    vad.calibrate(silence)
    ultra_quiet = bt_quiet * (0.004 / 0.012)
    assert vad.evaluate(ultra_quiet).is_active, "Calibrated ultra-quiet earphone signal should pass VAD!"

    proc, diag = preprocess_audio(bt_quiet)
    assert proc.shape == (TARGET_SAMPLES,)
    assert np.all(np.isfinite(proc))
    assert diag["agc_gain_db"] > 5.0, f"Expected positive AGC gain for quiet earphone, got {diag}"
    assert 0.06 <= diag["post_rms"] <= 0.25, f"AGC output RMS out of canonical range: {diag}"

    # 3. Feature extraction across profiles produces finite, exact dimensions
    for prof in ["clean", "bluetooth_hfp", "wired_earbud", "low_gain_distant"]:
        sim = simulate_microphone_profile(loud_src, prof, rng)
        p, _ = preprocess_audio(sim)
        spec = extract_spectrogram_tensor(p)
        ac = extract_acoustic_features(p)
        assert spec.shape == (64, 63, 3), f"Bad spec shape: {spec.shape}"
        assert ac.shape == (142,), f"Bad acoustic feature shape: {ac.shape}"
        assert np.all(np.isfinite(spec)) and np.all(np.isfinite(ac))

    print("[PASS] test_dsp_and_earphone_vad")


def test_multilingual_keywords() -> None:
    det = aegis_app.detector
    assert det.match_keywords("Please help me right now!")["keyword"] in {"help", "help me"}
    assert det.match_keywords("Bhai bachao jaldi")["keyword"] == "bachao"
    assert det.match_keywords("madat karo koi")["keyword"] in {"madat", "madat karo"}
    assert det.match_keywords("bachaao please")["match_type"] == "fuzzy"
    assert det.match_keywords("socorro necesito ayuda")["keyword"] in {"ayuda", "socorro"}
    assert det.match_keywords("Good morning, how is the weather today?") is None
    print("[PASS] test_multilingual_keywords")


def test_api_security_and_end_to_end() -> None:
    client = aegis_app.app.test_client()

    # 1. Security headers on home page
    r = client.get("/")
    assert r.status_code == 200
    assert r.headers.get("X-Frame-Options") == "DENY"
    assert r.headers.get("X-Content-Type-Options") == "nosniff"
    assert "default-src 'self'" in (r.headers.get("Content-Security-Policy") or "")
    assert "microphone=(self)" in (r.headers.get("Permissions-Policy") or "")

    # 2. Bootstrap returns CSRF token and NEVER leaks secrets
    rb = client.get("/api/bootstrap")
    assert rb.status_code == 200
    boot = rb.get_json()
    csrf = boot["csrf_token"]
    assert csrf and len(csrf) > 16
    raw_dump = json.dumps(boot)
    assert "EMAIL_PASSWORD" not in raw_dump and "TWILIO_TOKEN" not in raw_dump

    # 3. Mutating endpoint WITHOUT CSRF token MUST be blocked (403)
    r_bad = client.post("/api/settings", json={"dry_run": True})
    assert r_bad.status_code == 403, f"Expected 403 without CSRF, got {r_bad.status_code}"

    headers = {"X-CSRF-Token": csrf}

    # 4. Mutating endpoint WITH valid CSRF succeeds
    r_ok = client.post("/api/settings", json={"dry_run": True, "sensitivity": "balanced"}, headers=headers)
    assert r_ok.status_code == 200

    # 5. Audio upload security: reject bad extension & spoofed magic bytes
    r_exe = client.post(
        "/api/analyze-file",
        data={"file": (io.BytesIO(b"MZ\x90\x00fakeexe"), "malware.exe")},
        headers=headers,
        content_type="multipart/form-data",
    )
    assert r_exe.status_code == 400

    r_spoof = client.post(
        "/api/analyze-file",
        data={"file": (io.BytesIO(b"<script>alert(1)</script>"), "fake.wav")},
        headers=headers,
        content_type="multipart/form-data",
    )
    assert r_spoof.status_code == 400

    # 6. Valid WAV upload succeeds through full dual-model File Lab
    if aegis_app.detector.is_ready:
        buf = io.BytesIO()
        sf.write(buf, _make_sine_scream(2.5, 0.2), SAMPLE_RATE, format="WAV")
        buf.seek(0)
        r_wav = client.post(
            "/api/analyze-file",
            data={"file": (buf, "test_sample.wav")},
            headers=headers,
            content_type="multipart/form-data",
        )
        assert r_wav.status_code == 200, f"Valid WAV failed: {r_wav.get_json()}"
        rep = r_wav.get_json()
        assert rep["windows_analyzed"] >= 1
        assert 0.0 <= rep["peak_fused_score"] <= 1.0

        # 7. Live PCM chunk analysis works end-to-end
        pcm = _make_sine_scream(2.0, 0.03)  # quiet earphone level
        b64 = base64.b64encode(pcm.astype(np.float32).tobytes()).decode("ascii")
        r_chunk = client.post(
            "/api/analyze-chunk",
            json={"sample_rate": SAMPLE_RATE, "format": "f32", "pcm_b64": b64},
            headers=headers,
        )
        assert r_chunk.status_code == 200, f"Chunk failed: {r_chunk.get_json()}"
        ck = r_chunk.get_json()
        assert ck["active"] is True
        assert ck["agc_gain_db"] > 0

    # 8. Alert cancel window works safely in Dry-Run mode
    r_tr = client.post("/api/transcript", json={"text": "please help me!", "is_final": True}, headers=headers)
    assert r_tr.status_code == 200
    assert r_tr.get_json()["matched"] is True

    r_cancel = client.post("/api/alert/cancel", headers=headers)
    assert r_cancel.status_code == 200

    print("[PASS] test_api_security_and_end_to_end")


if __name__ == "__main__":
    test_dsp_and_earphone_vad()
    test_multilingual_keywords()
    test_api_security_and_end_to_end()
    print("\nALL AUTOMATED TESTS PASSED!")
