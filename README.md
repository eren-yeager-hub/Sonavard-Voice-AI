# Aegis Voice AI — Multi-Microphone Neural Distress & Emergency Sentinel

A complete, security-hardened, real-time acoustic distress and multilingual emergency detection platform built to work reliably across **Bluetooth earphones (HFP 300–3400 Hz), wired earbuds, USB headsets, and laptop microphones**.

---

## Why This Upgrade Fixes Earphone Mic & Dataset Overfitting

The original pipeline relied on hardcoded raw-amplitude gates (`RMS >= 0.07`, `Peak >= 0.6`), static MFCC means (which memorize microphone EQ curves), and a randomly split MLP where segments from the same file leaked across train and test sets.

**Aegis Voice AI v3.0** solves every one of those root causes:

1. **Microphone-Agnostic Signal Processing (`engine/audio_dsp.py`)**
   - **85 Hz Butterworth High-Pass Filter** strips cable bumps, breath pops, DC drift, and 50/60 Hz hum.
   - **Soft-Knee Automatic Gain Control (AGC)** maps quiet earphone mics (RMS `0.008–0.04`) and loud studio mics (RMS `0.25`) into the same canonical loudness envelope.
   - **Adaptive Noise-Floor VAD** tracks ambient room noise (with a 1-click **2-second Mic Calibration** in the UI) so quiet earphone audio is never discarded while steady fan/AC hiss stays filtered.

2. **Microphone-Invariant Dual AI Engine (`engine/features.py`, `engine/detector.py`)**
   - **Branch A — 3-Channel 2D Log-Mel CNN (`models/distress_cnn.keras`)**:
     - Channel 0: Per-clip Z-normalized Log-Mel Spectrogram (`120–6500 Hz`)
     - Channel 1: **Time-Delta Log-Mel** (mathematically cancels static microphone frequency response $H(f)$)
     - Channel 2: Per-frame Spectral Shape (invariant to global volume envelope)
     - Uses Dual Pooling (`GlobalAveragePooling + GlobalMaxPooling`) + SpecAugment to detect both sustained cries and brief screams/gasps.
   - **Branch B — Prosodic & Acoustic Ensemble (`models/acoustic_ensemble.joblib`)**:
     - `RobustScaler` + Soft-Voting Ensemble (`HistGradientBoosting` + `ExtraTrees` + `L2-MLP`) on 142 microphone-invariant features: CMVN MFCC dispersion, delta/delta-delta dynamics, spectral contrast, spectral flux/flatness, crest factor, universal `300–3400 Hz` sub-band ratios, and autocorrelation pitch/F0 stress indicators.

3. **Zero-Leakage Multi-Microphone Domain Randomization (`train.py`)**
   - Source-level deduplication + strict `StratifiedGroupKFold` grouped by original recording.
   - Training audio is augmented across simulated **Bluetooth HFP**, **Wired Earbud**, **Low-Gain Distant Room**, and **Noisy Street** profiles plus synthetic hard negatives (hiss, 50/60 Hz hum, cable taps).

4. **Parallel Multilingual Keyword Radar**
   - Runs **simultaneously** alongside acoustic distress detection (no alternating blind spots).
   - Supports English, Hindi/Hinglish, Spanish, French, and custom keywords with exact + fuzzy matching.

5. **Security & Safety Hardening (`engine/security.py`, `engine/alerts.py`)**
   - Safe localhost binding (`127.0.0.1:5000`) with `debug=False` by default.
   - Strict CSP (zero inline scripts), `X-Frame-Options: DENY`, `X-Content-Type-Options: nosniff`, `Permissions-Policy`.
   - Same-Origin + Session CSRF token validation and sliding-window rate limiting on mutating API routes.
   - Strict uploaded audio validation (10 MB cap, extension whitelist, magic-byte header verification, in-memory decoding).
   - **Safe Mode (Dry-Run)** enabled by default, **5-second False-Alarm Cancel Countdown**, cooldown enforcement, and privacy-first opt-in evidence recording.

---

## Quick Start

### 1. Configure Environment (Optional — for Live Email/SMS Alerts)
Copy `.env.example` to `.env` and fill in your Gmail App Password / Twilio credentials:
```powershell
Copy-Item .env.example .env
```
> By default `AEGIS_ALERT_DRY_RUN=true` (Safe Mode), so you can test everything safely without sending live SMS/Emails. Toggle Live Alerts from `.env` or the UI top bar when ready.

### 2. Train or Retrain the Multi-Microphone Models
```powershell
.\venv\Scripts\python.exe train.py
```

### 3. Run Cross-Microphone Robustness Benchmark
```powershell
.\venv\Scripts\python.exe evaluate.py
```

### 4. Launch the Next-Gen AI Command Interface
```powershell
.\venv\Scripts\python.exe app.py
```
Open **http://127.0.0.1:5000** in Chrome or Edge:
1. Select your **Bluetooth earphones / headset / mic** from the **Microphone & Adaptive DSP** dropdown.
2. Click **Calibrate Mic (2s)** while staying quiet to lock in your room's ambient floor.
3. Click **Arm Sentinel** — acoustic neural detection and multilingual speech keyword spotting run simultaneously in real time!

---

## Project Structure

```text
├── app.py                     # Hardened Flask + SSE real-time web server
├── config.py                  # Validated .env configuration loader
├── train.py                   # Zero-leakage multi-microphone training pipeline
├── evaluate.py                # Cross-microphone robustness benchmark CLI
├── .env.example               # Safe environment variable template
├── engine/
│   ├── audio_dsp.py           # High-pass, Soft AGC, Adaptive VAD, Multi-Mic Simulator
│   ├── features.py            # 3-Ch Log-Mel Tensor & 142-dim Prosodic Extractor
│   ├── detector.py            # Hybrid CNN + Acoustic Ensemble + Keyword Radar
│   ├── alerts.py              # Non-blocking TLS Email / Twilio SMS + Cancel Window
│   └── security.py            # CSP, CSRF, Rate Limiting, Audio Magic-Byte Validation
├── models/                    # Trained Keras 3 CNN, Acoustic Ensemble & Benchmark Metadata
├── templates/index.html       # Next-Gen AI Sentinel Interface
└── static/
    ├── css/style.css          # Obsidian Glassmorphism + Dynamic Threat Aura
    └── js/app.js              # 60 FPS Neural Orb, Waveform, Spectrogram & WebAudio Engine
```
