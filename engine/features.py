"""
Microphone-Invariant Dual Feature Extraction for Sonavard Voice AI.

1. Spectrogram Tensor (64, 63, 3) for the 2D CNN:
   - Ch 0: Per-clip Z-normalized Log-Mel Spectrogram (120 Hz - 6500 Hz)
   - Ch 1: Time-Delta Log-Mel (100% invariant to static microphone EQ H(f)!)
   - Ch 2: Per-frame Spectral Shape (zero-mean per time frame, invariant to gain envelope)

2. Acoustic & Prosodic Feature Vector (~140 features) for the Acoustic Ensemble:
   - CMVN MFCCs (C1..C20, dropping C0) dispersion & percentiles
   - MFCC Delta & Delta-Delta dynamics (EQ-invariant)
   - Spectral Contrast (6 bands — harmonic peak-to-valley scream signature)
   - Spectral Flatness, Flux, Roll-off, Centroid, ZCR
   - Temporal Envelope dynamics (Crest factor, RMS dynamic range, Onset strength)
   - Universal Earphone-Band Sub-Band Energy Ratios (300-3400 Hz)
   - Pitch / F0 vocal stress & instability metrics (fast autocorrelation pitch tracker)
"""

from __future__ import annotations

import math

import librosa
import numpy as np

from engine.audio_dsp import SAMPLE_RATE, TARGET_SAMPLES, fit_length

N_MELS = 64
N_FFT = 512
HOP_LENGTH = 512
FMIN = 120.0
FMAX = 6500.0
SPEC_FRAMES = 1 + TARGET_SAMPLES // HOP_LENGTH  # 63 frames for 32000 samples / 512


def extract_spectrogram_tensor(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Build a 3-channel microphone-invariant Log-Mel tensor of shape (64, 63, 3).
    Assumes audio has already been preprocessed with engine.audio_dsp.preprocess_audio.
    """
    y = fit_length(np.asarray(audio, dtype=np.float32), TARGET_SAMPLES, select_loudest=False)

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
        power=2.0,
    )
    log_mel = np.log(mel + 1e-6).astype(np.float32)

    # Ensure exact frame count
    if log_mel.shape[1] < SPEC_FRAMES:
        log_mel = np.pad(log_mel, ((0, 0), (0, SPEC_FRAMES - log_mel.shape[1])), mode="edge")
    else:
        log_mel = log_mel[:, :SPEC_FRAMES]

    # Channel 0: Per-clip standardized Log-Mel (removes global gain offset)
    ch0 = (log_mel - np.mean(log_mel)) / (np.std(log_mel) + 1e-6)
    ch0 = np.clip(ch0, -4.0, 4.0)

    # Channel 1: Time-Delta Log-Mel (mathematically cancels static mic frequency response!)
    delta = librosa.feature.delta(log_mel, width=5)
    ch1 = (delta - np.mean(delta)) / (np.std(delta) + 1e-6)
    ch1 = np.clip(ch1, -4.0, 4.0)

    # Channel 2: Per-frame spectral shape (normalizes each time slice across frequency)
    frame_mean = np.mean(log_mel, axis=0, keepdims=True)
    frame_std = np.std(log_mel, axis=0, keepdims=True) + 1e-6
    ch2 = np.clip((log_mel - frame_mean) / frame_std, -4.0, 4.0)

    tensor = np.stack([ch0, ch1, ch2], axis=-1).astype(np.float32)
    return np.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)


def _fast_pitch_stats(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Fast frame-level autocorrelation pitch (F0) tracker focused on human vocal range (90-1100 Hz).
    Captures elevated pitch, pitch instability, and scream/cry harmonicity cleanly on CPU.
    """
    frame_len = 512
    hop = 256
    min_lag = max(1, int(sr / 1100.0))  # ~14 samples (1100 Hz high scream)
    max_lag = min(frame_len - 2, int(sr / 90.0))  # ~177 samples (90 Hz deep voice)

    if audio.size < frame_len:
        return np.zeros(5, dtype=np.float32)

    frames = librosa.util.frame(audio, frame_length=frame_len, hop_length=hop)  # (frame_len, n_frames)
    window = np.hanning(frame_len).astype(np.float32)[:, None]
    wframes = frames * window

    # Vectorized FFT autocorrelation
    n_fft = 1024
    fft = np.fft.rfft(wframes, n=n_fft, axis=0)
    ac = np.fft.irfft(fft * np.conj(fft), n=n_fft, axis=0)[:frame_len, :]
    energy = ac[0, :] + 1e-10
    ac_norm = ac[min_lag : max_lag + 1, :] / energy[None, :]

    best_idx = np.argmax(ac_norm, axis=0)
    best_corr = ac_norm[best_idx, np.arange(ac_norm.shape[1])]
    lags = best_idx + min_lag

    # Frame energy gate so silence isn't counted as voiced
    frame_rms = np.sqrt(np.mean(frames * frames, axis=0) + 1e-12)
    rms_thresh = max(0.015, float(np.median(frame_rms)) * 0.5)
    voiced = (best_corr > 0.32) & (frame_rms > rms_thresh)

    voiced_frac = float(np.mean(voiced))
    if np.sum(voiced) < 3:
        return np.array([voiced_frac, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    f0_hz = sr / lags[voiced].astype(np.float32)
    f0_log = np.log2(np.clip(f0_hz, 80.0, 1200.0) / 200.0)  # centered around 200 Hz

    f0_med = float(np.median(f0_log))
    f0_p90 = float(np.percentile(f0_log, 90))
    f0_std = float(np.std(f0_log))
    f0_jump = float(np.mean(np.abs(np.diff(f0_log)))) if f0_log.size > 1 else 0.0

    return np.array([voiced_frac, f0_med, f0_p90, f0_std, f0_jump], dtype=np.float32)


def extract_acoustic_features(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Extract microphone-invariant prosodic, spectral, and dynamic features (142 dims).
    """
    y = fit_length(np.asarray(audio, dtype=np.float32), TARGET_SAMPLES, select_loudest=False)

    # STFT magnitude & power
    stft = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)) + 1e-8
    power = stft * stft
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)

    # 1. Mel & CMVN-Normalized MFCCs (drop C0 raw energy!)
    mel = librosa.feature.melspectrogram(S=power, sr=sr, n_mels=N_MELS, fmin=FMIN, fmax=FMAX)
    log_mel = librosa.power_to_db(mel + 1e-10)
    mfcc = librosa.feature.mfcc(S=log_mel, n_mfcc=21)[1:, :]  # C1..C20 -> (20, T)

    # Per-clip CMN on MFCCs removes constant channel transfer function!
    mfcc_cmn = mfcc - np.mean(mfcc, axis=1, keepdims=True)
    mfcc_std = np.std(mfcc_cmn, axis=1)
    mfcc_iqr = np.percentile(mfcc_cmn, 75, axis=1) - np.percentile(mfcc_cmn, 25, axis=1)
    mfcc_p90 = np.percentile(np.abs(mfcc_cmn), 90, axis=1)

    # 2. MFCC Deltas & Delta-Deltas (C1..C13) — completely static-EQ invariant
    delta = librosa.feature.delta(mfcc[:13, :], width=5)
    delta2 = librosa.feature.delta(mfcc[:13, :], order=2, width=5)
    delta_std = np.std(delta, axis=1)
    delta_p90 = np.percentile(np.abs(delta), 90, axis=1)
    delta2_std = np.std(delta2, axis=1)

    # 3. Spectral Contrast (6 bands: peak-to-valley harmonic ratio)
    contrast = librosa.feature.spectral_contrast(S=stft, sr=sr, fmin=150.0, n_bands=5)  # (6, T)
    contrast_mean = np.mean(contrast, axis=1)
    contrast_std = np.std(contrast, axis=1)

    # 4. Spectral Flatness, Flux, Roll-off, Centroid, ZCR
    flatness = librosa.feature.spectral_flatness(S=stft)[0]
    rolloff = librosa.feature.spectral_rolloff(S=stft, sr=sr, roll_percent=0.85)[0] / (0.5 * sr)
    centroid = librosa.feature.spectral_centroid(S=stft, sr=sr)[0] / (0.5 * sr)
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=N_FFT, hop_length=HOP_LENGTH)[0]

    # Normalized spectral flux
    stft_norm = stft / (np.sum(stft, axis=0, keepdims=True) + 1e-8)
    flux = np.sqrt(np.sum(np.diff(stft_norm, axis=1) ** 2, axis=0))

    spec_stats = np.array(
        [
            np.mean(flatness),
            np.std(flatness),
            np.percentile(flatness, 90),
            np.mean(flux),
            np.std(flux),
            np.percentile(flux, 95),
            np.mean(rolloff),
            np.std(rolloff),
            np.mean(centroid),
            np.std(centroid),
            np.percentile(centroid, 90),
            np.mean(zcr),
            np.std(zcr),
            np.percentile(zcr, 90),
        ],
        dtype=np.float32,
    )

    # 5. Gain-Invariant Temporal Envelope Dynamics
    rms_frames = librosa.feature.rms(S=stft, frame_length=N_FFT)[0] + 1e-8
    rms_db = 20.0 * np.log10(rms_frames / (np.max(rms_frames) + 1e-8))
    overall_rms = float(np.sqrt(np.mean(y * y) + 1e-10))
    crest_factor = float(np.max(np.abs(y))) / max(overall_rms, 1e-6)
    rms_dyn_range = float(np.percentile(rms_db, 95) - np.percentile(rms_db, 10))
    rms_diff_std = float(np.std(np.diff(rms_db)))
    active_frac = float(np.mean(rms_db > -20.0))

    onset_env = librosa.onset.onset_strength(S=log_mel, sr=sr)
    env_stats = np.array(
        [
            crest_factor,
            rms_dyn_range,
            rms_diff_std,
            active_frac,
            float(np.mean(onset_env)),
            float(np.std(onset_env)),
            float(np.percentile(onset_env, 95)),
        ],
        dtype=np.float32,
    )

    # 6. Universal Earphone / Bluetooth HFP Band (300 - 3400 Hz) Energy Ratios
    def _band_energy(lo: float, hi: float) -> float:
        mask = (freqs >= lo) & (freqs < hi)
        return float(np.sum(power[mask, :]))

    e_total = _band_energy(300.0, 3400.0) + 1e-10
    e_low = _band_energy(300.0, 800.0) / e_total
    e_mid1 = _band_energy(800.0, 1600.0) / e_total
    e_mid2 = _band_energy(1600.0, 2500.0) / e_total
    e_hi = _band_energy(2500.0, 3400.0) / e_total
    e_scream_ratio = (_band_energy(1200.0, 3400.0) + 1e-10) / (_band_energy(300.0, 1200.0) + 1e-10)

    band_stats = np.array(
        [e_low, e_mid1, e_mid2, e_hi, math.log1p(min(e_scream_ratio, 50.0))],
        dtype=np.float32,
    )

    # 7. Pitch / F0 Stress Indicators
    pitch_stats = _fast_pitch_stats(y, sr=sr)

    vec = np.concatenate(
        [
            mfcc_std,        # 20
            mfcc_iqr,        # 20
            mfcc_p90,        # 20
            delta_std,       # 13
            delta_p90,       # 13
            delta2_std,      # 13
            contrast_mean,   # 6
            contrast_std,    # 6
            spec_stats,      # 14
            env_stats,       # 7
            band_stats,      # 5
            pitch_stats,     # 5
        ]
    ).astype(np.float32)

    return np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)

