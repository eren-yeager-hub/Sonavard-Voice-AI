"""
Aegis Voice AI — Zero-Leakage Multi-Microphone Training Pipeline.

Solves dataset overfitting & earphone/Bluetooth mic failure via:
1. Source-level deduplication (eliminates `(1)` / `(2)` duplicate files).
2. Strict StratifiedGroupKFold grouped by source recording (zero data leakage).
3. Multi-microphone domain randomization during training (Bluetooth HFP 300-3400Hz,
   wired earbud EQ, low-gain distant room reverb, real ambient noise mixing,
   pitch/speed perturbation, and synthetic hard negatives).
4. Dual-Branch Hybrid Architecture:
   - Branch A: 3-channel Microphone-Invariant Log-Mel 2D CNN (Keras 3) with SpecAugment
   - Branch B: Prosodic/Acoustic Ensemble (HistGB + ExtraTrees + L2-MLP on RobustScaler)
5. Out-of-fold multi-microphone calibration & threshold optimization.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(max(1, (os.cpu_count() or 4) // 2)))

import joblib
import numpy as np
import soundfile as sf
import tensorflow as tf
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, VotingClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from tensorflow import keras
from tensorflow.keras import layers

import config
from engine.audio_dsp import (
    SAMPLE_RATE,
    TARGET_SAMPLES,
    fit_length,
    highpass_filter,
    perturb_speed_and_shift,
    preprocess_audio,
    resample_audio,
    simulate_microphone_profile,
)
from engine.features import (
    N_MELS,
    SPEC_FRAMES,
    extract_acoustic_features,
    extract_spectrogram_tensor,
)

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

MIC_PROFILES = ["clean", "bluetooth_hfp", "wired_earbud", "low_gain_distant", "noisy_street"]


@dataclass
class SourceClip:
    source_id: str
    label: int  # 1 = emergency, 0 = normal
    audio: np.ndarray


def _canonical_stem(filename: str) -> str:
    stem = Path(filename).stem
    stem = re.sub(r"\s*\(\d+\)\s*$", "", stem).strip().lower()
    return stem


def _audio_fingerprint(audio: np.ndarray) -> str:
    snippet = np.round(audio[: min(audio.size, SAMPLE_RATE * 3)] * 1000).astype(np.int16)
    return hashlib.md5(snippet.tobytes()).hexdigest()


def load_deduplicated_sources() -> list[SourceClip]:
    """Load unique raw recordings from processed_dataset/ plus user's personal recordings."""
    sources: list[SourceClip] = []
    seen_stems: set[tuple[int, str]] = set()
    seen_hashes: set[str] = set()

    def _add_file(path: Path, label: int, prefix: str) -> None:
        stem = _canonical_stem(path.name)
        if (label, stem) in seen_stems:
            return
        try:
            y, sr = sf.read(str(path), dtype="float32")
            y = resample_audio(y, sr, SAMPLE_RATE)
            if y.size < int(SAMPLE_RATE * 0.35):
                return
            rms = float(np.sqrt(np.mean(y * y) + 1e-12))
            if rms < 0.001:
                return
            fp = _audio_fingerprint(y)
            if fp in seen_hashes:
                return
            seen_stems.add((label, stem))
            seen_hashes.add(fp)
            sources.append(SourceClip(source_id=f"{prefix}_{stem}", label=label, audio=y))
        except Exception as exc:
            print(f"  [skip] {path.name}: {exc}", flush=True)

    for folder_name, label in [("emergency", 1), ("normal", 0)]:
        folder = config.BASE_DIR / "processed_dataset" / folder_name
        if not folder.exists():
            continue
        for p in sorted(folder.iterdir()):
            if p.suffix.lower() in {".wav", ".flac", ".ogg"}:
                _add_file(p, label, folder_name)

    seg_em = config.BASE_DIR / "segmented_dataset" / "emergency"
    if seg_em.exists():
        for p in sorted(seg_em.iterdir()):
            name = p.name
            if p.suffix.lower() == ".wav" and (
                (name[0].isdigit() and len(p.stem) <= 3) or name.startswith("WhatsApp")
            ):
                _add_file(p, 1, "personal_em")

    return sources


def slice_source_into_windows(clip: SourceClip, max_windows: int = 10) -> list[np.ndarray]:
    """Slice source recording into active 2.0s windows while preserving short clips (<2s)."""
    y = highpass_filter(clip.audio, SAMPLE_RATE)
    n = y.size

    if n <= TARGET_SAMPLES:
        return [fit_length(y, TARGET_SAMPLES, select_loudest=False)]

    hop = int(SAMPLE_RATE * (1.1 if clip.label == 1 else 2.0))
    candidates: list[np.ndarray] = []
    for start in range(0, n - TARGET_SAMPLES + 1, hop):
        seg = y[start : start + TARGET_SAMPLES]
        rms = float(np.sqrt(np.mean(seg * seg) + 1e-12))
        if rms >= 0.003:
            candidates.append(seg)

    if not candidates:
        candidates.append(fit_length(y, TARGET_SAMPLES, select_loudest=True))

    if len(candidates) > max_windows:
        idxs = np.linspace(0, len(candidates) - 1, max_windows, dtype=int)
        return [candidates[i] for i in idxs]

    return candidates


def generate_synthetic_hard_negatives(
    n_samples: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Synthetic microphone artifacts that must NEVER trigger emergencies:
    headset thermal hiss, 50/60Hz electrical hum, cable tap impulses, steady fan drones.
    """
    specs, acs = [], []
    t = np.arange(TARGET_SAMPLES, dtype=np.float32) / SAMPLE_RATE

    for i in range(n_samples):
        mode = i % 4
        if mode == 0:
            y = rng.normal(0.0, float(rng.uniform(0.005, 0.08)), size=TARGET_SAMPLES).astype(np.float32)
        elif mode == 1:
            f0 = 50.0 if rng.random() < 0.5 else 60.0
            y = (
                0.04 * np.sin(2 * np.pi * f0 * t)
                + 0.02 * np.sin(2 * np.pi * 2 * f0 * t)
                + 0.01 * np.sin(2 * np.pi * 3 * f0 * t)
                + rng.normal(0.0, 0.006, size=TARGET_SAMPLES)
            ).astype(np.float32)
        elif mode == 2:
            y = rng.normal(0.0, 0.004, size=TARGET_SAMPLES).astype(np.float32)
            pos = int(rng.integers(2000, TARGET_SAMPLES - 2000))
            width = int(rng.integers(80, 450))
            decay = np.exp(-np.linspace(0, 6, width, dtype=np.float32))
            y[pos : pos + width] += rng.normal(0.0, 0.25, size=width).astype(np.float32) * decay
        else:
            freq = float(rng.uniform(110.0, 320.0))
            y = (
                0.05 * np.sin(2 * np.pi * freq * t)
                + 0.02 * np.sin(2 * np.pi * freq * 2.0 * t)
                + rng.normal(0.0, 0.012, size=TARGET_SAMPLES)
            ).astype(np.float32)

        proc, _ = preprocess_audio(y, SAMPLE_RATE, TARGET_SAMPLES)
        specs.append(extract_spectrogram_tensor(proc, SAMPLE_RATE))
        acs.append(extract_acoustic_features(proc, SAMPLE_RATE))

    return (
        np.stack(specs, axis=0).astype(np.float32),
        np.stack(acs, axis=0).astype(np.float32),
        np.zeros(n_samples, dtype=np.int32),
    )


def precompute_source_features(sources: list[SourceClip], rng: np.random.Generator) -> dict:
    """
    Precompute per-source multi-microphone feature banks ONCE so 3-fold CV & final training
    have zero redundant feature extraction while preserving strict source-level grouping.
    """
    t0 = time.time()
    # Ambient noise pool built from normal sources (only synthetic noise used during cross-mic eval)
    ambient_pool = [
        fit_length(s.audio, TARGET_SAMPLES, select_loudest=False)
        for s in sources
        if s.label == 0
    ]

    # Store per-source index lists for train (augmented) and eval (per-mic profile)
    train_bank = {}  # src_idx -> (X_spec, X_ac, y)
    eval_bank = {}   # src_idx -> {prof: (X_spec, X_ac, y)}

    for idx, clip in enumerate(sources):
        max_win = 10 if clip.label == 1 else 6
        windows = slice_source_into_windows(clip, max_windows=max_win)

        tr_specs, tr_acs, tr_ys = [], [], []
        ev_dict = {p: ([], [], []) for p in MIC_PROFILES}

        for win in windows:
            # 1. Per-mic evaluation views (deterministic & unperturbed timing)
            for prof in MIC_PROFILES:
                # For eval, avoid mixing other recordings into noisy_street to ensure zero cross-source bleed
                sim = simulate_microphone_profile(win, prof, rng, ambient_pool=None)
                proc, _ = preprocess_audio(sim, SAMPLE_RATE, TARGET_SAMPLES)
                ev_dict[prof][0].append(extract_spectrogram_tensor(proc, SAMPLE_RATE))
                ev_dict[prof][1].append(extract_acoustic_features(proc, SAMPLE_RATE))
                ev_dict[prof][2].append(clip.label)

            # 2. Rich multi-mic training views (with pitch/speed perturbation + mic profiles + ambient mixing)
            tr_profiles = (
                ["clean", "bluetooth_hfp", "wired_earbud", "low_gain_distant", "noisy_street"]
                if clip.label == 1
                else ["clean", "bluetooth_hfp", "wired_earbud"]
            )
            for prof in tr_profiles:
                aug = win if prof == "clean" else perturb_speed_and_shift(win, rng)
                if prof != "clean":
                    aug = simulate_microphone_profile(aug, prof, rng, ambient_pool=ambient_pool)
                proc, _ = preprocess_audio(aug, SAMPLE_RATE, TARGET_SAMPLES)
                tr_specs.append(extract_spectrogram_tensor(proc, SAMPLE_RATE))
                tr_acs.append(extract_acoustic_features(proc, SAMPLE_RATE))
                tr_ys.append(clip.label)

        train_bank[idx] = (
            np.stack(tr_specs, axis=0).astype(np.float32),
            np.stack(tr_acs, axis=0).astype(np.float32),
            np.array(tr_ys, dtype=np.int32),
        )
        eval_bank[idx] = {
            p: (
                np.stack(ev_dict[p][0], axis=0).astype(np.float32),
                np.stack(ev_dict[p][1], axis=0).astype(np.float32),
                np.array(ev_dict[p][2], dtype=np.int32),
            )
            for p in MIC_PROFILES
        }

        if (idx + 1) % 20 == 0 or (idx + 1) == len(sources):
            print(f"  Featurized {idx + 1}/{len(sources)} source recordings ({time.time() - t0:.1f}s)...", flush=True)

    return {"train": train_bank, "eval": eval_bank}


class SpecAugmentLayer(layers.Layer):
    """Lightweight SpecAugment (frequency & time masking) active only during training."""

    def __init__(self, freq_mask_max: int = 8, time_mask_max: int = 10, **kwargs):
        super().__init__(**kwargs)
        self.freq_mask_max = freq_mask_max
        self.time_mask_max = time_mask_max

    def call(self, x, training=None):
        if not training:
            return x
        shape = tf.shape(x)
        f_dim = shape[1]
        t_dim = shape[2]

        f_w = tf.random.uniform([], 0, self.freq_mask_max + 1, dtype=tf.int32)
        f0 = tf.random.uniform([], 0, tf.maximum(1, f_dim - f_w), dtype=tf.int32)
        f_idx = tf.range(f_dim)
        f_mask = tf.cast((f_idx < f0) | (f_idx >= f0 + f_w), x.dtype)
        f_mask = tf.reshape(f_mask, [1, f_dim, 1, 1])

        t_w = tf.random.uniform([], 0, self.time_mask_max + 1, dtype=tf.int32)
        t0 = tf.random.uniform([], 0, tf.maximum(1, t_dim - t_w), dtype=tf.int32)
        t_idx = tf.range(t_dim)
        t_mask = tf.cast((t_idx < t0) | (t_idx >= t0 + t_w), x.dtype)
        t_mask = tf.reshape(t_mask, [1, 1, t_dim, 1])

        return x * f_mask * t_mask

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"freq_mask_max": self.freq_mask_max, "time_mask_max": self.time_mask_max})
        return cfg


def build_cnn_model() -> keras.Model:
    """Compact, CPU-fast, microphone-invariant 2D Spectrogram CNN with Dual Pooling."""
    reg = keras.regularizers.l2(1.5e-4)
    inp = layers.Input(shape=(N_MELS, SPEC_FRAMES, 3), name="log_mel_3ch")
    x = SpecAugmentLayer(freq_mask_max=8, time_mask_max=10, name="spec_augment")(inp)

    x = layers.Conv2D(24, (3, 3), padding="same", use_bias=False, kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("swish")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.SpatialDropout2D(0.12)(x)

    x = layers.SeparableConv2D(
        48, (3, 3), padding="same", use_bias=False, depthwise_regularizer=reg, pointwise_regularizer=reg
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("swish")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.SpatialDropout2D(0.18)(x)

    x = layers.SeparableConv2D(
        80, (3, 3), padding="same", use_bias=False, depthwise_regularizer=reg, pointwise_regularizer=reg
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("swish")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.SpatialDropout2D(0.22)(x)

    x = layers.Conv2D(96, (3, 3), padding="same", use_bias=False, kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("swish")(x)

    avg_p = layers.GlobalAveragePooling2D()(x)
    max_p = layers.GlobalMaxPooling2D()(x)
    x = layers.Concatenate()([avg_p, max_p])

    x = layers.Dropout(0.40)(x)
    x = layers.Dense(64, activation="swish", kernel_regularizer=reg)(x)
    x = layers.Dropout(0.30)(x)
    out = layers.Dense(1, activation="sigmoid", name="distress_prob")(x)

    model = keras.Model(inputs=inp, outputs=out, name="aegis_distress_cnn_v3")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1.3e-3),
        loss=keras.losses.BinaryCrossentropy(label_smoothing=0.05),
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )
    return model


def build_acoustic_ensemble() -> Pipeline:
    """Microphone-invariant prosodic/acoustic soft-voting ensemble."""
    hgb = HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.05,
        max_depth=5,
        min_samples_leaf=18,
        l2_regularization=1.5,
        class_weight="balanced",
        random_state=SEED,
    )
    et = ExtraTreesClassifier(
        n_estimators=240,
        max_depth=16,
        min_samples_leaf=4,
        max_features="sqrt",
        class_weight="balanced",
        n_jobs=4,
        random_state=SEED,
    )
    mlp = MLPClassifier(
        hidden_layer_sizes=(96, 48),
        activation="relu",
        alpha=0.06,
        batch_size=64,
        learning_rate_init=1e-3,
        max_iter=120,
        early_stopping=True,
        validation_fraction=0.15,
        random_state=SEED,
    )
    voting = VotingClassifier(
        estimators=[("hgb", hgb), ("et", et), ("mlp", mlp)],
        voting="soft",
        weights=[2.0, 1.5, 1.0],
    )
    return Pipeline([("scaler", RobustScaler()), ("clf", voting)])


def _compute_class_weights(y: np.ndarray) -> dict[int, float]:
    n = len(y)
    n_pos = max(1, int(np.sum(y == 1)))
    n_neg = max(1, int(np.sum(y == 0)))
    return {0: n / (2.0 * n_neg), 1: n / (2.0 * n_pos)}


def _gather_train(bank: dict, idxs: np.ndarray, hn: tuple[np.ndarray, np.ndarray, np.ndarray]):
    xs = np.concatenate([bank["train"][int(i)][0] for i in idxs] + [hn[0]], axis=0)
    xa = np.concatenate([bank["train"][int(i)][1] for i in idxs] + [hn[1]], axis=0)
    y = np.concatenate([bank["train"][int(i)][2] for i in idxs] + [hn[2]], axis=0)
    return xs, xa, y


def _gather_eval(bank: dict, idxs: np.ndarray, prof: str):
    xs = np.concatenate([bank["eval"][int(i)][prof][0] for i in idxs], axis=0)
    xa = np.concatenate([bank["eval"][int(i)][prof][1] for i in idxs], axis=0)
    y = np.concatenate([bank["eval"][int(i)][prof][2] for i in idxs], axis=0)
    return xs, xa, y


def _tune_fusion_and_threshold(
    p_cnn: np.ndarray,
    p_ac: np.ndarray,
    y_true: np.ndarray,
) -> tuple[float, float, dict[str, float]]:
    best_score = -1.0
    best_w = 0.5
    best_thr = 0.5

    for w in np.linspace(0.25, 0.75, 11):
        p = w * p_cnn + (1.0 - w) * p_ac
        for thr in np.linspace(0.45, 0.72, 28):
            preds = (p >= thr).astype(int)
            f1 = f1_score(y_true, preds, zero_division=0)
            rec = recall_score(y_true, preds, zero_division=0)
            prec = precision_score(y_true, preds, zero_division=0)
            objective = 0.65 * f1 + 0.15 * rec + 0.20 * prec
            if objective > best_score:
                best_score = objective
                best_w = float(w)
                best_thr = float(thr)

    p_best = best_w * p_cnn + (1.0 - best_w) * p_ac
    preds = (p_best >= best_thr).astype(int)
    metrics = {
        "roc_auc": round(float(roc_auc_score(y_true, p_best)), 4),
        "accuracy": round(float(accuracy_score(y_true, preds)), 4),
        "precision": round(float(precision_score(y_true, preds, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, preds, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, preds, zero_division=0)), 4),
    }
    return best_w, best_thr, metrics


def main() -> None:
    t0 = time.time()
    print("=" * 68, flush=True)
    print("AEGIS VOICE AI — ZERO-LEAKAGE MULTI-MICROPHONE TRAINING PIPELINE", flush=True)
    print("=" * 68, flush=True)

    sources = load_deduplicated_sources()
    n_em = sum(1 for s in sources if s.label == 1)
    n_no = sum(1 for s in sources if s.label == 0)
    print(f"Loaded {len(sources)} unique source recordings ({n_em} emergency, {n_no} normal)", flush=True)

    rng = np.random.default_rng(SEED)
    print("\n[Stage 1] Extracting multi-microphone feature banks...", flush=True)
    bank = precompute_source_features(sources, rng)
    hn_bundle = generate_synthetic_hard_negatives(90, np.random.default_rng(SEED + 77))

    groups = np.array([s.source_id for s in sources])
    y_sources = np.array([s.label for s in sources])

    # ------------------------------------------------------------------
    # STAGE 2: Honest 3-Fold Source-Grouped Cross-Validation
    # ------------------------------------------------------------------
    sgkf = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED)
    oof_cnn: dict[str, list[np.ndarray]] = {p: [] for p in MIC_PROFILES}
    oof_ac: dict[str, list[np.ndarray]] = {p: [] for p in MIC_PROFILES}
    oof_y: dict[str, list[np.ndarray]] = {p: [] for p in MIC_PROFILES}

    print("\n[Stage 2] Running 3-Fold Source-Grouped CV across 5 Microphone Profiles...", flush=True)
    for fold, (tr_idx, va_idx) in enumerate(sgkf.split(sources, y_sources, groups), start=1):
        f_t0 = time.time()
        X_spec_tr, X_ac_tr, y_tr = _gather_train(bank, tr_idx, hn_bundle)

        val_mics = {p: _gather_eval(bank, va_idx, p) for p in MIC_PROFILES}
        X_spec_va = np.concatenate([val_mics[p][0] for p in ["clean", "bluetooth_hfp", "wired_earbud"]], axis=0)
        y_va = np.concatenate([val_mics[p][2] for p in ["clean", "bluetooth_hfp", "wired_earbud"]], axis=0)

        print(
            f"\n--- Fold {fold}/3 | Train samples: {len(y_tr)} (pos={int(y_tr.sum())}) "
            f"| Unseen Val sources: {len(va_idx)} ---",
            flush=True,
        )

        ac_model = build_acoustic_ensemble()
        ac_model.fit(X_ac_tr, y_tr)

        keras.backend.clear_session()
        cnn_model = build_cnn_model()
        cnn_model.fit(
            X_spec_tr,
            y_tr,
            validation_data=(X_spec_va, y_va),
            epochs=22,
            batch_size=48,
            class_weight=_compute_class_weights(y_tr),
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor="val_auc", mode="max", patience=5, restore_best_weights=True
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor="val_auc", mode="max", factor=0.5, patience=2, min_lr=1e-5
                ),
            ],
            verbose=0,
        )

        for prof in MIC_PROFILES:
            xs, xa, yv = val_mics[prof]
            pc = cnn_model.predict(xs, batch_size=64, verbose=0).ravel()
            pa = ac_model.predict_proba(xa)[:, 1]
            oof_cnn[prof].append(pc)
            oof_ac[prof].append(pa)
            oof_y[prof].append(yv)

        fold_clean_auc = roc_auc_score(val_mics["clean"][2], 0.5 * oof_cnn["clean"][-1] + 0.5 * oof_ac["clean"][-1])
        fold_bt_auc = roc_auc_score(
            val_mics["bluetooth_hfp"][2],
            0.5 * oof_cnn["bluetooth_hfp"][-1] + 0.5 * oof_ac["bluetooth_hfp"][-1],
        )
        fold_eb_auc = roc_auc_score(
            val_mics["wired_earbud"][2],
            0.5 * oof_cnn["wired_earbud"][-1] + 0.5 * oof_ac["wired_earbud"][-1],
        )
        print(
            f"  Fold {fold} done ({time.time()-f_t0:.1f}s) | Unseen ROC-AUC -> "
            f"Clean: {fold_clean_auc:.4f} | Bluetooth HFP: {fold_bt_auc:.4f} | Wired Earbud: {fold_eb_auc:.4f}",
            flush=True,
        )

    all_cnn = np.concatenate([np.concatenate(oof_cnn[p]) for p in MIC_PROFILES])
    all_ac = np.concatenate([np.concatenate(oof_ac[p]) for p in MIC_PROFILES])
    all_y = np.concatenate([np.concatenate(oof_y[p]) for p in MIC_PROFILES])

    best_w, best_thr, overall_metrics = _tune_fusion_and_threshold(all_cnn, all_ac, all_y)

    print("\n" + "=" * 68, flush=True)
    print("OUT-OF-FOLD CROSS-MICROPHONE GENERALIZATION BENCHMARK (UNSEEN FILES)", flush=True)
    print("=" * 68, flush=True)
    print(f"Optimal Fusion: {best_w:.2f} * CNN + {1.0 - best_w:.2f} * AcousticEnsemble | Threshold: {best_thr:.2f}", flush=True)
    print(f"All-Mics Combined -> {overall_metrics}", flush=True)

    mic_benchmarks: dict[str, dict[str, float]] = {}
    for prof in MIC_PROFILES:
        pc = np.concatenate(oof_cnn[prof])
        pa = np.concatenate(oof_ac[prof])
        yv = np.concatenate(oof_y[prof])
        pf = best_w * pc + (1.0 - best_w) * pa
        preds = (pf >= best_thr).astype(int)
        m = {
            "roc_auc": round(float(roc_auc_score(yv, pf)), 4),
            "accuracy": round(float(accuracy_score(yv, preds)), 4),
            "precision": round(float(precision_score(yv, preds, zero_division=0)), 4),
            "recall": round(float(recall_score(yv, preds, zero_division=0)), 4),
            "f1": round(float(f1_score(yv, preds, zero_division=0)), 4),
            "cnn_auc": round(float(roc_auc_score(yv, pc)), 4),
            "acoustic_auc": round(float(roc_auc_score(yv, pa)), 4),
        }
        mic_benchmarks[prof] = m
        print(
            f"  {prof:18s} | AUC={m['roc_auc']:.4f} (CNN={m['cnn_auc']:.3f}, Ac={m['acoustic_auc']:.3f}) "
            f"| F1={m['f1']:.4f} | Rec={m['recall']:.4f} | Prec={m['precision']:.4f}",
            flush=True,
        )

    # ------------------------------------------------------------------
    # STAGE 3: Train Final Production Models on Full Deduplicated Dataset
    # ------------------------------------------------------------------
    print("\n[Stage 3] Training Final Production Models...", flush=True)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    all_idxs = np.arange(len(sources))
    _, X_ac_all, y_all = _gather_train(bank, all_idxs, hn_bundle)
    print(f"Final Acoustic Ensemble training samples: {len(y_all)} (pos={int(y_all.sum())})", flush=True)

    final_ac = build_acoustic_ensemble()
    final_ac.fit(X_ac_all, y_all)
    joblib.dump(final_ac, config.ENSEMBLE_MODEL_PATH, compress=3)
    print(f"Saved Acoustic Ensemble -> {config.ENSEMBLE_MODEL_PATH}", flush=True)

    # Final CNN trained on 88% grouped sources, monitored on 12% unseen multi-mic sources
    sg_final = StratifiedGroupKFold(n_splits=8, shuffle=True, random_state=SEED + 7)
    tr_idx, va_idx = next(sg_final.split(sources, y_sources, groups))
    X_spec_tr, _, y_tr = _gather_train(bank, tr_idx, hn_bundle)
    X_spec_va = np.concatenate(
        [_gather_eval(bank, va_idx, p)[0] for p in ["clean", "bluetooth_hfp", "wired_earbud"]], axis=0
    )
    y_va = np.concatenate(
        [_gather_eval(bank, va_idx, p)[2] for p in ["clean", "bluetooth_hfp", "wired_earbud"]], axis=0
    )

    keras.backend.clear_session()
    final_cnn = build_cnn_model()
    final_cnn.fit(
        X_spec_tr,
        y_tr,
        validation_data=(X_spec_va, y_va),
        epochs=26,
        batch_size=48,
        class_weight=_compute_class_weights(y_tr),
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_auc", mode="max", patience=6, restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_auc", mode="max", factor=0.5, patience=3, min_lr=1e-5
            ),
        ],
        verbose=1,
    )
    final_cnn.save(config.CNN_MODEL_PATH)
    print(f"Saved 2D Spectrogram CNN -> {config.CNN_MODEL_PATH}", flush=True)

    metadata = {
        "model_version": "aegis-v3.0-multimic",
        "trained_at_epoch": int(time.time()),
        "training_seconds": round(time.time() - t0, 1),
        "sample_rate": SAMPLE_RATE,
        "window_seconds": 2.0,
        "unique_source_recordings": {"emergency": n_em, "normal": n_no, "total": len(sources)},
        "augmented_train_samples": int(len(y_all)),
        "cnn_weight": round(best_w, 3),
        "acoustic_weight": round(1.0 - best_w, 3),
        "thresholds": {
            "high_sensitivity": round(max(0.32, best_thr - 0.10), 3),
            "balanced": round(best_thr, 3),
            "strict": round(min(0.85, best_thr + 0.10), 3),
        },
        "overall_oof_metrics": overall_metrics,
        "mic_benchmarks": mic_benchmarks,
    }
    config.METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved Model Metadata -> {config.METADATA_PATH}", flush=True)
    print(f"\nTraining complete in {time.time() - t0:.1f}s!", flush=True)


if __name__ == "__main__":
    main()
