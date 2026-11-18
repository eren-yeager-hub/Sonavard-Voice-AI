"""
Aegis Voice AI — Cross-Microphone Robustness Benchmark & Diagnostic Evaluator.

Evaluates the trained production models across 5 simulated microphone profiles
(Clean, Bluetooth HFP Earphones, Wired Earbuds, Low-Gain Distant, Noisy Street)
plus optional user-supplied audio files.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

import config
from engine.audio_dsp import SAMPLE_RATE, resample_audio, simulate_microphone_profile
from engine.detector import DistressDetectorEngine


def evaluate_file_across_mics(detector: DistressDetectorEngine, file_path: Path) -> None:
    y, sr = sf.read(str(file_path), dtype="float32")
    y = resample_audio(y, sr, SAMPLE_RATE)
    rng = np.random.default_rng(42)

    print(f"\nFile: {file_path.name} ({len(y)/SAMPLE_RATE:.2f}s)")
    print("-" * 72)
    for prof in ["clean", "bluetooth_hfp", "wired_earbud", "low_gain_distant", "noisy_street"]:
        sim = simulate_microphone_profile(y, prof, rng)
        detector.reset_temporal_state()
        res = detector.analyze_live_chunk(sim, sr=SAMPLE_RATE)
        status = "DISTRESS" if res["fused_score"] >= res["threshold"] else "NORMAL"
        print(
            f"  {prof:18s} | active={str(res['active']):5s} | raw_rms={res['raw_rms']:.4f} "
            f"| agc={res['agc_gain_db']:+5.1f}dB | CNN={res['cnn_score']:.3f} "
            f"| Ac={res['acoustic_score']:.3f} | Fused={res['fused_score']:.3f} -> {status}"
        )


def main() -> None:
    detector = DistressDetectorEngine()
    if not detector.is_ready:
        print("Models not found. Run `python train.py` first.")
        sys.exit(1)

    print("=" * 72)
    print("AEGIS VOICE AI — MODEL DIAGNOSTICS & MULTI-MIC ROBUSTNESS REPORT")
    print("=" * 72)

    if config.METADATA_PATH.is_file():
        meta = json.loads(config.METADATA_PATH.read_text(encoding="utf-8"))
        print(f"Model Version : {meta.get('model_version')}")
        print(f"Fusion Weights: CNN={meta.get('cnn_weight')} | Acoustic={meta.get('acoustic_weight')}")
        print(f"Thresholds    : {meta.get('thresholds')}")
        print(f"Overall OOF   : {meta.get('overall_oof_metrics')}")
        print("\nOut-of-Fold Cross-Microphone Generalization (Unseen Source Recordings):")
        for prof, m in meta.get("mic_benchmarks", {}).items():
            print(
                f"  {prof:18s} | AUC={m['roc_auc']:.4f} | F1={m['f1']:.4f} "
                f"| Rec={m['recall']:.4f} | Prec={m['precision']:.4f}"
            )

    # Test CLI-specified files, or sample emergency & normal recordings
    if len(sys.argv) > 1:
        targets = [Path(arg) for arg in sys.argv[1:] if Path(arg).is_file()]
    else:
        targets = []
        for candidate in [
            config.BASE_DIR / "segmented_dataset/emergency/10.wav",
            config.BASE_DIR / "processed_dataset/emergency/woman-scream-sound-hd-379381.wav",
            config.BASE_DIR / "processed_dataset/normal/talking-people-6368.wav",
            config.BASE_DIR / "processed_dataset/normal/laughing-man-117725.wav",
        ]:
            if candidate.is_file():
                targets.append(candidate)

    for path in targets:
        evaluate_file_across_mics(detector, path)


if __name__ == "__main__":
    main()
