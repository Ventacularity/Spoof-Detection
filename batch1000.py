"""
batch1000.py - Run inference on full VCTK dataset
EE 123 Project - Ved, Ching, Erick

Tests the saved model.pkl against all 3000 samples:
  - 1000 source (real)
  - 2000 target (real)  (actually 1000 but numbered 1001-2000)
  - 1000 converted (spoof)

Prints a summary at the end — no per-file output to avoid flooding terminal.

Usage:
    python batch1000.py
"""

import os
import glob
import numpy as np
import joblib
from ching import build_advanced_feature_vector

AUDIO_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLES_DIR = os.path.join(AUDIO_DIR, "Samples")

SOURCE_DIR    = os.path.join(SAMPLES_DIR, "source-vctk1k")
TARGET_DIR    = os.path.join(SAMPLES_DIR, "target-vctk1k")
CONVERTED_DIR = os.path.join(SAMPLES_DIR, "converted-vctk1k")

THRESHOLD = 0.45


def run_folder(folder, true_label, label_str, clf, scaler, sel_idx, phys_start):
    files = sorted(glob.glob(os.path.join(folder, "*.wav")))
    correct = 0
    total = 0
    errors = 0
    prob_list = []

    for i, filepath in enumerate(files):
        if (i + 1) % 100 == 0:
            print(f"    [{label_str}] {i+1}/{len(files)}...")
        try:
            fv = build_advanced_feature_vector(filepath)
            physics = fv[phys_start:]
            x = physics[sel_idx].reshape(1, -1)
            x_scaled = scaler.transform(x)
            prob_spoof = clf.predict_proba(x_scaled)[0][1]
            verdict = "SPOOF" if prob_spoof >= THRESHOLD else "REAL"
            prob_list.append(prob_spoof)
            if verdict == true_label:
                correct += 1
            total += 1
        except Exception as e:
            errors += 1

    return correct, total, errors, prob_list


def main():
    print("=" * 60)
    print("   batch1000.py -- Full Dataset Evaluation")
    print("   EE 123 Project -- Ved, Ching, Erick")
    print("=" * 60)

    # Load model
    model_path = os.path.join(AUDIO_DIR, "model.pkl")
    loaded = joblib.load(model_path)
    if len(loaded) == 4:
        clf, scaler, sel_idx, phys_start = loaded
    else:
        clf, scaler = loaded
        sel_idx = [0, 1, 3]
        phys_start = 120

    print(f"\n  Threshold: {THRESHOLD}")
    print(f"  Model: {model_path}\n")

    # Run on each folder
    print("[1/3] Testing source (real)...")
    src_correct, src_total, src_err, src_probs = run_folder(
        SOURCE_DIR, "REAL", "REAL/source", clf, scaler, sel_idx, phys_start)

    print("\n[2/3] Testing target (real)...")
    tgt_correct, tgt_total, tgt_err, tgt_probs = run_folder(
        TARGET_DIR, "REAL", "REAL/target", clf, scaler, sel_idx, phys_start)

    print("\n[3/3] Testing converted (spoof)...")
    spf_correct, spf_total, spf_err, spf_probs = run_folder(
        CONVERTED_DIR, "SPOOF", "SPOOF", clf, scaler, sel_idx, phys_start)

    # Summary
    real_correct = src_correct + tgt_correct
    real_total = src_total + tgt_total
    spoof_correct = spf_correct
    spoof_total = spf_total
    total_correct = real_correct + spoof_correct
    total = real_total + spoof_total

    print("\n" + "=" * 60)
    print("   RESULTS SUMMARY")
    print("=" * 60)
    print(f"  Source (real) : {src_correct:>5}/{src_total} correct  ({src_correct/src_total*100:.1f}%)  [{src_err} errors]")
    print(f"  Target (real) : {tgt_correct:>5}/{tgt_total} correct  ({tgt_correct/tgt_total*100:.1f}%)  [{tgt_err} errors]")
    print(f"  Real total    : {real_correct:>5}/{real_total} correct  ({real_correct/real_total*100:.1f}%)")
    print(f"  Converted     : {spoof_correct:>5}/{spoof_total} correct  ({spoof_correct/spoof_total*100:.1f}%)  [{spf_err} errors]")
    print("-" * 60)
    print(f"  OVERALL       : {total_correct:>5}/{total} correct  ({total_correct/total*100:.1f}%)")
    print("=" * 60)

    # Probability distributions
    print("\n  Avg spoof probability:")
    print(f"    Real  (source) : {np.mean(src_probs):.3f}")
    print(f"    Real  (target) : {np.mean(tgt_probs):.3f}")
    print(f"    Spoof          : {np.mean(spf_probs):.3f}")
    print(f"\n  (Higher = more likely spoof. Threshold = {THRESHOLD})")
    print("=" * 60)


if __name__ == "__main__":
    main()