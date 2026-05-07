"""
batch_test.py - Run inference on all 8 real/spoof pairs at once
EE 123 Project - Ved, Ching, Erick

Usage:
    python batch_test.py
"""

import os
import numpy as np
import joblib
from ching import build_advanced_feature_vector

AUDIO_DIR = os.path.dirname(os.path.abspath(__file__))
NUM_PAIRS = 8
THRESHOLD = 0.45
SELECTED_NAMES = ["Glottal Kurtosis", "Glottal Std", "Modulation Energy"]

def run_batch():
    model_path = os.path.join(AUDIO_DIR, "model.pkl")
    loaded = joblib.load(model_path)
    if len(loaded) == 4:
        clf, scaler, sel_idx, phys_start = loaded
    else:
        clf, scaler = loaded
        sel_idx = [0, 1, 3]
        phys_start = 120

    print("=" * 70)
    print("   Batch Test -- All 8 Real/Spoof Pairs")
    print("   EE 123 Project -- Ved, Ching, Erick")
    print("=" * 70)
    print(f"\n{'File':<15} {'Label':<8} {'Spoof%':>8} {'Real%':>8} {'Verdict':<10} {'Correct?'}")
    print("-" * 70)

    correct = 0
    total = 0
    spoof_correct = 0
    real_correct = 0

    for i in range(1, NUM_PAIRS + 1):
        for filepath, true_label in [
            (os.path.join(AUDIO_DIR, f"r{i}.wav"), "REAL"),
            (os.path.join(AUDIO_DIR, f"try{i}.wav"), "SPOOF")
        ]:
            if not os.path.exists(filepath):
                continue

            fv = build_advanced_feature_vector(filepath)
            physics = fv[phys_start:]
            x = physics[sel_idx].reshape(1, -1)
            x_scaled = scaler.transform(x)
            prob_spoof = clf.predict_proba(x_scaled)[0][1]
            prob_real = 1 - prob_spoof

            verdict = "SPOOF" if prob_spoof >= THRESHOLD else "REAL"
            is_correct = verdict == true_label
            if is_correct:
                correct += 1
                if true_label == "SPOOF":
                    spoof_correct += 1
                else:
                    real_correct += 1
            total += 1

            tick = "correct" if is_correct else "WRONG"
            fname = os.path.basename(filepath)
            print(f"{fname:<15} {true_label:<8} {prob_spoof*100:>7.1f}% {prob_real*100:>7.1f}% {verdict:<10} {tick}")

    print("-" * 70)
    print(f"\n  Real  caught: {real_correct}/8")
    print(f"  Spoof caught: {spoof_correct}/8")
    print(f"  Overall     : {correct}/{total} = {correct/total*100:.1f}%")
    print("=" * 70)

if __name__ == "__main__":
    run_batch()