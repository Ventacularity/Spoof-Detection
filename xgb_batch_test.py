"""
xgb_batch_test.py - Run XGBoost inference on the prof's 8 real/spoof pairs
EE 123 Project - Ved, Ching, Erick

Usage:
    python xgb_batch_test.py
"""

import os
import pickle
import numpy as np
from ching import build_advanced_feature_vector

AUDIO_DIR  = os.path.dirname(os.path.abspath(__file__))
THRESHOLD  = 0.5
NUM_PAIRS  = 8

TEST_FILES = (
    [(f"r{i}.wav",   "REAL")  for i in range(1, NUM_PAIRS + 1)] +
    [(f"try{i}.wav", "SPOOF") for i in range(1, NUM_PAIRS + 1)]
)


def load_models():
    with open(os.path.join(AUDIO_DIR, "tower2_physics_xgb.pkl"), "rb") as f:
        model_phys = pickle.load(f)
    with open(os.path.join(AUDIO_DIR, "physics_scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    return model_phys, scaler


def main():
    print("=" * 60)
    print("  xgb_batch_test.py -- Prof's 8 Pair Evaluation")
    print("  EE 123 Project -- Ved, Ching, Erick")
    print("=" * 60)

    model_phys, scaler = load_models()

    print(f"\n{'File':<15} {'True Label':<10} {'Spoof%':>8} {'Verdict':<10} {'Correct?'}")
    print("-" * 57)

    correct = real_correct = spoof_correct = total = 0

    for filename, true_label in TEST_FILES:
        path = os.path.join(AUDIO_DIR, filename)
        if not os.path.exists(path):
            print(f"{filename:<15} FILE NOT FOUND")
            continue

        try:
            fv = build_advanced_feature_vector(path)
            x  = fv[120:].reshape(1, -1)
            x_scaled   = scaler.transform(x)
            prob_spoof  = model_phys.predict_proba(x_scaled)[0][1]
            verdict     = "SPOOF" if prob_spoof >= THRESHOLD else "REAL"
            is_correct  = verdict == true_label

            if is_correct:
                correct += 1
                if true_label == "REAL":
                    real_correct += 1
                else:
                    spoof_correct += 1
            total += 1

            tick = "✓" if is_correct else "✗"
            print(f"{filename:<15} {true_label:<10} {prob_spoof*100:>7.1f}% {verdict:<10} {tick}")

        except Exception as e:
            print(f"{filename:<15} ERROR: {e}")

    print("-" * 57)
    print(f"\n  Real  caught : {real_correct}/{NUM_PAIRS}")
    print(f"  Spoof caught : {spoof_correct}/{NUM_PAIRS}")
    print(f"  Overall      : {correct}/{total} = {correct/total*100:.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
