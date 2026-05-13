"""
ensemble_test.py - Combined LR + XGBoost Spoof Detector
EE 123 Project - Ved, Ching, Erick

Combines both models on the prof's 8 pairs using three voting strategies:
  - OR:  spoof if EITHER model says spoof (maximize spoof recall)
  - AND: spoof if BOTH models say spoof (maximize precision)
  - AVG: spoof if average probability >= threshold

Usage:
    python ensemble_test.py
"""

import os
import pickle
import numpy as np
import joblib
from ching import build_advanced_feature_vector

AUDIO_DIR  = os.path.dirname(os.path.abspath(__file__))
NUM_PAIRS  = 8
LR_THRESHOLD  = 0.45
XGB_THRESHOLD = 0.50
AVG_THRESHOLD = 0.45

TEST_FILES = (
    [(f"r{i}.wav",   "REAL")  for i in range(1, NUM_PAIRS + 1)] +
    [(f"try{i}.wav", "SPOOF") for i in range(1, NUM_PAIRS + 1)]
)


def load_models():
    # Logistic Regression
    lr_path = os.path.join(AUDIO_DIR, "model.pkl")
    loaded = joblib.load(lr_path)
    if len(loaded) == 4:
        clf_lr, scaler_lr, sel_idx, phys_start = loaded
    else:
        clf_lr, scaler_lr = loaded
        sel_idx = [0, 2, 3, 5, 6]
        phys_start = 120

    # XGBoost
    with open(os.path.join(AUDIO_DIR, "tower2_physics_xgb.pkl"), "rb") as f:
        clf_xgb = pickle.load(f)
    with open(os.path.join(AUDIO_DIR, "physics_scaler.pkl"), "rb") as f:
        scaler_xgb = pickle.load(f)

    return clf_lr, scaler_lr, sel_idx, phys_start, clf_xgb, scaler_xgb


def run_ensemble(clf_lr, scaler_lr, sel_idx, phys_start, clf_xgb, scaler_xgb):

    results = []

    for filename, true_label in TEST_FILES:
        path = os.path.join(AUDIO_DIR, filename)
        if not os.path.exists(path):
            continue

        fv = build_advanced_feature_vector(path)
        physics = fv[phys_start:]

        # LR probability
        x_lr = physics[sel_idx].reshape(1, -1)
        x_lr_scaled = scaler_lr.transform(x_lr)
        prob_lr = clf_lr.predict_proba(x_lr_scaled)[0][1]

        # XGBoost probability
        x_xgb = physics.reshape(1, -1)
        x_xgb_scaled = scaler_xgb.transform(x_xgb)
        prob_xgb = clf_xgb.predict_proba(x_xgb_scaled)[0][1]

        results.append({
            "filename": filename,
            "true_label": true_label,
            "prob_lr": prob_lr,
            "prob_xgb": prob_xgb,
        })

    return results


def evaluate(results, strategy):
    correct = real_correct = spoof_correct = 0

    print(f"\n{'File':<15} {'True':<8} {'LR%':>6} {'XGB%':>6} {'Avg%':>6} {'Verdict':<10} {'OK?'}")
    print("-" * 63)

    for r in results:
        prob_lr  = r["prob_lr"]
        prob_xgb = r["prob_xgb"]
        prob_avg = (prob_lr + prob_xgb) / 2
        true_label = r["true_label"]

        if strategy == "OR":
            spoof = prob_lr >= LR_THRESHOLD or prob_xgb >= XGB_THRESHOLD
        elif strategy == "AND":
            spoof = prob_lr >= LR_THRESHOLD and prob_xgb >= XGB_THRESHOLD
        elif strategy == "AVG":
            spoof = prob_avg >= AVG_THRESHOLD

        verdict = "SPOOF" if spoof else "REAL"
        is_correct = verdict == true_label
        if is_correct:
            correct += 1
            if true_label == "REAL":
                real_correct += 1
            else:
                spoof_correct += 1

        tick = "✓" if is_correct else "✗"
        print(f"{r['filename']:<15} {true_label:<8} {prob_lr*100:>5.1f}% {prob_xgb*100:>5.1f}% "
              f"{prob_avg*100:>5.1f}% {verdict:<10} {tick}")

    total = len(results)
    print("-" * 63)
    print(f"  Real  caught : {real_correct}/{NUM_PAIRS}")
    print(f"  Spoof caught : {spoof_correct}/{NUM_PAIRS}")
    print(f"  Overall      : {correct}/{total} = {correct/total*100:.1f}%")
    return correct, total


def main():
    print("=" * 63)
    print("  ensemble_test.py -- LR + XGBoost Combined Detector")
    print("  EE 123 Project -- Ved, Ching, Erick")
    print("=" * 63)

    print("\nLoading models...")
    clf_lr, scaler_lr, sel_idx, phys_start, clf_xgb, scaler_xgb = load_models()

    print("Extracting features from 16 files...")
    results = run_ensemble(clf_lr, scaler_lr, sel_idx, phys_start, clf_xgb, scaler_xgb)

    for strategy in ["OR", "AND", "AVG"]:
        print(f"\n{'='*63}")
        print(f"  STRATEGY: {strategy}", end="")
        if strategy == "OR":
            print("  (spoof if EITHER model flags it)")
        elif strategy == "AND":
            print("  (spoof if BOTH models flag it)")
        elif strategy == "AVG":
            print(f"  (spoof if avg probability >= {AVG_THRESHOLD})")
        print("=" * 63)
        evaluate(results, strategy)

    print("\n" + "=" * 63)
    print("  Summary:")
    print(f"  LR threshold : {LR_THRESHOLD}")
    print(f"  XGB threshold: {XGB_THRESHOLD}")
    print(f"  AVG threshold: {AVG_THRESHOLD}")
    print("=" * 63)


if __name__ == "__main__":
    main()
