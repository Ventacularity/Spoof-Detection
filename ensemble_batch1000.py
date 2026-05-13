"""
ensemble_batch1000.py - Ensemble LR + XGBoost on full VCTK dataset
EE 123 Project - Ved, Ching, Erick

Tests all three ensemble strategies across 2000 samples:
  - 1000 source (real)
  - 1000 converted (spoof)

Strategies:
  OR:  spoof if EITHER model flags it
  AND: spoof if BOTH models flag it
  AVG: spoof if average probability >= threshold

Usage:
    python ensemble_batch1000.py
"""

import os
import glob
import pickle
import numpy as np
import joblib
from tqdm import tqdm
from ching import build_advanced_feature_vector

AUDIO_DIR     = os.path.dirname(os.path.abspath(__file__))
SAMPLES_DIR   = os.path.join(AUDIO_DIR, "Samples")
SOURCE_DIR    = os.path.join(SAMPLES_DIR, "source-vctk1k")
CONVERTED_DIR = os.path.join(SAMPLES_DIR, "converted-vctk1k")

LR_THRESHOLD  = 0.45
XGB_THRESHOLD = 0.50
AVG_THRESHOLD = 0.45


def load_models():
    # Logistic Regression
    loaded = joblib.load(os.path.join(AUDIO_DIR, "model.pkl"))
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


def run_folder(folder, true_label, label_str,
               clf_lr, scaler_lr, sel_idx, phys_start,
               clf_xgb, scaler_xgb):

    files = sorted(glob.glob(os.path.join(folder, "*.wav")))
    results = []
    errors = 0

    for filepath in tqdm(files, desc=f"  {label_str}", ncols=80):
        try:
            fv = build_advanced_feature_vector(filepath)
            physics = fv[phys_start:]

            # LR
            x_lr = physics[sel_idx].reshape(1, -1)
            prob_lr = clf_lr.predict_proba(scaler_lr.transform(x_lr))[0][1]

            # XGBoost
            x_xgb = physics.reshape(1, -1)
            prob_xgb = clf_xgb.predict_proba(scaler_xgb.transform(x_xgb))[0][1]

            results.append({
                "true_label": true_label,
                "prob_lr": prob_lr,
                "prob_xgb": prob_xgb,
            })
        except Exception:
            errors += 1

    print(f"    Done: {len(results)} processed, {errors} errors")
    return results


def evaluate_strategy(results, strategy):
    correct = real_correct = spoof_correct = 0
    real_total = spoof_total = 0

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

        if true_label == "REAL":
            real_total += 1
            if is_correct:
                real_correct += 1
        else:
            spoof_total += 1
            if is_correct:
                spoof_correct += 1

        if is_correct:
            correct += 1

    total = len(results)
    return correct, real_correct, spoof_correct, real_total, spoof_total, total


def print_strategy_results(strategy, correct, real_correct, spoof_correct,
                           real_total, spoof_total, total,
                           src_probs_lr, src_probs_xgb,
                           spf_probs_lr, spf_probs_xgb):
    print(f"\n{'='*60}")
    print(f"  STRATEGY: {strategy}", end="")
    if strategy == "OR":
        print(f"  (spoof if EITHER model flags it)")
    elif strategy == "AND":
        print(f"  (spoof if BOTH models flag it)")
    elif strategy == "AVG":
        print(f"  (spoof if avg prob >= {AVG_THRESHOLD})")
    print("=" * 60)
    print(f"  Real  correct : {real_correct:>5}/{real_total}  ({real_correct/real_total*100:.1f}%)")
    print(f"  Spoof correct : {spoof_correct:>5}/{spoof_total}  ({spoof_correct/spoof_total*100:.1f}%)")
    print(f"  Overall       : {correct:>5}/{total}  ({correct/total*100:.1f}%)")
    print(f"\n  Avg spoof probability:")
    print(f"    Real  -- LR: {np.mean(src_probs_lr):.3f}  XGB: {np.mean(src_probs_xgb):.3f}  "
          f"Avg: {np.mean([(l+x)/2 for l,x in zip(src_probs_lr, src_probs_xgb)]):.3f}")
    print(f"    Spoof -- LR: {np.mean(spf_probs_lr):.3f}  XGB: {np.mean(spf_probs_xgb):.3f}  "
          f"Avg: {np.mean([(l+x)/2 for l,x in zip(spf_probs_lr, spf_probs_xgb)]):.3f}")


def main():
    print("=" * 60)
    print("  ensemble_batch1000.py -- Full VCTK Ensemble Evaluation")
    print("  EE 123 Project -- Ved, Ching, Erick")
    print("=" * 60)

    print("\nLoading models...")
    clf_lr, scaler_lr, sel_idx, phys_start, clf_xgb, scaler_xgb = load_models()

    print("\n[1/2] Processing source (real)...")
    src_results = run_folder(SOURCE_DIR, "REAL", "source",
                             clf_lr, scaler_lr, sel_idx, phys_start,
                             clf_xgb, scaler_xgb)

    print("\n[2/2] Processing converted (spoof)...")
    spf_results = run_folder(CONVERTED_DIR, "SPOOF", "converted",
                             clf_lr, scaler_lr, sel_idx, phys_start,
                             clf_xgb, scaler_xgb)

    all_results = src_results + spf_results

    # Pull prob lists for distribution reporting
    src_probs_lr  = [r["prob_lr"]  for r in src_results]
    src_probs_xgb = [r["prob_xgb"] for r in src_results]
    spf_probs_lr  = [r["prob_lr"]  for r in spf_results]
    spf_probs_xgb = [r["prob_xgb"] for r in spf_results]

    print("\n" + "=" * 60)
    print("  RESULTS ACROSS ALL STRATEGIES")
    print("=" * 60)

    summary = []
    for strategy in ["OR", "AND", "AVG"]:
        correct, real_correct, spoof_correct, real_total, spoof_total, total = \
            evaluate_strategy(all_results, strategy)
        print_strategy_results(strategy, correct, real_correct, spoof_correct,
                               real_total, spoof_total, total,
                               src_probs_lr, src_probs_xgb,
                               spf_probs_lr, spf_probs_xgb)
        summary.append((strategy, correct, real_correct, spoof_correct, total,
                        real_total, spoof_total))

    # Final summary table
    print("\n" + "=" * 60)
    print("  SUMMARY TABLE")
    print("=" * 60)
    print(f"  {'Strategy':<8} {'Real':>10} {'Spoof':>10} {'Overall':>10}")
    print("  " + "-" * 42)
    for strategy, correct, real_correct, spoof_correct, total, real_total, spoof_total in summary:
        print(f"  {strategy:<8} {real_correct/real_total*100:>9.1f}% "
              f"{spoof_correct/spoof_total*100:>9.1f}% "
              f"{correct/total*100:>9.1f}%")

    # Also print individual model baselines for comparison
    print(f"\n  Individual model baselines (from previous runs):")
    print(f"  {'LR alone':<8} {'83.4%':>10} {'62.7%':>10} {'76.5%':>10}")
    print(f"  {'XGB alone':<9} {'88.5%':>9} {'62.7%':>10} {'76.5%':>10}")
    print("=" * 60)


if __name__ == "__main__":
    main()
