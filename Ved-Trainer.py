"""
Ved-Trainer.py - Spoof Detector Retrainer
EE 123 Project - Ved, Ching, Erick

Trains a logistic regression on SPARC voice conversion samples.
Real samples: r1.wav - r12.wav  (label: 0)
Spoof samples: try1.wav - try8.wav (label: 1)

Uses leave-one-out cross validation since we have few samples.

Based on feature_tables.txt analysis across 8 pairs:
  - Modulation Energy: real > spoof in 7/8 pairs, 21% avg diff  <- BEST
  - Glottal Kurtosis:  real > spoof in 6/8 pairs, 17% avg diff  <- GOOD
  - Glottal Std:       real > spoof in 7/8 pairs, 13% avg diff  <- GOOD
  - Phase Variance:    0.0% avg diff                            <- DROPPED
  - Rolloff Std:       flips direction, 1.1% avg diff           <- DROPPED
  - ZCR Std:           speaker-dependent, not spoof-dependent   <- DROPPED
  - Aliasing Mean:     speaker-dependent, inconsistent          <- DROPPED

Usage:
    python Ved-Trainer.py

Output:
    model.pkl  -- saved logistic regression + scaler
"""

import os
import glob
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import classification_report, confusion_matrix
from ching import build_advanced_feature_vector

# ─────────────────────────────────────────────
AUDIO_DIR = os.path.dirname(os.path.abspath(__file__))

# Physics block starts at index 120 in ching.py's feature vector:
# 120: Glottal Kurtosis  <- GOOD   (real > spoof in 6/8, 17% avg diff)
# 121: Glottal Std       <- GOOD   (real > spoof in 7/8, 13% avg diff)
# 122: Spectral Flux Std <- WEAK   (6.5% avg diff, inconsistent direction)
# 123: Modulation Energy <- BEST   (real > spoof in 7/8, 21% avg diff)
# 124: Rolloff Std       <- DROPPED (flips direction, only 1.1% avg diff)
# 125: ZCR Std           <- DROPPED (speaker-dependent, not spoof-dependent)
# 126: Aliasing Mean     <- DROPPED (speaker-dependent, inconsistent)
# Phase Variance         <- DROPPED (0.0% avg diff, completely useless)
#
# Decision rule: real voices have HIGHER values for all 3 selected features.
# Lower values = more likely SPARC vocoder output = spoof.
PHYSICS_START = 120
SELECTED_INDICES = [0, 1, 3]  # Glottal Kurtosis, Glottal Std, Modulation Energy
SELECTED_NAMES = ["Glottal Kurtosis", "Glottal Std", "Modulation Energy"]
# ─────────────────────────────────────────────


def load_dataset(audio_dir):
    real_files = sorted(glob.glob(os.path.join(audio_dir, "r*.wav")))
    spoof_files = sorted(glob.glob(os.path.join(audio_dir, "try*.wav")))

    if len(real_files) == 0 or len(spoof_files) == 0:
        raise FileNotFoundError(
            f"Could not find audio files in {audio_dir}.\n"
            f"Make sure r1.wav-r12.wav and try1.wav-try8.wav are in the same folder."
        )

    print(f"Found {len(real_files)} real files and {len(spoof_files)} spoof files.")
    print(f"Total samples: {len(real_files) + len(spoof_files)}\n")

    X, y, filenames = [], [], []
    all_files = [(f, 0) for f in real_files] + [(f, 1) for f in spoof_files]

    for filepath, label in all_files:
        filename = os.path.basename(filepath)
        label_str = "REAL" if label == 0 else "SPOOF"
        print(f"  Extracting [{label_str}] {filename}...")
        try:
            features = build_advanced_feature_vector(filepath)
            X.append(features)
            y.append(label)
            filenames.append(filename)
        except Exception as e:
            print(f"  Warning: Skipping {filename}: {e}")

    return np.array(X), np.array(y), filenames


def evaluate_loocv(X_selected, y, filenames):
    print("\n" + "=" * 60)
    print("   LEAVE-ONE-OUT CROSS VALIDATION")
    print(f"   Features used: {SELECTED_NAMES}")
    print("=" * 60)

    loo = LeaveOneOut()
    y_true, y_pred, y_prob = [], [], []

    for train_idx, test_idx in loo.split(X_selected):
        X_train, X_test = X_selected[train_idx], X_selected[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X_train_scaled, y_train)

        prob = clf.predict_proba(X_test_scaled)[0][1]
        pred = int(prob >= 0.5)

        y_true.append(y_test[0])
        y_pred.append(pred)
        y_prob.append(prob)

        label_str = "REAL " if y_test[0] == 0 else "SPOOF"
        result_str = "correct" if pred == y_test[0] else "WRONG"
        print(f"  [{label_str}] {filenames[test_idx[0]]:<15} -> prob={prob:.3f}  {result_str}")

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    correct = np.sum(y_true == y_pred)
    total = len(y_true)
    accuracy = correct / total * 100

    print(f"\n  Overall LOOCV Accuracy: {correct}/{total} = {accuracy:.1f}%")
    print("\n" + classification_report(y_true, y_pred, target_names=["Real", "Spoof"]))
    print("Confusion Matrix (rows=actual, cols=predicted):")
    print("               Pred:Real  Pred:Spoof")
    cm = confusion_matrix(y_true, y_pred)
    print(f"  Actual:Real      {cm[0][0]}          {cm[0][1]}")
    print(f"  Actual:Spoof     {cm[1][0]}          {cm[1][1]}")

    return accuracy


def train_final_model(X_selected, y):
    print("\n" + "=" * 60)
    print("   TRAINING FINAL MODEL ON ALL DATA")
    print("=" * 60)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_selected)

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_scaled, y)

    print(f"  Trained on {len(y)} samples ({np.sum(y==0)} real, {np.sum(y==1)} spoof)")
    print(f"  Feature coefficients:")
    for name, coef in zip(SELECTED_NAMES, clf.coef_[0]):
        direction = "spoof" if coef > 0 else "real"
        print(f"    {name:<22}: {coef:+.4f}  (higher = {direction})")

    return clf, scaler


def main():
    print("=" * 60)
    print("   Ved-Trainer -- SPARC Spoof Detector Retrainer")
    print("   EE 123 Project -- Ved, Ching, Erick")
    print("=" * 60 + "\n")

    print("[1/4] Loading audio files and extracting features...")
    X, y, filenames = load_dataset(AUDIO_DIR)

    print(f"\n[2/4] Selecting features: {SELECTED_NAMES}")
    X_physics = X[:, PHYSICS_START:]
    X_selected = X_physics[:, SELECTED_INDICES]
    print(f"      Full vector: {X.shape[1]} dims -> physics: {X_physics.shape[1]} -> selected: {X_selected.shape[1]}")

    print("\n[3/4] Running Leave-One-Out Cross Validation...")
    accuracy = evaluate_loocv(X_selected, y, filenames)

    print("\n[4/4] Training final model and saving...")
    clf, scaler = train_final_model(X_selected, y)

    model_path = os.path.join(AUDIO_DIR, "model.pkl")
    joblib.dump((clf, scaler, SELECTED_INDICES, PHYSICS_START), model_path)
    print(f"\n  Saved model to: {model_path}")

    print("\n" + "=" * 60)
    print(f"   DONE -- LOOCV Accuracy: {accuracy:.1f}%")
    print("   Run inference.py to test on new audio files.")
    print("=" * 60)


if __name__ == "__main__":
    main()