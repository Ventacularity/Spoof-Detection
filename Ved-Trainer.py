"""
Ved-Trainer.py - Spoof Detector Retrainer
EE 123 Project - Ved, Ching, Erick

Trains a logistic regression on SPARC voice conversion samples.

Dataset (inside Samples/ folder):
  Real:  source-vctk1k/  (1000 files)
         target-vctk1k/  (1000 files)
  Spoof: converted-vctk1k/ (1000 files)

Features used (from feature_analysis.txt — 1000 pair analysis):
  KEEP:
  - ZCR Std          d=1.214  consistency=93.8%  <- NEW #1 (was wrongly dropped)
  - Glottal Kurtosis d=1.015  consistency=81.5%  <- confirmed good
  - Aliasing Mean    d=1.004  consistency=91.6%  <- NEW #3 (was wrongly dropped)
  - Spectral Flux    d=0.959  consistency=79.2%  <- NEW #4 (was wrongly dropped)
  - Modulation Energy d=0.352 consistency=60.5%  <- weakest keeper
  DROP:
  - Rolloff Std      d=0.293  consistency=59.4%  <- below threshold
  - Glottal Std      d=0.132  consistency=55.0%  <- basically coin flip

Physics block indices (relative to index 120):
  0: Glottal Kurtosis  <- KEEP
  1: Glottal Std       <- DROP
  2: Spectral Flux Std <- KEEP
  3: Modulation Energy <- KEEP
  4: Rolloff Std       <- DROP
  5: ZCR Std           <- KEEP
  6: Aliasing Mean     <- KEEP

Usage:
    python Ved-Trainer.py

Output:
    model.pkl - saved logistic regression + scaler
"""

import os
import glob
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix
from ching import build_advanced_feature_vector

# ─────────────────────────────────────────────
AUDIO_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLES_DIR = os.path.join(AUDIO_DIR, "Samples")

SOURCE_DIR    = os.path.join(SAMPLES_DIR, "source-vctk1k")
TARGET_DIR    = os.path.join(SAMPLES_DIR, "target-vctk1k")
CONVERTED_DIR = os.path.join(SAMPLES_DIR, "converted-vctk1k")

PHYSICS_START = 120
# Updated based on 1000-pair feature_analysis.txt
SELECTED_INDICES = [0, 2, 3, 5, 6]  # Glottal Kurtosis, Spectral Flux, Modulation Energy, ZCR Std, Aliasing Mean
SELECTED_NAMES = ["Glottal Kurtosis", "Spectral Flux Std", "Modulation Energy", "ZCR Std", "Aliasing Mean"]

CV_FOLDS = 5
# ─────────────────────────────────────────────


def extract_features_from_folder(folder, label, label_str):
    files = sorted(glob.glob(os.path.join(folder, "*.wav")))
    X, y, names = [], [], []
    errors = 0

    for i, filepath in enumerate(files):
        filename = os.path.basename(filepath)
        if (i + 1) % 100 == 0:
            print(f"    [{label_str}] {i+1}/{len(files)} processed ({errors} errors)...")
        try:
            features = build_advanced_feature_vector(filepath)
            X.append(features)
            y.append(label)
            names.append(filename)
        except Exception as e:
            errors += 1

    print(f"    [{label_str}] Done: {len(X)} extracted, {errors} skipped.")
    return X, y, names


def load_dataset():
    print(f"  Loading from: {SAMPLES_DIR}\n")

    print(f"  [1/3] Source (real): {SOURCE_DIR}")
    X_src, y_src, n_src = extract_features_from_folder(SOURCE_DIR, 0, "REAL/source")

    print(f"\n  [2/3] Target (real): {TARGET_DIR}")
    X_tgt, y_tgt, n_tgt = extract_features_from_folder(TARGET_DIR, 0, "REAL/target")

    print(f"\n  [3/3] Converted (spoof): {CONVERTED_DIR}")
    X_spf, y_spf, n_spf = extract_features_from_folder(CONVERTED_DIR, 1, "SPOOF")

    X = np.array(X_src + X_tgt + X_spf)
    y = np.array(y_src + y_tgt + y_spf)
    names = n_src + n_tgt + n_spf

    n_real = len(X_src) + len(X_tgt)
    n_spoof = len(X_spf)
    print(f"\n  Total: {len(X)} samples ({n_real} real, {n_spoof} spoof)")
    return X, y, names


def evaluate_kfold(X_selected, y):
    print("\n" + "=" * 60)
    print(f"   {CV_FOLDS}-FOLD STRATIFIED CROSS VALIDATION")
    print(f"   Features: {SELECTED_NAMES}")
    print("=" * 60)

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000, random_state=42))
    ])

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    y_pred = cross_val_predict(pipeline, X_selected, y, cv=cv)

    correct = np.sum(y_pred == y)
    total = len(y)
    accuracy = correct / total * 100

    print(f"\n  Overall {CV_FOLDS}-Fold Accuracy: {correct}/{total} = {accuracy:.1f}%\n")
    print(classification_report(y, y_pred, target_names=["Real", "Spoof"]))

    cm = confusion_matrix(y, y_pred)
    print("Confusion Matrix (rows=actual, cols=predicted):")
    print(f"               Pred:Real  Pred:Spoof")
    print(f"  Actual:Real    {cm[0][0]:>6}      {cm[0][1]:>6}")
    print(f"  Actual:Spoof   {cm[1][0]:>6}      {cm[1][1]:>6}")

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

    for folder in [SOURCE_DIR, TARGET_DIR, CONVERTED_DIR]:
        if not os.path.exists(folder):
            print(f"ERROR: Could not find folder: {folder}")
            print(f"Make sure the Samples/ folder is in the same directory as Ved-Trainer.py")
            return

    print("[1/4] Loading audio files and extracting features...")
    print("      (This will take a while -- ~3000 files to process)\n")
    X, y, filenames = load_dataset()

    print(f"\n[2/4] Selecting features: {SELECTED_NAMES}")
    X_physics = X[:, PHYSICS_START:]
    X_selected = X_physics[:, SELECTED_INDICES]
    print(f"      Full vector: {X.shape[1]} dims -> physics: {X_physics.shape[1]} -> selected: {X_selected.shape[1]}")

    print(f"\n[3/4] Running {CV_FOLDS}-Fold Cross Validation...")
    accuracy = evaluate_kfold(X_selected, y)

    print("\n[4/4] Training final model and saving...")
    clf, scaler = train_final_model(X_selected, y)

    model_path = os.path.join(AUDIO_DIR, "model.pkl")
    joblib.dump((clf, scaler, SELECTED_INDICES, PHYSICS_START), model_path)
    print(f"\n  Saved model to: {model_path}")

    print("\n" + "=" * 60)
    print(f"   DONE -- CV Accuracy: {accuracy:.1f}%")
    print("   Run inference.py or batch_test.py to test on audio.")
    print("=" * 60)


if __name__ == "__main__":
    main()