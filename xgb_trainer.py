"""
xgb_trainer.py - XGBoost Spoof Detector Trainer
EE 123 Project - Ved, Ching, Erick

Trains two XGBoost models on 1000 real vs 1000 spoof VCTK pairs:
  - Model A (tower1): MFCC biometric classifier
  - Model B (tower2): Physics anti-spoofing classifier

Saves:
  - tower1_mfcc_xgb.pkl
  - tower2_physics_xgb.pkl
  - physics_scaler.pkl

Usage:
    python xgb_trainer.py
"""

import os
import glob
import pickle
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from ching import build_advanced_feature_vector

# ─────────────────────────────────────────────
AUDIO_DIR     = os.path.dirname(os.path.abspath(__file__))
SAMPLES_DIR   = os.path.join(AUDIO_DIR, "Samples")
REAL_DIR      = os.path.join(SAMPLES_DIR, "source-vctk1k")
FAKE_DIR      = os.path.join(SAMPLES_DIR, "converted-vctk1k")
TEST_SIZE     = 0.2
RANDOM_STATE  = 42
THRESHOLD     = 0.5

PHYSICS_FEATURE_NAMES = [
    "Glottal Kurtosis", "Glottal Std", "Spectral Flux Std",
    "Modulation Energy", "Rolloff Std", "ZCR Std",
    "Aliasing Mean", "Aliasing Std"
]
# ─────────────────────────────────────────────


def process_single_file(file_data):
    file_path, label = file_data
    try:
        features = build_advanced_feature_vector(file_path)
        return features, label, file_path
    except Exception as e:
        return None, label, file_path


def load_dataset():
    real_files = sorted(glob.glob(os.path.join(REAL_DIR, "*.wav")))
    fake_files = sorted(glob.glob(os.path.join(FAKE_DIR, "*.wav")))

    print(f"  Found {len(real_files)} real files in: {REAL_DIR}")
    print(f"  Found {len(fake_files)} fake files in: {FAKE_DIR}")

    if len(real_files) == 0 or len(fake_files) == 0:
        print("ERROR: No files found. Check your folder paths.")
        exit(1)

    all_files  = real_files + fake_files
    all_labels = [0] * len(real_files) + [1] * len(fake_files)
    tasks = list(zip(all_files, all_labels))

    safe_cores = min(8, max(1, os.cpu_count() // 2))
    print(f"\n  Firing up {safe_cores} CPU cores via joblib...")

    results = Parallel(n_jobs=safe_cores, batch_size="auto")(
        delayed(process_single_file)(task)
        for task in tqdm(tasks, desc="Extracting features", ncols=80)
    )

    X, y, failed = [], [], []
    for features, label, path in results:
        if features is not None:
            X.append(features)
            y.append(label)
        else:
            failed.append(path)

    if failed:
        print(f"\n  Skipped {len(failed)} files due to errors.")

    print(f"\n  Dataset: {len(X)} samples ({y.count(0)} real, {y.count(1)} spoof)")
    return np.array(X), np.array(y)


def train(X, y):
    # Split features
    X_mfcc    = X[:, 0:120]
    X_physics = X[:, 120:]

    X_mfcc_train, X_mfcc_test, y_train, y_test = train_test_split(
        X_mfcc, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    X_phys_train, X_phys_test, _, _ = train_test_split(
        X_physics, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

    # Scale physics features
    scaler = StandardScaler()
    X_phys_train_scaled = scaler.fit_transform(X_phys_train)
    X_phys_test_scaled  = scaler.transform(X_phys_test)

    # ── Model A: MFCC Biometric Classifier ──
    print("\n" + "=" * 60)
    print("  MODEL A: MFCC Biometric Classifier (XGBoost)")
    print("=" * 60)
    xgb_mfcc = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=RANDOM_STATE,
        eval_metric="logloss"
    )
    xgb_mfcc.fit(X_mfcc_train, y_train)
    mfcc_preds = xgb_mfcc.predict(X_mfcc_test)
    print(f"  Accuracy: {accuracy_score(y_test, mfcc_preds)*100:.2f}%\n")
    print(classification_report(y_test, mfcc_preds, target_names=["Real", "Fake"]))

    # ── Model B: Physics Anti-Spoofing Classifier ──
    print("=" * 60)
    print("  MODEL B: Physics Anti-Spoofing Classifier (XGBoost)")
    print("=" * 60)
    xgb_phys = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        eval_metric="logloss"
    )
    xgb_phys.fit(X_phys_train_scaled, y_train)
    phys_probs = xgb_phys.predict_proba(X_phys_test_scaled)[:, 1]
    phys_preds = (phys_probs >= THRESHOLD).astype(int)
    print(f"  Accuracy (threshold={THRESHOLD}): {accuracy_score(y_test, phys_preds)*100:.2f}%\n")
    print(classification_report(y_test, phys_preds, target_names=["Real", "Fake"]))

    # Feature importances
    print("  Physics Feature Importances:")
    importances = xgb_phys.feature_importances_
    sorted_feats = sorted(zip(PHYSICS_FEATURE_NAMES, importances),
                          key=lambda x: x[1], reverse=True)
    for name, score in sorted_feats:
        print(f"    {name:<22}: {score:.4f} ({score*100:.1f}%)")

    return xgb_mfcc, xgb_phys, scaler


def save_models(xgb_mfcc, xgb_phys, scaler):
    with open("tower1_mfcc_xgb.pkl", "wb") as f:
        pickle.dump(xgb_mfcc, f)
    with open("tower2_physics_xgb.pkl", "wb") as f:
        pickle.dump(xgb_phys, f)
    with open("physics_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print("\n  Saved: tower1_mfcc_xgb.pkl, tower2_physics_xgb.pkl, physics_scaler.pkl")


def main():
    print("=" * 60)
    print("  xgb_trainer.py -- XGBoost Spoof Detector Trainer")
    print("  EE 123 Project -- Ved, Ching, Erick")
    print("=" * 60 + "\n")

    print("[1/3] Loading dataset...")
    X, y = load_dataset()

    print("\n[2/3] Training models...")
    xgb_mfcc, xgb_phys, scaler = train(X, y)

    print("\n[3/3] Saving models...")
    save_models(xgb_mfcc, xgb_phys, scaler)

    print("\n" + "=" * 60)
    print("  Done! Run xgb_batch_test.py to test on the prof's 8 pairs.")
    print("=" * 60)


if __name__ == "__main__":
    main()
