"""
train_all.py - Train All Spoof Detection Models
EE 123 Project - Ved, Ching, Erick

Trains on 2000 VCTK samples (balanced):
  Real  : source-vctk1k/ (1000)
  Spoof : converted-vctk1k/ (1000)

Feature vector layout (from ching.py):
  [0]        Glottal Kurtosis
  [1]        Glottal Std
  [2]        Spectral Flux Std
  [3]        Spectral Flux Max
  [4]        Modulation Energy
  [5..124]   MFCC means + stds (120 features)
  [125]      Aliasing Mean
  [126]      Aliasing Std

Saves:
  model_lr.pkl    - Logistic Regression (5 physics features, 5-fold CV)
  model_xgb.pkl   - XGBoost (7 physics features, 80/20 split)
  model_mfcc.pkl  - XGBoost MFCC biometric (80/20 split)

Usage:
    python train_all.py
"""

import os
import glob
import pickle
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from ching import build_advanced_feature_vector

# ─────────────────────────────────────────────
AUDIO_DIR     = os.path.dirname(os.path.abspath(__file__))
SAMPLES_DIR   = os.path.join(AUDIO_DIR, "Samples")
SOURCE_DIR    = os.path.join(SAMPLES_DIR, "source-vctk1k")
CONVERTED_DIR = os.path.join(SAMPLES_DIR, "converted-vctk1k")

PHYSICS_INDICES = [0, 1, 2, 3, 4, 125, 126]
LR_INDICES      = [0, 2, 4, 125, 126]
MFCC_START      = 5
MFCC_END        = 125

PHYSICS_NAMES = [
    "Glottal Kurtosis", "Glottal Std", "Spectral Flux Std",
    "Spectral Flux Max", "Modulation Energy", "Aliasing Mean", "Aliasing Std"
]
LR_NAMES = ["Glottal Kurtosis", "Spectral Flux Std",
            "Modulation Energy", "Aliasing Mean", "Aliasing Std"]

RANDOM_STATE = 42
TEST_SIZE    = 0.2
CV_FOLDS     = 5
# ─────────────────────────────────────────────


def process_file(args):
    path, label = args
    try:
        fv = build_advanced_feature_vector(path)
        return fv, label
    except Exception:
        return None, label


def load_dataset():
    source_files    = sorted(glob.glob(os.path.join(SOURCE_DIR,    "*.wav")))
    converted_files = sorted(glob.glob(os.path.join(CONVERTED_DIR, "*.wav")))

    print(f"  Source    (real) : {len(source_files)} files")
    print(f"  Converted (spoof): {len(converted_files)} files")
    print(f"  Total            : {len(source_files) + len(converted_files)} files")

    tasks = [(f, 0) for f in source_files] + [(f, 1) for f in converted_files]
    cores = min(8, max(1, os.cpu_count() // 2))
    print(f"\n  Using {cores} CPU cores...")

    results = Parallel(n_jobs=cores, batch_size="auto")(
        delayed(process_file)(t)
        for t in tqdm(tasks, desc="  Extracting features", ncols=75)
    )

    X, y, errors = [], [], 0
    for fv, label in results:
        if fv is not None:
            X.append(fv)
            y.append(label)
        else:
            errors += 1

    X, y = np.array(X), np.array(y)
    print(f"\n  Done: {len(X)} samples ({np.sum(y==0)} real, {np.sum(y==1)} spoof)"
          f"  [{errors} errors]")
    return X, y


def train_logistic_regression(X, y):
    print("\n" + "=" * 60)
    print("  MODEL 1: Logistic Regression")
    print(f"  Features: {LR_NAMES}")
    print("=" * 60)

    X_phys_all  = X[:, PHYSICS_INDICES]
    lr_relative = [PHYSICS_INDICES.index(i) for i in LR_INDICES]
    X_lr        = X_phys_all[:, lr_relative]

    scaler_lr   = StandardScaler()
    X_lr_scaled = scaler_lr.fit_transform(X_lr)

    clf_lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    cv     = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    y_pred = cross_val_predict(clf_lr, X_lr_scaled, y, cv=cv)

    acc = accuracy_score(y, y_pred) * 100
    print(f"\n  {CV_FOLDS}-Fold CV Accuracy: {acc:.1f}%\n")
    print(classification_report(y, y_pred, target_names=["Real", "Spoof"]))

    cm = confusion_matrix(y, y_pred)
    print(f"  Confusion Matrix:")
    print(f"                   Pred:Real  Pred:Spoof")
    print(f"    Actual:Real      {cm[0][0]:>6}      {cm[0][1]:>6}")
    print(f"    Actual:Spoof     {cm[1][0]:>6}      {cm[1][1]:>6}")

    clf_lr.fit(X_lr_scaled, y)
    print(f"\n  Feature coefficients:")
    for name, coef in zip(LR_NAMES, clf_lr.coef_[0]):
        print(f"    {name:<22}: {coef:+.4f}  ({'→ spoof' if coef > 0 else '→ real'})")

    return clf_lr, scaler_lr, lr_relative, acc


def train_xgboost(X, y):
    print("\n" + "=" * 60)
    print("  MODEL 2: XGBoost (Physics)")
    print(f"  Features: {PHYSICS_NAMES}")
    print("=" * 60)

    X_phys = X[:, PHYSICS_INDICES]
    X_train, X_test, y_train, y_test = train_test_split(
        X_phys, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

    scaler_xgb  = StandardScaler()
    X_train_s   = scaler_xgb.fit_transform(X_train)
    X_test_s    = scaler_xgb.transform(X_test)

    clf_xgb = XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=2,
        random_state=RANDOM_STATE, eval_metric="logloss")
    clf_xgb.fit(X_train_s, y_train)

    probs = clf_xgb.predict_proba(X_test_s)[:, 1]
    preds = (probs >= 0.5).astype(int)
    acc   = accuracy_score(y_test, preds) * 100

    print(f"\n  80/20 Split Accuracy: {acc:.1f}%\n")
    print(classification_report(y_test, preds, target_names=["Real", "Spoof"]))

    cm = confusion_matrix(y_test, preds)
    print(f"  Confusion Matrix:")
    print(f"                   Pred:Real  Pred:Spoof")
    print(f"    Actual:Real      {cm[0][0]:>6}      {cm[0][1]:>6}")
    print(f"    Actual:Spoof     {cm[1][0]:>6}      {cm[1][1]:>6}")

    print(f"\n  Feature Importances:")
    importances = clf_xgb.feature_importances_
    sorted_feats = sorted(zip(PHYSICS_NAMES, importances),
                          key=lambda x: x[1], reverse=True)
    for name, score in sorted_feats:
        bar = "█" * int(score * 50)
        print(f"    {name:<22}: {score:.4f} ({score*100:.1f}%)  {bar}")

    # MFCC biometric model
    print("\n" + "=" * 60)
    print("  MODEL 2B: XGBoost (MFCC Biometric)")
    print("=" * 60)
    X_mfcc = X[:, MFCC_START:MFCC_END]
    X_m_train, X_m_test, yt, yv = train_test_split(
        X_mfcc, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    clf_mfcc = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        random_state=RANDOM_STATE, eval_metric="logloss")
    clf_mfcc.fit(X_m_train, yt)
    mfcc_acc = accuracy_score(yv, clf_mfcc.predict(X_m_test)) * 100
    print(f"\n  MFCC Biometric Accuracy: {mfcc_acc:.1f}%")

    return clf_xgb, scaler_xgb, clf_mfcc, acc, mfcc_acc


def save_models(clf_lr, scaler_lr, lr_relative,
                clf_xgb, scaler_xgb, clf_mfcc):
    with open(os.path.join(AUDIO_DIR, "model_lr.pkl"), "wb") as f:
        pickle.dump((clf_lr, scaler_lr, lr_relative, PHYSICS_INDICES), f)
    with open(os.path.join(AUDIO_DIR, "model_xgb.pkl"), "wb") as f:
        pickle.dump((clf_xgb, scaler_xgb, PHYSICS_INDICES), f)
    with open(os.path.join(AUDIO_DIR, "model_mfcc.pkl"), "wb") as f:
        pickle.dump((clf_mfcc, MFCC_START, MFCC_END), f)
    print("\n  Saved: model_lr.pkl, model_xgb.pkl, model_mfcc.pkl")


def main():
    print("=" * 60)
    print("  train_all.py — Spoof Detector Trainer")
    print("  EE 123 Project — Ved, Ching, Erick")
    print("=" * 60 + "\n")

    for d in [SOURCE_DIR, CONVERTED_DIR]:
        if not os.path.exists(d):
            print(f"ERROR: Missing folder: {d}")
            exit(1)

    print("[1/3] Loading dataset...")
    X, y = load_dataset()

    print("\n[2/3] Training models...")
    clf_lr, scaler_lr, lr_relative, lr_acc = train_logistic_regression(X, y)
    clf_xgb, scaler_xgb, clf_mfcc, xgb_acc, mfcc_acc = train_xgboost(X, y)

    print("\n[3/3] Saving models...")
    save_models(clf_lr, scaler_lr, lr_relative,
                clf_xgb, scaler_xgb, clf_mfcc)

    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE")
    print("=" * 60)
    print(f"  LR  accuracy  : {lr_acc:.1f}%  ({CV_FOLDS}-fold CV, 2000 samples)")
    print(f"  XGB accuracy  : {xgb_acc:.1f}%  (80/20 split, 2000 samples)")
    print(f"  MFCC accuracy : {mfcc_acc:.1f}%  (80/20 split, 2000 samples)")
    print(f"\n  Run prof.py, report1000.py, or report500.py next.")
    print("=" * 60)


if __name__ == "__main__":
    main()