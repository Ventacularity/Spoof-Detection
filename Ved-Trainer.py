"""
Ved-Trainer.py - Spoof Detector Retrainer
EE 123 Project - Ved, Ching, Erick

Trains a logistic regression on SPARC voice conversion samples.
Real samples: r1.wav - r12.wav  (label: 0)
Spoof samples: try1.wav - try8.wav (label: 1)

Uses leave-one-out cross validation since we have few samples.
Trains only on physics features (not MFCCs) to avoid overfitting.

Usage:
    python Ved-Trainer.py
    
Output:
    model.pkl  — saved logistic regression + scaler
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
# CONFIG — change this path if needed
AUDIO_DIR = os.path.dirname(os.path.abspath(__file__))  # same folder as this script

# Physics feature indices (from ching.py's 128-dim vector)
# These are indices 120 onwards: glottal kurtosis, glottal std, flux std,
# modulation energy, rolloff std, ZCR std, aliasing mean, aliasing std
PHYSICS_START = 120
# ─────────────────────────────────────────────


def load_dataset(audio_dir):
    """
    Loads all r*.wav (real) and try*.wav (spoof) files from the given directory.
    Returns feature matrix X and label vector y.
    """
    real_files = sorted(glob.glob(os.path.join(audio_dir, "r*.wav")))
    spoof_files = sorted(glob.glob(os.path.join(audio_dir, "try*.wav")))

    if len(real_files) == 0 or len(spoof_files) == 0:
        raise FileNotFoundError(
            f"Could not find audio files in {audio_dir}.\n"
            f"Make sure r1.wav-r12.wav and try1.wav-try8.wav are in the same folder."
        )

    print(f"Found {len(real_files)} real files and {len(spoof_files)} spoof files.")
    print(f"Total samples: {len(real_files) + len(spoof_files)}\n")

    X = []
    y = []
    filenames = []

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
            print(f"  ⚠️  Skipping {filename}: {e}")

    X = np.array(X)
    y = np.array(y)

    return X, y, filenames


def evaluate_loocv(X_physics, y, filenames):
    """
    Leave-One-Out Cross Validation.
    With only 20 samples, this is the most honest evaluation we can do.
    Each iteration: train on 19 samples, test on the 1 left out.
    """
    print("\n" + "=" * 55)
    print("   LEAVE-ONE-OUT CROSS VALIDATION")
    print("=" * 55)

    loo = LeaveOneOut()
    y_true = []
    y_pred = []
    y_prob = []

    for train_idx, test_idx in loo.split(X_physics):
        X_train, X_test = X_physics[train_idx], X_physics[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Scale inside the loop to prevent data leakage
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train logistic regression
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X_train_scaled, y_train)

        prob = clf.predict_proba(X_test_scaled)[0][1]
        pred = int(prob >= 0.5)

        y_true.append(y_test[0])
        y_pred.append(pred)
        y_prob.append(prob)

        label_str = "REAL " if y_test[0] == 0 else "SPOOF"
        result_str = "✅ correct" if pred == y_test[0] else "❌ wrong"
        print(f"  [{label_str}] {filenames[test_idx[0]]:<15} → prob={prob:.3f}  {result_str}")

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    correct = np.sum(y_true == y_pred)
    total = len(y_true)
    accuracy = correct / total * 100

    print(f"\n  Overall LOOCV Accuracy: {correct}/{total} = {accuracy:.1f}%")
    print("\n" + classification_report(y_true, y_pred, target_names=["Real", "Spoof"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    return accuracy


def train_final_model(X_physics, y):
    """
    Train the final model on ALL available data.
    This is what gets saved to disk for inference.
    """
    print("\n" + "=" * 55)
    print("   TRAINING FINAL MODEL ON ALL DATA")
    print("=" * 55)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_physics)

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_scaled, y)

    print(f"  Trained on {len(y)} samples ({np.sum(y==0)} real, {np.sum(y==1)} spoof)")
    print(f"  Physics feature coefficients:")
    feature_names = [
        "Glottal Kurtosis", "Glottal Std", "Spectral Flux Std",
        "Modulation Energy", "Rolloff Std", "ZCR Std",
        "Aliasing Mean", "Aliasing Std"
    ]
    for name, coef in zip(feature_names, clf.coef_[0]):
        direction = "↑ spoof" if coef > 0 else "↓ real"
        print(f"    {name:<22}: {coef:+.4f}  ({direction})")

    return clf, scaler


def main():
    print("=" * 55)
    print("   Ved-Trainer — SPARC Spoof Detector Retrainer")
    print("   EE 123 Project — Ved, Ching, Erick")
    print("=" * 55 + "\n")

    # 1. Load dataset
    print("[1/4] Loading audio files and extracting features...")
    X, y, filenames = load_dataset(AUDIO_DIR)

    # 2. Slice out physics features only
    print(f"\n[2/4] Isolating physics features (indices {PHYSICS_START}+)...")
    X_physics = X[:, PHYSICS_START:]
    print(f"      Feature dims: {X.shape[1]} total → {X_physics.shape[1]} physics features used")

    # 3. Evaluate with LOOCV
    print("\n[3/4] Running Leave-One-Out Cross Validation...")
    accuracy = evaluate_loocv(X_physics, y, filenames)

    # 4. Train final model on all data and save
    print("\n[4/4] Training final model and saving...")
    clf, scaler = train_final_model(X_physics, y)

    model_path = os.path.join(AUDIO_DIR, "model.pkl")
    joblib.dump((clf, scaler), model_path)
    print(f"\n  ✅ Saved model to: {model_path}")

    print("\n" + "=" * 55)
    print(f"   DONE — LOOCV Accuracy: {accuracy:.1f}%")
    print("   Run inference.py to test on new audio files.")
    print("=" * 55)


if __name__ == "__main__":
    main()