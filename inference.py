"""
inference.py - Run spoof detection on a single audio file
EE 123 Project - Ved, Ching, Erick

Uses model.pkl trained by Ved-Trainer.py on 3 features:
  - Glottal Kurtosis   (real > spoof in 6/8 pairs, 17% avg diff)
  - Glottal Std        (real > spoof in 7/8 pairs, 13% avg diff)
  - Modulation Energy  (real > spoof in 7/8 pairs, 21% avg diff)

Usage:
    python inference.py <audio_file.wav>
"""

import sys
import os
import numpy as np
import joblib
from ching import build_advanced_feature_vector

PHYSICS_START = 120
SELECTED_INDICES = [0, 1, 3]  # Glottal Kurtosis, Glottal Std, Modulation Energy
SELECTED_NAMES = ["Glottal Kurtosis", "Glottal Std", "Modulation Energy"]
THRESHOLD = 0.45

def detect(audio_path, model_path=None):
    if model_path is None:
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.pkl")

    if not os.path.exists(model_path):
        print(f"Error: model.pkl not found at {model_path}")
        print("Run Ved-Trainer.py first to train the model.")
        return

    # Load model — handles both old format (2 values) and new format (4 values)
    loaded = joblib.load(model_path)
    if len(loaded) == 4:
        clf, scaler, sel_idx, phys_start = loaded
    else:
        clf, scaler = loaded
        sel_idx = SELECTED_INDICES
        phys_start = PHYSICS_START

    print(f"\nAnalyzing: {os.path.basename(audio_path)}")
    print("-" * 45)

    # Extract features
    try:
        full_vec = build_advanced_feature_vector(audio_path)
    except Exception as e:
        print(f"Error extracting features: {e}")
        return

    # Select the 3 relevant physics features
    physics = full_vec[phys_start:]
    x = physics[sel_idx].reshape(1, -1)

    # Print feature values for transparency
    for name, val in zip(SELECTED_NAMES, x[0]):
        print(f"  {name:<22}: {val:.4f}")

    # Scale and predict
    x_scaled = scaler.transform(x)
    prob_spoof = clf.predict_proba(x_scaled)[0][1]
    prob_real = 1 - prob_spoof

    print(f"\n  Spoof probability : {prob_spoof:.4f}")
    print(f"  Real  probability : {prob_real:.4f}")
    print(f"  Threshold         : {THRESHOLD}")
    print("-" * 45)

    if prob_spoof >= THRESHOLD:
        print(f"  VERDICT: SPOOF (confidence: {prob_spoof*100:.1f}%)")
    else:
        print(f"  VERDICT: REAL  (confidence: {prob_real*100:.1f}%)")
    print("-" * 45)

    return prob_spoof


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inference.py <audio_file.wav>")
        sys.exit(1)
    detect(sys.argv[1])