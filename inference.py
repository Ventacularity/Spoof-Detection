import numpy as np
import joblib
from ching import build_advanced_feature_vector

physical_idx = [0, 2, 4, -2, -1]

model, mu = joblib.load("model.pkl")

def detect(audio_path):
    x = build_advanced_feature_vector(audio_path)
    delta = np.abs(x - mu)
    delta = delta[physical_idx]

    prob = model.predict_proba([delta])[0][1]

    if prob > 0.5:
        print(f"SPOOF (confidence: {prob:.2f})")
    else:
        print(f"REAL (confidence: {1 - prob:.2f})")