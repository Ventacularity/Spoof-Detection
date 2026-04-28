import os
import numpy as np
from ching import build_advanced_feature_vector
from sklearn.linear_model import LogisticRegression
import joblib

BASE_PATH = os.path.expanduser("~/Downloads/spoofer")

physical_idx = [0, 2, 4, -2, -1]

X = []
y = []

for folder in os.listdir(BASE_PATH):
    folder_path = os.path.join(BASE_PATH, folder)

    if not os.path.isdir(folder_path):
        continue

    files = os.listdir(folder_path)

    real_file = None
    spoof_file = None

    for f in files:
        if "gt" in f:
            real_file = os.path.join(folder_path, f)
        elif "resynth" in f:
            spoof_file = os.path.join(folder_path, f)

    if real_file and spoof_file:
        x_real = build_advanced_feature_vector(real_file)
        x_spoof = build_advanced_feature_vector(spoof_file)

        # spoof example
        delta = np.abs(x_real - x_spoof)
        X.append(delta[physical_idx])
        y.append(1)

        # real baseline
        delta_real = np.zeros_like(delta)
        X.append(delta_real[physical_idx])
        y.append(0)

# Train model
model = LogisticRegression()
model.fit(X, y)

# Compute mean real vector
real_features = []
for folder in os.listdir(BASE_PATH):
    folder_path = os.path.join(BASE_PATH, folder)
    if not os.path.isdir(folder_path):
        continue

    for f in os.listdir(folder_path):
        if "gt" in f:
            path = os.path.join(folder_path, f)
            real_features.append(build_advanced_feature_vector(path))

mu = np.mean(real_features, axis=0)

# Save everything
joblib.dump((model, mu), "model.pkl")

print("Training complete. Saved model.pkl")