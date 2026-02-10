import os
import torch
import joblib
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "saved_models", "final_heart_model.pth")
SCALER_PATH = os.path.join(BASE_DIR, "saved_models", "scaler.pkl")

model = torch.load(MODEL_PATH, map_location="cpu")
model.eval()

scaler = joblib.load(SCALER_PATH)

def predict_heart_disease(features):
    # 🔴 THIS LINE IS CRITICAL
    features = np.array(features).reshape(1, -1)

    features = scaler.transform(features)
    features = torch.tensor(features, dtype=torch.float32)

    with torch.no_grad():
        output = model(features)

    return int(output.argmax(dim=1).item())
