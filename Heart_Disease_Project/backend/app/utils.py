import os
import torch
import joblib
import numpy as np
from .model import HeartModel

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "saved_models", "final_heart_model.pth")
SCALER_PATH = os.path.join(BASE_DIR, "saved_models", "scaler.pkl")

# 1️⃣ Create model object
model = HeartModel(input_size=13)

# 2️⃣ Load weights
state_dict = torch.load(MODEL_PATH, map_location="cpu")
model.load_state_dict(state_dict)

# 3️⃣ Set eval mode
model.eval()

# Load scaler
scaler = joblib.load(SCALER_PATH)

def predict_heart_disease(features):
    features = np.array(features).reshape(1, -1)
    features = scaler.transform(features)
    features = torch.tensor(features, dtype=torch.float32)

    with torch.no_grad():
        output = model(features)

    return int(output.argmax(dim=1).item())
