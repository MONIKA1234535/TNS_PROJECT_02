import os
import torch
import joblib
import numpy as np
from .model import HeartModel

# Base directory: backend/app → backend
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Paths to saved model files
MODEL_PATH = os.path.join(BASE_DIR, "saved_models", "final_heart_model.pth")
SCALER_PATH = os.path.join(BASE_DIR, "saved_models", "scaler.pkl")

# Load model architecture
model = HeartModel(input_size=13)

# Load trained weights (state_dict)
state_dict = torch.load(MODEL_PATH, map_location="cpu")
model.load_state_dict(state_dict)

# Set model to evaluation mode
model.eval()

# Load scaler
scaler = joblib.load(SCALER_PATH)


def predict_heart_disease(features):
    """
    features: list of 13 numerical values
    returns: 0 (No Heart Disease) or 1 (Heart Disease)
    """

    # Convert input to numpy array and reshape
    features = np.array(features, dtype=float).reshape(1, -1)

    # Scale features
    features = scaler.transform(features)

    # Convert to tensor
    features_tensor = torch.tensor(features, dtype=torch.float32)

    # Model inference
    with torch.no_grad():
        output = model(features_tensor)          # shape: [1, 1]
        probability = torch.sigmoid(output).item()
        prediction = 1 if probability >= 0.5 else 0

    return prediction
