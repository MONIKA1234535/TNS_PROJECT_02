import torch
import joblib
from .model import HeartModel

MODEL_PATH = "saved_models/final_heart_model.pth"
SCALER_PATH = "saved_models/scaler.pkl"

model = HeartModel()
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

scaler = joblib.load(SCALER_PATH)

def predict_heart_disease(features):
    scaled = scaler.transform([features])
    tensor = torch.FloatTensor(scaled)
    with torch.no_grad():
        prob = model(tensor).item()
    return prob
