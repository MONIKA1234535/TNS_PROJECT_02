from fastapi import FastAPI
from pydantic import BaseModel
import torch
from model import HeartModel

app = FastAPI()

# Load Model
model = HeartModel()
model.load_state_dict(torch.load("final_heart_model.pth", map_location=torch.device('cpu')))
model.eval()

class PatientData(BaseModel):
    features: list

@app.post("/predict")
def predict(data: PatientData):
    input_tensor = torch.FloatTensor([data.features])
    with torch.no_grad():
        output = model(input_tensor).item()
    
    prediction = 1 if output >= 0.5 else 0
    return {"prediction": prediction, "probability": round(output, 4)}