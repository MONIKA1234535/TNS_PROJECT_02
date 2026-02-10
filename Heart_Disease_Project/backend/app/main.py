from fastapi import FastAPI
from .schemas import PredictionRequest, PredictionResponse
from .utils import predict_heart_disease

app = FastAPI(title="Heart Disease Predictor")

@app.get("/")
def root():
    return {"status": "Backend running"}

@app.post("/predict", response_model=PredictionResponse)
def predict(data: PredictionRequest):
    prob = predict_heart_disease(data.features)
    return {
        "probability": prob,
        "prediction": 1 if prob > 0.5 else 0
    }
