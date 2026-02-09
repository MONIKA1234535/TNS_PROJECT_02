# Heart Disease Prediction System

A full-stack machine learning application using **FastAPI** (Backend) and **Streamlit** (Frontend) to predict heart disease risk based on clinical parameters.

## 🚀 Project Architecture
- **Model:** PyTorch Neural Network (Binary Classification).
- **Backend:** FastAPI service for high-performance model inference.
- **Frontend:** Streamlit web interface for user input and visualization.
- **Containerization:** Docker & Docker Compose ready.

## 🛠️ How to Run Locally

### 1. Start the Backend
```bash
cd backend
python -m uvicorn main:app --reload --port 8000