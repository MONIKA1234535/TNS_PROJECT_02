import streamlit as st
import requests

# Set page title and icon
st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️")

st.title("❤️ Heart Disease Prediction")
st.write("Enter the patient's clinical data below to get a health assessment.")

# --- Input Fields ---
# We use two columns to make the UI look cleaner
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=50)
    sex = st.selectbox("Sex (1=M, 0=F)", [1, 0])
    cp = st.slider("Chest Pain Type (0-3)", 0, 3, 1)
    trestbps = st.number_input("Resting BP (mm Hg)", value=120)
    chol = st.number_input("Cholesterol (mg/dl)", value=200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 (1=True, 0=False)", [0, 1])

with col2:
    restecg = st.slider("Resting ECG Results (0-2)", 0, 2, 0)
    thalach = st.number_input("Max Heart Rate Achieved", value=150)
    exang = st.selectbox("Exercise Induced Angina (1=Yes, 0=No)", [0, 1])
    oldpeak = st.number_input("ST Depression (Oldpeak)", value=1.0, format="%.1f")
    slope = st.slider("ST Slope (0-2)", 0, 2, 1)
    ca = st.slider("Number of Major Vessels (0-4)", 0, 4, 0)
    thal = st.slider("Thalassemia (0-3)", 0, 3, 1)

# Organize features into a list for the API
features = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]

# --- Prediction Logic ---
if st.button("Analyze Patient Data"):
    try:
        # Send data to the FastAPI backend running on localhost
        res = requests.post("http://localhost:8000/predict", json={"features": features})
        
        if res.status_code == 200:
            result = res.json()
            probability = result['probability']
            
            st.divider()
            if result['prediction'] == 1:
                st.error(f"### ⚠️ Prediction: Heart Disease Detected")
                st.write(f"Confidence Level: **{probability * 100:.2f}%**")
            else:
                st.success(f"### ✅ Prediction: Healthy / Low Risk")
                st.write(f"Confidence Level: **{(1 - probability) * 100:.2f}%**")
        else:
            st.error("Backend Error: Received an invalid response from the server.")
            
    except requests.exceptions.ConnectionError:
        st.error("🚨 Connection Error: Could not reach the Backend.")
        st.info("Make sure your Backend terminal is running with: `python -m uvicorn main:app --reload`")