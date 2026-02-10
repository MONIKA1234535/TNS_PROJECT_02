import streamlit as st
import requests

st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️"
)

st.title("❤️ Heart Disease Prediction App")
st.write("Enter patient details to predict heart disease risk")

# Input fields (13 features)
features = []
labels = [
    "Age",
    "Sex (1 = Male, 0 = Female)",
    "Chest Pain Type",
    "Resting Blood Pressure",
    "Cholesterol",
    "Fasting Blood Sugar",
    "Resting ECG",
    "Max Heart Rate",
    "Exercise Induced Angina",
    "ST Depression",
    "ST Slope",
    "Number of Major Vessels",
    "Thalassemia"
]

for label in labels:
    value = st.number_input(label, value=0.0)
    features.append(value)

if st.button("Predict"):
    try:
        response = requests.post(
            "https://tns-project-02.onrender.com",
            json={"features": features}
        )

        if response.status_code == 200:
            result = response.json()
            st.success(f"Prediction Probability: {result['probability']:.2f}")
            if result["prediction"] == 1:
                st.error("⚠️ High risk of Heart Disease")
            else:
                st.success("✅ Low risk of Heart Disease")
        else:
            st.error("Backend error. Check FastAPI logs.")

    except Exception as e:
        st.error(f"Could not connect to backend: {e}")
