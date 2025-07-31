import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# Load model and scaler
model = joblib.load("../models/rf_model_streamlined.pkl")
scaler = joblib.load("../models/scaler_streamlined.joblib")

# SHAP setup
shap.initjs()

# Page config
st.set_page_config(page_title="CKD Risk Predictor", layout="centered")
st.title("🧬 Chronic Kidney Disease (CKD) Risk Predictor")
st.markdown("This tool predicts the risk of CKD based on your health and genetic profile.")

# Sidebar inputs
st.sidebar.header("Enter your information")

age = st.sidebar.slider("Age", 18, 90, 45)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
prs = st.sidebar.slider("Simulated PRS Score", 0.5, 3.0, 1.5, step=0.1)
smoking = st.sidebar.selectbox("Do you smoke?", ["Yes", "No"])
physical_activity = st.sidebar.selectbox("Physically active?", ["Yes", "No"])

# Map categorical inputs to numeric
gender_val = 1 if gender == "Male" else 2
smoking_val = 1 if smoking == "Yes" else 2
activity_val = 1 if physical_activity == "Yes" else 2

# Form input DataFrame
input_data = pd.DataFrame([{
    "RIDAGEYR": age,
    "RIAGENDR": gender_val,
    "LBXGH": prs,
    "SMQ020": smoking_val,
    "PAQ605": activity_val
}])

# Scale inputs
input_scaled = scaler.transform(input_data)

# Predict button
if st.button("🔍 Predict CKD Risk"):
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.error(f"⚠️ High risk of CKD! Probability: {probability:.2f}")
    else:
        st.success(f"✅ Low risk of CKD. Probability: {probability:.2f}")

    # SHAP explanation
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_scaled)

    st.markdown("### 🔍 SHAP Explanation (Why this prediction?)")
    fig, ax = plt.subplots()
    shap.force_plot(
        explainer.expected_value[1],
        shap_values[1][0],
        input_data,
        matplotlib=True,
        show=False
    )
    st.pyplot(fig)
