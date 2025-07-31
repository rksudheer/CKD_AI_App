import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# Page setup
st.set_page_config(page_title="CKD Risk Predictor", layout="centered")
st.title("🧠 CKD Risk Predictor (Streamlined)")
st.markdown("Enter patient information to estimate Chronic Kidney Disease (CKD) risk using a trained Random Forest model.")

# Load model and scaler
model = joblib.load("../models/rf_model_streamlined.pkl")
scaler = joblib.load("../models/scaler_streamlined.joblib")

# Define feature names used during training
feature_names = ['RIDAGEYR', 'RIAGENDR', 'LBXGH', 'SMQ020', 'PAQ605']

# SHAP Explainer Setup
explainer = shap.TreeExplainer(model)

# User inputs
age = st.slider("Age", min_value=18, max_value=90, value=45, step=1)
gender = st.selectbox("Gender", options=["Male", "Female"])
prs = st.slider("Simulated PRS (Polygenic Risk Score)", min_value=0.0, max_value=1.0, value=0.4, step=0.01)
smoking = st.radio("Do you currently smoke?", options=["Yes", "No"])
physical = st.radio("Do you do physical activity regularly?", options=["Yes", "No"])

# Encode categorical variables
gender_val = 1 if gender == "Male" else 2
smoking_val = 1 if smoking == "Yes" else 2
physical_val = 2 if physical == "Yes" else 1  # ✅ fixed encoding

# Predict button
if st.button("Predict CKD Risk"):
    input_df = pd.DataFrame([[
        age,
        gender_val,
        prs,
        smoking_val,
        physical_val
    ]], columns=feature_names)

    # Scale input
    input_df_scaled = pd.DataFrame(scaler.transform(input_df), columns=input_df.columns)

    # Predict
    prediction = model.predict(input_df_scaled)
    prediction_proba = model.predict_proba(input_df_scaled)

    # SHAP Explainability
    shap_values = explainer.shap_values(input_df_scaled)

    # Debug Table: Feature Contribution
    shap_debug_df = pd.DataFrame({
        "Feature": input_df.columns,
        "Input Value": input_df.iloc[0],
        "SHAP Value (CKD=Yes)": shap_values[1][0]
    }).sort_values(by="SHAP Value (CKD=Yes)", ascending=False)

    st.subheader("🔍 Feature Contribution (SHAP)")
    st.dataframe(shap_debug_df.style.format(precision=4))

    # Risk Output
    risk = "Yes" if prediction[0] == 1 else "No"
    st.subheader(f"🧬 CKD Risk: {risk}")
    st.text(f"Prediction Confidence (CKD=Yes): {prediction_proba[0][1]*100:.2f}%")

    # SHAP Force Plot
    st.subheader("📊 SHAP Force Plot")
    shap.initjs()
    fig, ax = plt.subplots(figsize=(10, 1))
    shap.force_plot(explainer.expected_value[1], shap_values[1][0], input_df_scaled.iloc[0], matplotlib=True, show=False)
    st.pyplot(fig)
