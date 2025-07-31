import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# Set Streamlit page config
st.set_page_config(page_title="CKD Risk Predictor", layout="centered")

# Load model and scaler
model = joblib.load("../models/rf_model_streamlined.pkl")
scaler = joblib.load("../models/scaler_streamlined.joblib")

# Define feature names
feature_names = ['RIDAGEYR', 'RIAGENDR', 'LBXGH', 'SMQ020', 'PAQ605']

# Function to get user input
def get_user_input():
    st.subheader("📝 Enter Patient Details")

    age = st.number_input("Age", min_value=10, max_value=90, value=45)
    gender = st.selectbox("Gender", ["Male", "Female"])
    prs = st.slider("Polygenic Risk Score (Simulated)", min_value=0.0, max_value=1.0, value=0.55)
    smoking = st.selectbox("Smoking Status", ["Yes", "No"])
    physical = st.selectbox("Physical Activity (≥10 min)", ["Yes", "No"])

    # Convert to model-usable values
    gender_val = 1 if gender == "Male" else 2
    smoking_val = 1 if smoking == "Yes" else 2
    physical_val = 2 if physical == "Yes" else 1

    input_dict = {
        'RIDAGEYR': age,
        'RIAGENDR': gender_val,
        'LBXGH': prs,
        'SMQ020': smoking_val,
        'PAQ605': physical_val
    }

    return pd.DataFrame([input_dict])

# Function to simulate test patient
def load_test_patient(n=1):
    # Use pre-defined test inputs
    if n == 1:
        return pd.DataFrame([{
            'RIDAGEYR': 65,
            'RIAGENDR': 1,
            'LBXGH': 0.75,
            'SMQ020': 1,
            'PAQ605': 2
        }])
    elif n == 2:
        return pd.DataFrame([{
            'RIDAGEYR': 40,
            'RIAGENDR': 2,
            'LBXGH': 0.22,
            'SMQ020': 2,
            'PAQ605': 1
        }])

# Title
st.title("🧠 CKD Prediction App (Streamlined)")

# Option to select mode
mode = st.radio("Choose Input Mode:", ["🧑‍⚕️ Enter Details", "🧪 Test Patient 1", "🧪 Test Patient 2"])

if mode == "🧑‍⚕️ Enter Details":
    input_df = get_user_input()
elif mode == "🧪 Test Patient 1":
    input_df = load_test_patient(1)
elif mode == "🧪 Test Patient 2":
    input_df = load_test_patient(2)

# Scale the input
input_scaled = scaler.transform(input_df)

# Predict and explain
if st.button("🔍 Predict CKD Risk"):
    pred = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]

    st.subheader("🔎 Prediction Result")
    st.write(f"**CKD Risk:** {'Yes' if pred == 1 else 'No'}")
    st.write(f"**Probability of CKD:** {prob:.2f}")

    # SHAP explanation
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_scaled)

    st.subheader("🧠 SHAP Explanation")
    try:
        st.set_option('deprecation.showPyplotGlobalUse', True)
        shap.force_plot(explainer.expected_value[1], shap_values[1][0], input_df.iloc[0], matplotlib=True, show=False)
        st.pyplot(bbox_inches='tight')
    except Exception as e:
        st.error(f"SHAP explanation failed: {e}")

    # Optional: show feature values
    st.subheader("📋 Input Summary")
    st.write(input_df)
