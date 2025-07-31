import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import os
from matplotlib import rcParams
import sys

# Set page config
st.set_page_config(
    page_title="CKD Risk Prediction",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
<style>
    .big-font {
        font-size:18px !important;
    }
    .result-box {
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .high-risk {
        background-color: #ffcccc;
        border-left: 5px solid #ff0000;
    }
    .low-risk {
        background-color: #ccffcc;
        border-left: 5px solid #00aa00;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
        border-left: 5px solid #ffc107;
    }
    .success-box {
        background-color: #d4edda;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
        border-left: 5px solid #28a745;
    }
</style>
""", unsafe_allow_html=True)

# Load model and scaler with error handling
@st.cache_resource
def load_assets():
    try:
        model_path = os.path.join("models", "rf_model_streamlined.pkl")
        scaler_path = os.path.join("models", "scaler_streamlined.joblib")
        
        if not os.path.exists(model_path):
            st.error(f"Model file not found at {model_path}")
            sys.exit(1)
        if not os.path.exists(scaler_path):
            st.error(f"Scaler file not found at {scaler_path}")
            sys.exit(1)
            
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model/scaler: {str(e)}")
        sys.exit(1)

model, scaler = load_assets()

# Define streamlined feature columns
feature_names = ["RIDAGEYR", "RIAGENDR", "LBXGH", "SMQ020", "PAQ605"]
display_names = {
    "RIDAGEYR": "Age",
    "RIAGENDR": "Gender",
    "LBXGH": "Genetic Risk Score",
    "SMQ020": "Smoking Status",
    "PAQ605": "Physical Activity"
}

# Configure matplotlib
rcParams['font.size'] = 10
rcParams['axes.titlepad'] = 15

# Sidebar
with st.sidebar:
    st.title("🩺 CKD Risk Prediction")
    st.markdown("""
    **How to use:**
    1. Fill in your details
    2. Click 'Predict CKD Risk'
    3. View results and explanations
    """)
    st.markdown("---")
    st.markdown("Built with ❤️ using Streamlit + SHAP")

# Main content
st.title("Chronic Kidney Disease Risk Assessment")
st.markdown("Predict your risk of developing CKD using clinical and lifestyle factors")

# User input section
with st.expander("📝 Enter Your Information", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age", 18, 90, 50, help="Select your current age")
        gender = st.selectbox("Gender", options=[("Male", 1), ("Female", 2)], 
                            format_func=lambda x: x[0], help="Select your biological sex")
    with col2:
        prs = st.slider("Genetic Risk Score (PRS)", 0.0, 1.0, 0.5, 0.01,
                       help="Polygenic risk score for kidney function")
    
    st.markdown("---")
    col3, col4 = st.columns(2)
    with col3:
        smoking = st.selectbox("Do you currently smoke?", 
                             options=[("Yes", 1), ("No", 2)], 
                             format_func=lambda x: x[0])
    with col4:
        physical_activity = st.selectbox("Daily physical activity (10+ min)", 
                                       options=[("Yes", 1), ("No", 2)], 
                                       format_func=lambda x: x[0])

# Prepare input
input_df = pd.DataFrame([[age, gender[1], prs, smoking[1], physical_activity[1]]], 
                       columns=feature_names)

try:
    input_scaled = scaler.transform(input_df)
except Exception as e:
    st.error(f"Error scaling input: {str(e)}")
    st.stop()

# Prediction section
if st.button("🔍 Predict CKD Risk", use_container_width=True):
    with st.spinner("Analyzing your risk profile..."):
        try:
            prediction = model.predict(input_scaled)[0]
            proba = model.predict_proba(input_scaled)
            
            # Handle different probability output formats
            if proba.shape[1] == 2:  # Binary classification
                prob = proba[0][1]
            else:  # Possibly regression or one-class
                prob = proba[0][0]
            
            # Display results
            st.subheader("Results")
            if prediction == 1:
                st.markdown(f"""
                <div class="result-box high-risk">
                    <h3>🔴 High Risk of CKD</h3>
                    <p class="big-font">Probability: <strong>{prob:.1%}</strong></p>
                    <p>Consult your healthcare provider for further evaluation.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-box low-risk">
                    <h3>🟢 Low Risk of CKD</h3>
                    <p class="big-font">Probability: <strong>{prob:.1%}</strong></p>
                    <p>Maintain healthy habits to keep your kidneys healthy!</p>
                </div>
                """, unsafe_allow_html=True)
            
            # SHAP explanation with guaranteed visualization
            st.subheader("📊 Risk Factor Analysis")
            st.markdown("""
            Below you can see which factors contributed most to your prediction.
            Positive values increase risk, negative values decrease risk.
            """)
            
            try:
                # Initialize SHAP explainer
                explainer = shap.TreeExplainer(model)
                
                # Calculate SHAP values - guaranteed to work with new approach
                try:
                    # New foolproof SHAP calculation
                    shap_values = explainer(input_scaled)
                    
                    # Handle different SHAP output formats
                    if hasattr(shap_values, 'values'):
                        values = shap_values.values
                    else:
                        values = shap_values
                    
                    # Ensure we have correct shape (1, n_features)
                    if len(values.shape) == 3:
                        # For multi-class models, take values for predicted class
                        values = values[0, :, prediction]
                    elif len(values.shape) == 2:
                        values = values[0]
                    else:
                        values = values
                        
                    # Reshape to (1, n_features) if needed
                    if len(values) == len(feature_names):
                        values = values.reshape(1, -1)
                    
                except Exception as e:
                    raise Exception(f"SHAP calculation failed: {str(e)}")
                
                # Create guaranteed visualization
                with st.expander("Feature Impact Analysis", expanded=True):
                    try:
                        # Convert to proper format
                        if values.shape[0] != 1:
                            values = values[:1]  # Take first sample only
                        
                        # Create impact dataframe
                        impact_df = pd.DataFrame({
                            'Feature': [display_names.get(f, f) for f in feature_names],
                            'Impact': values[0]
                        }).sort_values('Impact', key=abs, ascending=False)
                        
                        # Plot using matplotlib (always works)
                        fig, ax = plt.subplots(figsize=(10, 5))
                        colors = ['#ff6b6b' if x > 0 else '#6bafff' for x in impact_df['Impact']]
                        bars = ax.barh(impact_df['Feature'], impact_df['Impact'], color=colors)
                        
                        # Add value labels
                        for bar in bars:
                            width = bar.get_width()
                            ax.text(width + (0.01 if width > 0 else -0.01), 
                                   bar.get_y() + bar.get_height()/2,
                                   f"{width:.3f}", 
                                   va='center',
                                   ha='left' if width > 0 else 'right')
                        
                        ax.set_xlabel("Impact on Prediction")
                        ax.set_title("How Each Factor Affects Your Risk")
                        ax.grid(True, linestyle='--', alpha=0.3)
                        st.pyplot(fig, use_container_width=True)
                        plt.close()
                        
                        # Show numerical values
                        st.dataframe(
                            impact_df.style.format({'Impact': '{:.3f}'}),
                            hide_index=True,
                            use_container_width=True
                        )
                        
                        st.markdown("""
                        <div class="success-box">
                            <p>✅ Feature impact analysis completed successfully</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    except Exception as e:
                        st.markdown(f"""
                        <div class="warning-box">
                            <p>⚠️ Could not generate visualization. Showing SHAP values:</p>
                        </div>
                        """, unsafe_allow_html=True)
                        st.write(values)
            
            except Exception as e:
                st.markdown(f"""
                <div class="warning-box">
                    <p>⚠️ Could not generate detailed explanation. Error: {str(e)}</p>
                    <p>Here are the most important features from the model:</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Fallback: Show feature importances
                try:
                    if hasattr(model, 'feature_importances_'):
                        importances = pd.DataFrame({
                            'Feature': [display_names.get(f, f) for f in feature_names],
                            'Importance': model.feature_importances_
                        }).sort_values('Importance', ascending=False)
                        
                        st.dataframe(
                            importances.style.format({'Importance': '{:.3f}'}),
                            hide_index=True,
                            use_container_width=True
                        )
                except:
                    st.write("Feature importance not available for this model type")
        
        except Exception as e:
            st.error(f"An error occurred during prediction: {str(e)}")

# Footer
st.markdown("---")
st.caption("Note: This tool provides risk estimates only and should not replace professional medical advice.")