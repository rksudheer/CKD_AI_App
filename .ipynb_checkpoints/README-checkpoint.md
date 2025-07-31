# CKD_AI_App

An interactive AI application to predict Chronic Kidney Disease (CKD) using simulated Polygenic Risk Scores (PRS), clinical, and lifestyle data. Built with scikit-learn and Streamlit.

---

## 🔧 Project Structure
CKD_AI_App/
├── app/
│ └── app.py # Streamlit frontend (main entry point)
├── data/
│ └── ckd_simulated_input.csv # Streamlined dataset with 6 features
├── models/
│ ├── scaler_minimal.joblib # Scaler for app inputs
│ ├── rf_minimal_model.pkl # Streamlined Random Forest model
│ ├── random_forest_model.joblib # Full RF model (SMOTE)
│ ├── logistic_regression_model.pkl
│ └── shap_explainer_rf.joblib # SHAP explainer object
├── notebooks/
│ ├── 01_logistic_regression.ipynb
│ ├── 02_random_forest.ipynb
│ ├── 03_explain_rf_with_shap.ipynb
│ └── 04_streamlit_prototype.ipynb

---

## ✅ How to Run the Streamlit App

1. Open Anaconda Prompt or Terminal
2. Navigate to the project folder:
   ```bash
   cd C:/Users/DELL/CKD_AI_App

 streamlit run app/app.py