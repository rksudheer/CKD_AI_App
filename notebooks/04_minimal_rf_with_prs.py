# 04_minimal_rf_with_prs.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os

# 📍 Set absolute path to models directory
model_dir = os.path.abspath("../../models/")
os.makedirs(model_dir, exist_ok=True)

# 📂 Confirm current working directory
print("📂 Current Working Directory:", os.getcwd())
print("📁 Model directory set to:", model_dir)

# 🧠 Load data
import os

data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/ckd_simulated_input.csv"))
print("📄 Loading data from:", data_path)
df = pd.read_csv(data_path)


# 🎯 Define features (minimal)
features = ['RIDAGEYR', 'RIAGENDR', 'LBXGH', 'SMQ020', 'PAQ605']  # Age, Gender, PRS, Smoking, Physical Activity
target = 'CKD'

X = df[features]
y = df[target]

# 🧼 Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 🤖 Model training
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

# 🧪 Evaluation
y_pred = model.predict(X_test)
print("\n✅ Classification Report:\n", classification_report(y_test, y_pred))
print("\n🧾 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

import os

# Force correct save directory
save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../models"))
os.makedirs(save_dir, exist_ok=True)

model_path = os.path.join(save_dir, "minimal_rf_model.pkl")
scaler_path = os.path.join(save_dir, "minimal_scaler.joblib")

joblib.dump(model, model_path)
joblib.dump(scaler, scaler_path)

print(f"✅ Model saved to {model_path}")
print(f"✅ Scaler saved to {scaler_path}")

print(f"\n✅ Scaler saved to: {scaler_path}")
print(f"✅ Model saved to:  {model_path}")