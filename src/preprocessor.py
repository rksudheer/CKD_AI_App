# src/preprocessor.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_split_data(csv_path, target_column='ckd', test_size=0.2, random_state=42):
    """
    Loads CSV data, separates features and target, splits into train and test.
    """
    df = pd.read_csv(csv_path)
    
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset.")
    
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)

def scale_data(X_train, X_test):
    """
    Standardizes training and testing data using StandardScaler.
    Returns the scaler object as well.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, scaler