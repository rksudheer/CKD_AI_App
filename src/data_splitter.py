# CKD_AI_App/src/data_splitter.py

import pandas as pd
from sklearn.model_selection import train_test_split

def split_data(df, target_column="CKD", test_size=0.2, random_state=42):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)