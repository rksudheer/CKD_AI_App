# CKD_AI_App/src/data_loader.py

import pandas as pd
import os

df = load_ckd_data("data/ckd_simulated_input.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found.")
    return pd.read_csv(file_path)