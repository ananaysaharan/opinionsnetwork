import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
import os
import joblib

def preprocess_data():
    print("--- Day 16: Data Preprocessing (Tensor Setup) ---")
    
    # 1. Load the CSV
    input_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output', 'surrogate_training_data.csv'))
    if not os.path.exists(input_path):
        print(f"Error: File not found at {input_path}")
        return

    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    
    # 2. Prepare Inputs (X)
    # Columns: input_c, input_mu
    X_raw = df[['input_c', 'input_mu']].values
    
    # 3. Prepare Targets (Y)
    # Columns: output_opinion_t1 to output_opinion_t100 (100 steps)
    # We exclude t0 as per the request for shape (5000, 100) and typical time series forecasting where t0 is initial state
    # If the user wanted t0-t99, they would likely specify. t1-t100 captures the evolution.
    # Let's verify the columns exist
    y_cols = [f'output_opinion_t{i}' for i in range(1, 101)]
    Y_raw = df[y_cols].values
    
    print(f"X_raw shape: {X_raw.shape}")
    print(f"Y_raw shape: {Y_raw.shape}")
    
    # 4. Normalize
    print("Normalizing data with MinMaxScaler...")
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    X_scaled = scaler_x.fit_transform(X_raw)
    Y_scaled = scaler_y.fit_transform(Y_raw)
    
    # 5. Convert to PyTorch Tensors
    print("Converting to PyTorch Tensors...")
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    Y_tensor = torch.tensor(Y_scaled, dtype=torch.float32)
    
    print(f"X_tensor shape: {X_tensor.shape}")
    print(f"Y_tensor shape: {Y_tensor.shape}")
    
    # Save tensors for next steps
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output'))
    torch.save(X_tensor, os.path.join(output_dir, 'surrogate_X.pt'))
    torch.save(Y_tensor, os.path.join(output_dir, 'surrogate_Y.pt'))
    
    # Save scalers too, we might need them to inverse transform later
    joblib.dump(scaler_x, os.path.join(output_dir, 'surrogate_scaler_x.pkl'))
    joblib.dump(scaler_y, os.path.join(output_dir, 'surrogate_scaler_y.pkl'))
    
    print("Preprocessing complete. Tensors and scalers saved to output directory.")

if __name__ == "__main__":
    preprocess_data()
