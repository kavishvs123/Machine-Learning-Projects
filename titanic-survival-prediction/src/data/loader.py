import pandas as pd
import os

def load_data(file_path):
    """
    Loads CSV data from a specific path.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f" The file {file_path} was not found.")
    
    df = pd.read_csv(file_path)
    print(f"Data loaded from {file_path} with shape {df.shape}")
    return df