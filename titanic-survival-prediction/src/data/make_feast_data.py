import pandas as pd
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Import your preprocessing function
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.features.preprocess import preprocess_data

def create_feast_source():
    # 1. Define Paths Dynamically
    current_dir = Path(__file__).resolve().parent
    
    # Go up two levels to get the root (titanic-survival-predictions)
    project_root = current_dir.parent.parent
    
    # Define input and output paths relative to root
    input_path = project_root / "data" / "titanic_train.csv"
    output_dir = project_root / "feature_repo" / "data"
    output_path = output_dir / "titanic_features.parquet"

    print(f"Project Root detected at: {project_root}")

    # 2. Load Raw Data
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found at: {input_path}")
    
    df = pd.read_csv(input_path)
    
    # 3. Process it (Clean/Feature Engineer)
    df = preprocess_data(df)
    
    # 4. Add Feast Requirements
    # Add timestamp (set to yesterday so it's valid for retrieval)
    df['event_timestamp'] = pd.Timestamp.now() - timedelta(days=1)
    
    # Ensure ID column exists (reload if it was dropped during preprocessing)
    if "PassengerId" not in df.columns:
        raw = pd.read_csv(input_path)
        df["PassengerId"] = raw["PassengerId"]

    # 5. Create Directory and Save
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df.to_parquet(output_path)
    print(f"Titanic features saved to: {output_path}")

if __name__ == "__main__":
    create_feast_source()