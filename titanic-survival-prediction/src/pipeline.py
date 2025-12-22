import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Import our custom modules
from models.train import build_model_pipeline
from src.data.feast_loader import get_training_data_from_feast

def run_training_pipeline():
    print("Starting Pipeline")

    print("\n Fetching data from Feature Store...")
    train_df = get_training_data_from_feast()

    # 2. Preprocess
    print("\n Processing Data")
    y = train_df["Survived"]
    X = train_df.drop("Survived", axis=1)
    
    X_processed = preprocess_data(X)
    test_processed = preprocess_data(test_df)

    print(f"   Processed Train Shape: {X_processed.shape}")

    # 3. Split Data
    X_train, X_val, y_train, y_val = train_test_split(X_processed, y, test_size=0.2, random_state=42)

    # 4. Build & Train Model
    print("\n Training Stacking Classifier")
    model = build_model_pipeline()
    model.fit(X_train, y_train)
    print("   Training Complete.")

    # 5. Evaluate
    print("\n Evaluation on Validation Set:")
    y_pred = model.predict(X_val)
    print(f"   Accuracy:  {accuracy_score(y_val, y_pred):.4f}")
    print(f"   Precision: {precision_score(y_val, y_pred):.4f}")
    print(f"   Recall:    {recall_score(y_val, y_pred):.4f}")
    print(f"   F1 Score:  {f1_score(y_val, y_pred):.4f}")

    # 6. Generate Submission
    print("\n Generating Submission")
    test_predictions = model.predict(test_processed)
    
    submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": test_predictions
    })
    
    submission.to_csv("submission.csv", index=False)
    print("Saved to submission.csv")

if __name__ == "__main__":
    run_training_pipeline()