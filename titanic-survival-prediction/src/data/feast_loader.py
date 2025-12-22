import feast
import pandas as pd
import os

def get_training_data_from_feast(repo_path="feature_repo"):
    """
    Fetches historical training data from the Feast offline store.
    """
    # 1. Initialize the Feature Store
    store = feast.FeatureStore(repo_path=repo_path)

    # 2. Define the "Entity DataFrame" (The Labels)
    # We use the original CSV to get the IDs and Ground Truth (Survived).
    # MLOps - CSV acts as "Label Store".
    
    # We use the relative path logic from before to find the data folder
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    label_path = os.path.join(root_dir, "data/titanic_train.csv")
    
    entity_df = pd.read_csv(label_path)[["PassengerId", "Survived"]]
    
    # Add the timestamp to ask for the data "as of now"
    entity_df["event_timestamp"] = pd.Timestamp.now()

    # 3. Request Features
    feature_vector = [
        "titanic_passenger_stats:Pclass",
        "titanic_passenger_stats:Age",
        "titanic_passenger_stats:FamilySize",
        "titanic_passenger_stats:Fare",
        "titanic_passenger_stats:Sex",
        "titanic_passenger_stats:Embarked",
        "titanic_passenger_stats:Title",
        "titanic_passenger_stats:CabinLetter",
        "titanic_passenger_stats:TicketPrefix",
    ]

    training_df = store.get_historical_features(
        entity_df=entity_df,
        features=feature_vector
    ).to_df()

    # 4. Clean up
    # Feast adds 'event_timestamp'; we can drop it for training.
    training_df.drop(columns=["event_timestamp"], inplace=True)
    
    print(f"Fetched {training_df.shape} rows from Feast.")
    return training_df