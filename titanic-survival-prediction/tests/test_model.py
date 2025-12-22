import pandas as pd
from src.models.train import build_model_pipeline

def test_model_pipeline_can_fit_and_predict():
    # 1. Arrange: Create dummy data with >= 5 examples per class
    # We need at least 5 rows where Survived=0 and 5 where Survived=1
    data = {
        "Age": [22, 38, 26, 35, 35, 28, 50, 19, 25, 40],
        "Fare": [7.25, 71.28, 7.92, 53.1, 8.05, 8.45, 51.8, 26.0, 13.0, 30.0],
        "Pclass": [3, 1, 3, 1, 3, 3, 1, 2, 2, 2],
        "FamilySize": [2, 2, 1, 2, 1, 1, 1, 3, 1, 2],
        "Sex": ["male", "female", "female", "female", "male", "male", "male", "female", "female", "male"],
        "Embarked": ["S", "C", "S", "S", "S", "Q", "S", "S", "C", "C"],
        "Title": ["Mr", "Mrs", "Miss", "Mrs", "Mr", "Mr", "Mr", "Miss", "Mrs", "Mr"],
        "CabinLetter": ["U", "C", "U", "C", "U", "U", "E", "U", "U", "U"],
        "TicketPrefix": ["A/5", "PC", "STON/O2", "113803", "373450", "330877", "17463", "248738", "237736", "248698"]
    }
    
    X = pd.DataFrame(data)
    
    # Target must have at least 5 zeros and 5 ones
    y = pd.Series([0, 1, 1, 1, 0, 0, 0, 1, 1, 0]) 

    # 2. Act
    model = build_model_pipeline()
    model.fit(X, y)
    predictions = model.predict(X)

    # 3. Assert
    assert len(predictions) == 10
    assert isinstance(model, object)