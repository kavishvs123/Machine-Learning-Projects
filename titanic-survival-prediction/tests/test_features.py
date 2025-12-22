import pandas as pd
import pytest
from src.features.preprocess import preprocess_data

@pytest.fixture
def raw_titanic_data():
    """Creates a small dataframe for testing."""
    data = {
        "PassengerId": [1, 2, 3],
        "Name": [
            "Braund, Mr. Owen Harris",
            "Cumings, Mrs. John Bradley (Florence Briggs Thayer)",
            "Heikkinen, Miss. Laina"
        ],
        "Sex": ["male", "female", "female"],
        "Age": [22.0, 38.0, None], # Includes a missing val
        "SibSp": [1, 1, 0],
        "Parch": [0, 0, 0],
        "Ticket": ["A/5 21171", "PC 17599", "STON/O2. 3101282"],
        "Fare": [7.25, 71.28, 7.92],
        "Cabin": [None, "C85", None],
        "Embarked": ["S", "C", "S"],
        "Pclass": [3, 1, 3]
    }
    return pd.DataFrame(data)

def test_preprocess_generates_expected_columns(raw_titanic_data):
    # Act
    processed_df = preprocess_data(raw_titanic_data)
    
    # Assert
    expected_cols = ["Title", "FamilySize", "CabinLetter", "TicketPrefix"]
    for col in expected_cols:
        assert col in processed_df.columns, f"Column {col} is missing!"

def test_title_extraction(raw_titanic_data):
    # Act
    processed_df = preprocess_data(raw_titanic_data)
    
    # Assert
    # 1st passenger is Mr., 2nd is Mrs.
    assert processed_df.iloc[0]["Title"] == "Mr"
    assert processed_df.iloc[1]["Title"] == "Mrs"

def test_family_size_calculation(raw_titanic_data):
    # Act
    processed_df = preprocess_data(raw_titanic_data)
    
    # Assert
    # Passenger 1: SibSp 1 + Parch 0 + 1 = 2
    assert processed_df.iloc[0]["FamilySize"] == 2

def test_missing_age_imputation(raw_titanic_data):
    # Act
    processed_df = preprocess_data(raw_titanic_data)
    
    # Assert
    # Passenger 3 had NaN Age. It should be filled now.
    assert not pd.isnull(processed_df.iloc[2]["Age"])
