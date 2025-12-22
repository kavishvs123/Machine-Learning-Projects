from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

def build_model_pipeline():
    """
    Creates the Stacking Classifier Pipeline.
    """
    # Define Feature Types
    categorical_features = ["Sex", "Embarked", "Title", "CabinLetter", "TicketPrefix"]
    numeric_features = ["Age", "Fare", "Pclass", "FamilySize"]

    # Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", "passthrough", numeric_features)
        ]
    )

    # Define Base Learners
    base_learners = [
        ("rf", RandomForestClassifier(n_estimators=200, random_state=42)),
        ("gb", GradientBoostingClassifier(n_estimators=200, random_state=42)),
        ("svc", SVC(probability=True, random_state=42))
    ]

    # Define Stacking Classifier
    stacking_model = StackingClassifier(
        estimators=base_learners,
        final_estimator=LogisticRegression(max_iter=10000),
        cv=5
    )

    # Build Full Pipeline
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", stacking_model)
    ])
    
    return model