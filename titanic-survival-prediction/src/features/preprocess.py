import pandas as pd
import numpy as np
import re

def extract_title(name):
    """Helper function to extract title from name string."""
    
    search = re.search(' ([A-Za-z]+)\.', name)
    if search:
        return search.group(1)
    return ""

def preprocess_data(df, is_train=True):
    """
    Applies all cleaning and feature engineering steps.
    """
    df = df.copy()
    
    # 1. Title Extraction
    df['Title'] = df['Name'].apply(extract_title)
    
    # Normalize Titles
    df['Title'] = df['Title'].replace(
        ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 
        'Rare'
    )
    df['Title'] = df['Title'].replace({'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'})

    # 2. Family Size
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

    # 3. Cabin Letter
    df['CabinLetter'] = df['Cabin'].fillna('U').str[0]

    # 4. Ticket Prefix
    df['TicketPrefix'] = df['Ticket'].apply(lambda x: x.split()[0] if not x.split()[0].isdigit() else 'None')

    # 5. Handling Missing Values (Simple Logic from Notebook)
    # Note: In a pure production system, we would calculate mode/medians on Train and apply to Test.
    # For now, we replicate your notebook logic.
    
    if 'Embarked' in df.columns:
        df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    
    # Pclass-based imputation for Fare and Age
    if 'Fare' in df.columns:
        df['Fare'] = df['Fare'].fillna(df.groupby("Pclass")["Fare"].transform("median"))
        
    if 'Age' in df.columns:
        df['Age'] = df.groupby("Pclass")["Age"].transform(lambda x: x.fillna(x.median()))

    # 6. Drop Redundant Features
    drop_cols = ["PassengerId", "Name", "Ticket", "Cabin", "SibSp", "Parch"]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    return df