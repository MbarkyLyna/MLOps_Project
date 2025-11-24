# model_pipeline.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib


def prepare_data(filename):
    """
    Load and preprocess Titanic dataset.
    Returns X_train, X_val, y_train, y_val
    """
    df = pd.read_csv(filename)

    # Handle missing values
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

    # Encode categorical variables
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)

    # Drop columns not used by model
    df = df.drop(["Name", "Ticket", "Cabin"], axis=1)

    # Features and target
    X = df.drop(["Survived", "PassengerId"], axis=1)
    y = df["Survived"]

    # Split
    return train_test_split(X, y, test_size=0.2, random_state=42)


def train_model(X_train, y_train, n_estimators=200):
    """
    Train a Random Forest model and return it.
    """
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    return rf


def evaluate_model(model, X_val, y_val):
    """
    Evaluate model performance and print accuracy.
    """
    preds = model.predict(X_val)
    acc = accuracy_score(y_val, preds)
    print(f"Validation Accuracy: {acc:.4f}")
    return acc


def save_model(model, filename):
    """Save model using joblib"""
    joblib.dump(model, filename)


def load_model(filename):
    """Load model using joblib"""
    return joblib.load(filename)
