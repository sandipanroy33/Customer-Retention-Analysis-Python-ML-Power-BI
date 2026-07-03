"""
feature_engineering.py
-----------------------
Feature engineering and train/validation/test split for the customer
churn dataset:
- Buckets 'tenure' into a categorical 'tenure_group' feature
- Encodes the target variable (Churn_New -> 0/1)
- Splits the data into train (70%), validation (15%) and test (15%) sets
- One-hot encodes categorical predictors

Usage:
    python scripts/feature_engineering.py
Reads:  data/processed/cleaned_customer_churn.csv
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split

PROCESSED_PATH = os.path.join("data", "processed", "cleaned_customer_churn.csv")


def add_tenure_group(df: pd.DataFrame) -> pd.DataFrame:
    """Bucket tenure (in months) into readable groups."""
    df["tenure_group"] = pd.cut(
        df["tenure"],
        bins=[0, 12, 36, 60, df["tenure"].max()],
        labels=["New (0-12)", "Medium (13-36)", "Long (37-60)", "Very Long (60+)"],
        include_lowest=True,
    )
    return df


def build_target_and_features(df: pd.DataFrame):
    """Split the dataframe into feature matrix X and target vector y."""
    y = df["Churn_New"].map({"Yes": 1, "No": 0})
    X = df.drop(["Churn_New", "tenure"], axis=1)
    return X, y


def train_val_test_split(X, y, random_state: int = 1):
    """Split into 70% train / 15% validation / 15% test, stratified on y."""
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=random_state
    )
    X_test, X_val, y_test, y_val = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=random_state
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def one_hot_encode(X_train, X_val, X_test):
    """One-hot encode categorical columns (drop_first to avoid collinearity)."""
    X_train = pd.get_dummies(X_train, drop_first=True)
    X_val = pd.get_dummies(X_val, drop_first=True)
    X_test = pd.get_dummies(X_test, drop_first=True)

    # Align columns across splits in case a category is missing in one split
    X_train, X_val = X_train.align(X_val, join="left", axis=1, fill_value=0)
    X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)

    return X_train, X_val, X_test


def main():
    df = pd.read_csv(PROCESSED_PATH)
    print(f"Loaded cleaned data: {df.shape[0]} rows, {df.shape[1]} columns")

    df = add_tenure_group(df)
    X, y = build_target_and_features(df)

    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y)
    print(f"Train: {X_train.shape} | Validation: {X_val.shape} | Test: {X_test.shape}")

    X_train, X_val, X_test = one_hot_encode(X_train, X_val, X_test)
    print(f"After encoding -> Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")

    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    main()
