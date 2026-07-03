"""
data_cleaning.py
-----------------
Cleans the raw customer churn dataset:
- Fixes invalid/negative tenure values
- Strips currency symbols from MonthlyCharges / TotalCharges and converts to numeric
- Imputes missing numerical values with the median
- Standardizes inconsistent text formatting in categorical columns
  (PaymentMethod, Churn)
- Removes duplicate rows

Usage:
    python scripts/data_cleaning.py
Reads:  data/raw/customer_churn.csv
Writes: data/processed/cleaned_customer_churn.csv
"""

import os
import numpy as np
import pandas as pd

RAW_PATH = os.path.join("data", "raw", "customer_churn.csv")
PROCESSED_PATH = os.path.join("data", "processed", "cleaned_customer_churn.csv")


def load_data(path: str = RAW_PATH) -> pd.DataFrame:
    """Load the raw churn dataset."""
    return pd.read_csv(path)


def clean_tenure(df: pd.DataFrame) -> pd.DataFrame:
    """Fix negative/missing tenure values."""
    df.loc[df["tenure"] < 0, "tenure"] = 0
    df["tenure"] = df["tenure"].fillna(0).astype(int)
    return df


def clean_charges(df: pd.DataFrame) -> pd.DataFrame:
    """Strip currency symbols and convert MonthlyCharges/TotalCharges to numeric."""
    cols_to_clean = ["MonthlyCharges", "TotalCharges"]

    for col in cols_to_clean:
        df[col] = df[col].replace(["nan", "$nan", "£nan"], np.nan)
        df[col] = (
            df[col]
            .astype(str)
            .str.replace("$", "", regex=False)
            .str.replace("£", "", regex=False)
            .str.replace(",", "", regex=False)
        )

    df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    df["MonthlyCharges"] = df["MonthlyCharges"].fillna(df["MonthlyCharges"].median())

    df.loc[df["TotalCharges"] < 0, "TotalCharges"] = 0
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    return df


def clean_payment_method(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize the PaymentMethod column's text formatting."""
    df["PaymentMethod_New"] = df["PaymentMethod"].copy()
    df["PaymentMethod_New"] = df["PaymentMethod_New"].str.strip()
    df["PaymentMethod_New"] = df["PaymentMethod_New"].str.lower()
    df["PaymentMethod_New"] = df["PaymentMethod_New"].str.replace(r"\s+", " ", regex=True)
    df["PaymentMethod_New"] = df["PaymentMethod_New"].str.replace(r"\s*\(\s*", " (", regex=True)
    df["PaymentMethod_New"] = df["PaymentMethod_New"].str.replace(r"\s*\)", ")", regex=True)
    df["PaymentMethod_New"] = df["PaymentMethod_New"].str.title()
    df.drop("PaymentMethod", axis=1, inplace=True)
    return df


def clean_churn(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize the Churn (target) column's text formatting."""
    df["Churn_New"] = df["Churn"].copy()
    df["Churn_New"] = df["Churn_New"].str.strip()
    df["Churn_New"] = df["Churn_New"].str.lower()
    df["Churn_New"] = df["Churn_New"].str.replace(r"\s+", " ", regex=True)
    df["Churn_New"] = df["Churn_New"].str.capitalize()
    df.drop("Churn", axis=1, inplace=True)
    return df


def clean_senior_citizen(df: pd.DataFrame) -> pd.DataFrame:
    """Map SeniorCitizen 0/1 flag to Yes/No category."""
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"}).astype("category")
    return df


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows, keeping the first occurrence."""
    return df.drop_duplicates(keep="first")


def clean_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full cleaning pipeline on the raw dataframe."""
    df = clean_tenure(df)
    df = clean_charges(df)
    df = clean_payment_method(df)
    df = clean_churn(df)
    df = clean_senior_citizen(df)
    df = drop_duplicates(df)
    return df


def main():
    df = load_data()
    print(f"Loaded raw data: {df.shape[0]} rows, {df.shape[1]} columns")

    df = clean_pipeline(df)
    print(f"After cleaning: {df.shape[0]} rows, {df.shape[1]} columns")

    os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)
    print(f"Cleaned data saved to: {PROCESSED_PATH}")


if __name__ == "__main__":
    main()
