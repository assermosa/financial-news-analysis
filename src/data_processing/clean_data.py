import pandas as pd

def clean_data(df):
    """
    Cleans the dataset by handling missing values and removing duplicates.
    """
    # Check for missing values
    if df.isnull().sum().any():
        df = df.dropna()  # or use df.fillna(method="ffill")

    # Check for duplicates
    if df.duplicated().sum() > 0:
        df = df.drop_duplicates()

    return df