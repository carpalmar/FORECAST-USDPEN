import pandas as pd
from typing import List


def load_data(
    path: str, date_col: str, value_col: str, rename_cols: List[str] = None
) -> pd.DataFrame:
    """
    Loads time series data from a CSV file.

    Args:
        path (str): The path to the CSV file.
        date_col (str): The name of the date column.
        value_col (str): The name of the value column to be used.
        rename_cols (List[str]): A list of two strings to rename date and value
                                 columns, e.g., ['ds', 'y'].

    Returns:
        pd.DataFrame: A DataFrame with a datetime index and the value column.
    """
    df = pd.read_csv(path)

    # Keep only the necessary columns
    df = df[[date_col, value_col]]

    # Convert date column to datetime and set as index
    df[date_col] = pd.to_datetime(df[date_col])
    df.set_index(date_col, inplace=True)

    # Rename columns if specified
    if rename_cols and len(rename_cols) == 2:
        df.index.name = rename_cols[0]
        df.rename(columns={value_col: rename_cols[1]}, inplace=True)
    else:
        df.index.name = "date"

    # Drop rows with missing values
    df = df.dropna()

    return df
