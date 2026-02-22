"""
Data loading utilities.

Goal: keep I/O concerns here and return a pandas DataFrame that downstream modules can use.
"""

import pandas as pd

from src.config import CONFIG


def load_flight_data(path: str | None = None) -> pd.DataFrame:
    """
    Load the flight price dataset from CSV.

    Args:
        path: Optional CSV path. If None, uses CONFIG.data_path.

    Returns:
        A pandas DataFrame containing the raw dataset.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
    """
    csv_path = path or str(CONFIG.data_path)
    df = pd.read_csv(csv_path)

    # drop accidental index columns if present.
    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed")].copy()

    # normalize the column names.
    df.columns = (
        df.columns.str.replace(" ", "_")
        .str.replace("(", "")
        .str.replace(")", "")
        .str.replace("&", "and")
        .str.lower()
    )

    return df
