import re
import pandas as pd
from pathlib import Path
from typing import Tuple


def clean_text(text: str) -> str:
    """
    Basic text cleaning:
    - lowercasing
    - remove non-alphanumeric characters
    - collapse multiple spaces
    """
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_dataset(path: Path) -> pd.DataFrame:
    """
    Load dataset from CSV and normalize columns.

    Expected:
        - a 'text' column for reviews
        - a 'label' column that is either:
          * 0/1 or
          * 'negative'/'positive'
    """
    df = pd.read_csv(path)

    if "text" not in df.columns:
        raise ValueError("CSV must contain a 'text' column.")

    if "label" not in df.columns:
        raise ValueError("CSV must contain a 'label' column.")

    # Normalize label
    if df["label"].dtype == "object":
        label_map = {"negative": 0, "positive": 1}
        df["label"] = df["label"].map(label_map)

    if set(df["label"].unique()) - {0, 1}:
        raise ValueError("Label column must be 0/1 or 'negative'/'positive'.")

    df["clean_text"] = df["text"].apply(clean_text)
    return df


def train_test_split_dataframe(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    from sklearn.model_selection import train_test_split

    X = df["clean_text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test
