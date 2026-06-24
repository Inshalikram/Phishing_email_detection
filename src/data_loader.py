from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import CONFIG
from .preprocess import preprocess_corpus
from .utils import ensure_dir, safe_label_map


def load_dataset(csv_path: str, text_column: str, label_column: str) -> pd.DataFrame:
    # low_memory=False avoids mixed dtype parsing issues
    df = pd.read_csv(csv_path, low_memory=False)

    # Remove unwanted Excel index columns like "Unnamed: 0"
    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed", case=False)]

    if text_column not in df.columns:
        raise ValueError(
            f"Text column '{text_column}' not found. Available columns: {list(df.columns)}"
        )

    if label_column not in df.columns:
        raise ValueError(
            f"Label column '{label_column}' not found. Available columns: {list(df.columns)}"
        )

    # Keep only required columns
    df = df[[text_column, label_column]].copy()
    df.columns = ["text", "label"]

    # Clean text column
    df["text"] = df["text"].astype(str).str.strip()

    # Convert labels safely
    df["label"] = df["label"].apply(safe_label_map)

    # Drop invalid rows
    df = df.dropna(subset=["text", "label"]).copy()
    df = df[df["text"] != ""].copy()
    df = df[df["label"].isin([0, 1])].copy()

    if df.empty:
        raise ValueError("No valid rows left after cleaning dataset.")

    # Preprocess text
    df["clean_text"] = preprocess_corpus(df["text"].tolist())

    return df.reset_index(drop=True)


def split_dataset(df: pd.DataFrame):
    train_df, temp_df = train_test_split(
        df,
        test_size=CONFIG.test_size + CONFIG.val_size,
        stratify=df["label"],
        random_state=CONFIG.random_state,
    )

    relative_test_size = CONFIG.test_size / (CONFIG.test_size + CONFIG.val_size)

    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_size,
        stratify=temp_df["label"],
        random_state=CONFIG.random_state,
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def save_splits(train_df, val_df, test_df, output_dir: str = "data/processed"):
    output_path = ensure_dir(output_dir)
    train_df.to_csv(output_path / "train.csv", index=False)
    val_df.to_csv(output_path / "val.csv", index=False)
    test_df.to_csv(output_path / "test.csv", index=False)