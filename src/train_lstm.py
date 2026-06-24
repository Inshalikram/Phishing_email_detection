from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

from .data_loader import load_dataset, split_dataset, save_splits
from .evaluate import compute_metrics, save_metrics, save_confusion_matrix
from .utils import ensure_dir, save_joblib


MAX_WORDS = 10000
MAX_LEN = 200
EMBED_DIM = 128


def train_lstm(csv_path: str, text_column: str, label_column: str):
    print("Loading dataset...")
    df = load_dataset(csv_path, text_column, label_column)

    train_df, val_df, test_df = split_dataset(df)
    save_splits(train_df, val_df, test_df)

    print("Preparing tokenizer...")
    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(train_df["clean_text"])

    X_train = tokenizer.texts_to_sequences(train_df["clean_text"])
    X_val = tokenizer.texts_to_sequences(val_df["clean_text"])
    X_test = tokenizer.texts_to_sequences(test_df["clean_text"])

    X_train = pad_sequences(X_train, maxlen=MAX_LEN, padding="post", truncating="post")
    X_val = pad_sequences(X_val, maxlen=MAX_LEN, padding="post", truncating="post")
    X_test = pad_sequences(X_test, maxlen=MAX_LEN, padding="post", truncating="post")

    y_train = np.array(train_df["label"])
    y_val = np.array(val_df["label"])
    y_test = np.array(test_df["label"])

    print("Building LSTM model...")
    model = Sequential([
        Embedding(input_dim=MAX_WORDS, output_dim=EMBED_DIM, input_length=MAX_LEN),
        LSTM(64, return_sequences=False),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=2,
        restore_best_weights=True
    )

    print("Training LSTM...")
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=3,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1
    )

    print("Evaluating LSTM...")
    y_prob = model.predict(X_test).ravel()
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = compute_metrics(y_test, y_pred, y_prob)
    save_metrics(metrics, "lstm")
    save_confusion_matrix(y_test, y_pred, "lstm")

    model_dir = ensure_dir("saved_models")
    model.save(model_dir / "lstm_model.h5")
    save_joblib(tokenizer, model_dir / "lstm_tokenizer.joblib")

    print("LSTM model saved successfully.")
    print("Metrics:", metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--label-column", default="label")
    args = parser.parse_args()

    train_lstm(args.csv, args.text_column, args.label_column)