from __future__ import annotations

import argparse

from src.train_bert import train_bert_pipeline
from src.train_lstm import train_lstm_pipeline
from src.train_ml import train_and_evaluate_classical


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--skip-lstm", action="store_true")
    parser.add_argument("--skip-bert", action="store_true")
    args = parser.parse_args()

    print("Training classical ML models...")
    classical_results = train_and_evaluate_classical(args.csv, args.text_column, args.label_column)
    print(classical_results)

    if not args.skip_lstm:
        print("Training LSTM model...")
        lstm_results = train_lstm_pipeline(args.csv, args.text_column, args.label_column)
        print(lstm_results)

    if not args.skip_bert:
        print("Training BERT model...")
        bert_results = train_bert_pipeline(args.csv, args.text_column, args.label_column)
        print(bert_results)
