from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from .data_loader import load_dataset, save_splits, split_dataset
from .evaluate import compute_metrics, save_confusion_matrix, save_metrics
from .features import build_bow, build_tfidf
from .utils import save_joblib, ensure_dir


def train_and_evaluate_classical(csv_path: str, text_column: str, label_column: str):
    print("Loading dataset...")
    df = load_dataset(csv_path, text_column, label_column)
    print(f"Dataset loaded successfully. Total rows: {len(df)}")

    print("Splitting dataset...")
    train_df, val_df, test_df = split_dataset(df)
    save_splits(train_df, val_df, test_df)
    print(f"Train: {len(train_df)}, Validation: {len(val_df)}, Test: {len(test_df)}")

    results = {}

    # Ensure directories exist
    model_dir = ensure_dir("saved_models")
    reports_dir = ensure_dir("models/reports")

    # ---------------------------
    # 1. Naive Bayes on BoW
    # ---------------------------
    print("Training Naive Bayes...")
    bow_vectorizer, X_train_bow, X_val_bow, X_test_bow = build_bow(
        train_df["clean_text"], val_df["clean_text"], test_df["clean_text"]
    )

    nb_model = MultinomialNB()
    nb_model.fit(X_train_bow, train_df["label"])
    nb_pred = nb_model.predict(X_test_bow)
    nb_prob = nb_model.predict_proba(X_test_bow)[:, 1]

    nb_metrics = compute_metrics(test_df["label"], nb_pred, nb_prob)
    save_metrics(nb_metrics, "naive_bayes")
    save_confusion_matrix(test_df["label"], nb_pred, "naive_bayes")

    save_joblib(nb_model, model_dir / "naive_bayes_model.joblib")
    save_joblib(bow_vectorizer, model_dir / "bow_vectorizer.joblib")

    results["naive_bayes"] = nb_metrics
    print("Naive Bayes trained and saved successfully.")

    # ---------------------------
    # 2. SVM on TF-IDF
    # ---------------------------
    print("Training SVM...")
    tfidf_vectorizer, X_train_tfidf, X_val_tfidf, X_test_tfidf = build_tfidf(
        train_df["clean_text"], val_df["clean_text"], test_df["clean_text"]
    )

    svm_model = LinearSVC()
    svm_model.fit(X_train_tfidf, train_df["label"])
    svm_pred = svm_model.predict(X_test_tfidf)

    svm_metrics = compute_metrics(test_df["label"], svm_pred)
    save_metrics(svm_metrics, "svm")
    save_confusion_matrix(test_df["label"], svm_pred, "svm")

    save_joblib(svm_model, model_dir / "svm_model.joblib")
    save_joblib(tfidf_vectorizer, model_dir / "tfidf_vectorizer.joblib")

    results["svm"] = svm_metrics
    print("SVM trained and saved successfully.")

    # ---------------------------
    # 3. Random Forest on TF-IDF
    # ---------------------------
    print("Training Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf_model.fit(X_train_tfidf, train_df["label"])
    rf_pred = rf_model.predict(X_test_tfidf)
    rf_prob = rf_model.predict_proba(X_test_tfidf)[:, 1]

    rf_metrics = compute_metrics(test_df["label"], rf_pred, rf_prob)
    save_metrics(rf_metrics, "random_forest")
    save_confusion_matrix(test_df["label"], rf_pred, "random_forest")

    save_joblib(rf_model, model_dir / "random_forest_model.joblib")

    results["random_forest"] = rf_metrics
    print("Random Forest trained and saved successfully.")

    # ---------------------------
    # Save summary report
    # ---------------------------
    summary_path = reports_dir / "classical_summary.csv"
    pd.DataFrame(results).T.to_csv(summary_path)
    print(f"Summary report saved at: {summary_path}")

    print("All classical models trained and saved successfully.")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--label-column", default="label")
    args = parser.parse_args()

    try:
        train_and_evaluate_classical(args.csv, args.text_column, args.label_column)
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise