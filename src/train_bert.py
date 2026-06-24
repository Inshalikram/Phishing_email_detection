from __future__ import annotations

import argparse
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

from .data_loader import load_dataset, split_dataset, save_splits
from .evaluate import save_metrics
from .utils import ensure_dir


MODEL_NAME = "bert-base-uncased"


def tokenize_function(examples, tokenizer):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=128,
    )


def compute_hf_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="binary",
        zero_division=0,
    )
    acc = accuracy_score(labels, predictions)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def train_bert(csv_path: str, text_column: str, label_column: str):
    print("Loading dataset...")
    df = load_dataset(csv_path, text_column, label_column)

    train_df, val_df, test_df = split_dataset(df)
    save_splits(train_df, val_df, test_df)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_dataset = Dataset.from_dict({
        "text": train_df["clean_text"].tolist(),
        "label": [int(x) for x in train_df["label"].tolist()],
    })
    val_dataset = Dataset.from_dict({
        "text": val_df["clean_text"].tolist(),
        "label": [int(x) for x in val_df["label"].tolist()],
    })
    test_dataset = Dataset.from_dict({
        "text": test_df["clean_text"].tolist(),
        "label": [int(x) for x in test_df["label"].tolist()],
    })

    print("Tokenizing datasets...")
    train_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer), batched=True
    )
    val_dataset = val_dataset.map(
        lambda x: tokenize_function(x, tokenizer), batched=True
    )
    test_dataset = test_dataset.map(
        lambda x: tokenize_function(x, tokenizer), batched=True
    )

    train_dataset = train_dataset.rename_column("label", "labels")
    val_dataset = val_dataset.rename_column("label", "labels")
    test_dataset = test_dataset.rename_column("label", "labels")

    train_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"]
    )
    val_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"]
    )
    test_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"]
    )

    print("Loading BERT model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        problem_type="single_label_classification",
    )

    output_dir = ensure_dir("saved_models/bert_model")

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=1,
        weight_decay=0.01,
        logging_steps=20,
        load_best_model_at_end=True,
        report_to="none",
        save_total_limit=1,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_hf_metrics,
    )

    print("Training BERT...")
    trainer.train()

    print("Evaluating BERT...")
    results = trainer.evaluate(test_dataset)

    metrics = {
        "accuracy": results.get("eval_accuracy"),
        "precision": results.get("eval_precision"),
        "recall": results.get("eval_recall"),
        "f1": results.get("eval_f1"),
    }
    save_metrics(metrics, "bert")

    print("Saving model and tokenizer...")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    print("BERT model saved successfully.")
    print("Metrics:", metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--label-column", default="label")
    args = parser.parse_args()

    train_bert(args.csv, args.text_column, args.label_column)