from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .utils import ensure_dir, save_json


def compute_metrics(y_true, y_pred, y_prob=None):
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "classification_report": classification_report(y_true, y_pred, zero_division=0, output_dict=True),
    }
    if y_prob is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            metrics["roc_auc"] = None
    else:
        metrics["roc_auc"] = None
    return metrics


def save_confusion_matrix(y_true, y_pred, model_name: str, output_dir: str = "models/reports"):
    output_path = ensure_dir(output_dir)
    fig, ax = plt.subplots(figsize=(6, 5))
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Legitimate", "Phishing"])
    disp.plot(ax=ax, colorbar=False)
    plt.title(f"Confusion Matrix - {model_name}")
    save_path = output_path / f"{model_name}_confusion_matrix.png"
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    return save_path


def save_metrics(metrics: dict, model_name: str, output_dir: str = "models/reports"):
    output_path = ensure_dir(output_dir)
    save_json(metrics, output_path / f"{model_name}_metrics.json")
