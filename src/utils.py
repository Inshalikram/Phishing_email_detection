from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_joblib(obj: Any, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    joblib.dump(obj, path)


def load_joblib(path: str | Path) -> Any:
    return joblib.load(path)


def safe_label_map(value: Any) -> int | None:
    """
    Convert different label formats into binary values:
    phishing -> 1
    legitimate/safe -> 0

    Returns None for invalid/unrecognized values so they can be dropped later.
    """
    if value is None:
        return None

    v = str(value).strip().lower()

    phishing_values = {
        "1",
        "true",
        "yes",
        "phishing",
        "phishing email",
        "spam",
        "malicious",
        "fraud",
        "scam",
    }

    safe_values = {
        "0",
        "false",
        "no",
        "legitimate",
        "safe",
        "safe email",
        "ham",
        "benign",
        "normal",
        "not phishing",
    }

    if v in phishing_values:
        return 1

    if v in safe_values:
        return 0

    # handle numeric labels like 1.0 / 0.0
    try:
        num = float(v)
        if num == 1.0:
            return 1
        if num == 0.0:
            return 0
    except ValueError:
        pass

    # invalid row / shifted text / bad label
    return None