from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from src.preprocess import preprocess_text


class Predictor:
    def __init__(self, model_type: str = "svm"):
        self.model_type = model_type
        self.model = None
        self.vectorizer = None
        self.tokenizer = None
        self._load()

    def _load(self):
        model_dir = Path("saved_models")

        if self.model_type == "naive_bayes":
            self.model = joblib.load(model_dir / "naive_bayes_model.joblib")
            self.vectorizer = joblib.load(model_dir / "bow_vectorizer.joblib")

        elif self.model_type == "svm":
            self.model = joblib.load(model_dir / "svm_model.joblib")
            self.vectorizer = joblib.load(model_dir / "tfidf_vectorizer.joblib")

        elif self.model_type == "random_forest":
            self.model = joblib.load(model_dir / "random_forest_model.joblib")
            self.vectorizer = joblib.load(model_dir / "tfidf_vectorizer.joblib")

        elif self.model_type == "lstm":
            from tensorflow.keras.models import load_model
            self.model = load_model(model_dir / "lstm_model.h5")
            self.tokenizer = joblib.load(model_dir / "lstm_tokenizer.joblib")

        elif self.model_type == "bert":
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch

            bert_path = model_dir / "bert_model"

            self.bert_tokenizer = AutoTokenizer.from_pretrained(str(bert_path))
            self.bert_model = AutoModelForSequenceClassification.from_pretrained(str(bert_path))
            self.bert_model.eval()
            self.torch = torch

        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def predict(self, input_text: str):
        clean_text = preprocess_text(input_text)

        if self.model_type in ["naive_bayes", "svm", "random_forest"]:
            X = self.vectorizer.transform([clean_text])
            pred = self.model.predict(X)[0]

            confidence = None
            if hasattr(self.model, "predict_proba"):
                confidence = float(np.max(self.model.predict_proba(X)))
            elif hasattr(self.model, "decision_function"):
                score = self.model.decision_function(X)
                confidence = float(1 / (1 + np.exp(-np.abs(score[0]))))

            label = "Phishing" if int(pred) == 1 else "Legitimate"

            return {
                "input_text": input_text,
                "clean_text": clean_text,
                "prediction": label,
                "label": int(pred),
                "confidence": confidence,
                "model_type": self.model_type,
            }

        elif self.model_type == "lstm":
            from tensorflow.keras.preprocessing.sequence import pad_sequences

            seq = self.tokenizer.texts_to_sequences([clean_text])
            X = pad_sequences(seq, maxlen=200, padding="post", truncating="post")

            prob = float(self.model.predict(X, verbose=0)[0][0])
            pred = 1 if prob >= 0.5 else 0
            label = "Phishing" if pred == 1 else "Legitimate"

            return {
                "input_text": input_text,
                "clean_text": clean_text,
                "prediction": label,
                "label": pred,
                "confidence": prob,
                "model_type": self.model_type,
            }

        elif self.model_type == "bert":
            inputs = self.bert_tokenizer(
                input_text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=128,
            )

            with self.torch.no_grad():
                outputs = self.bert_model(**inputs)
                probs = self.torch.softmax(outputs.logits, dim=1)
                pred = int(self.torch.argmax(probs, dim=1).item())
                confidence = float(self.torch.max(probs).item())

            label = "Phishing" if pred == 1 else "Legitimate"

            return {
                "input_text": input_text,
                "clean_text": clean_text,
                "prediction": label,
                "label": pred,
                "confidence": confidence,
                "model_type": self.model_type,
            }

        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")