
from __future__ import annotations

from collections import Counter, defaultdict
from flask import Flask, render_template, request

from src.predict import Predictor
from src.db import save_prediction, get_all_predictions

app = Flask(__name__)

AVAILABLE_MODELS = ["svm", "naive_bayes", "random_forest", "lstm", "bert"]


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    clean_text = None
    error = None
    selected_model = "svm"
    email_text = ""
    history = []
    signals = []
    final_decision = None
    model_summary = {}

    if request.method == "POST":
        email_text = request.form.get("email_text", "").strip()
        selected_model = request.form.get("model", "svm")

        if selected_model not in AVAILABLE_MODELS:
            selected_model = "svm"

        if not email_text:
            error = "Please enter email text."
        else:
            try:
                predictor = Predictor(model_type=selected_model)
                result = predictor.predict(email_text)

                prediction = result.get("prediction")
                confidence = result.get("confidence")
                clean_text = result.get("clean_text")
                final_decision = prediction

                keywords = [
                    "urgent", "verify", "login", "click", "account",
                    "password", "suspended", "bank", "security", "confirm"
                ]
                text_lower = email_text.lower()
                signals = [k for k in keywords if k in text_lower]

                save_prediction(
                    email_text=email_text,
                    model_used=selected_model,
                    prediction=prediction if prediction is not None else "Unknown",
                    confidence=str(confidence) if confidence is not None else "None",
                    clean_text=clean_text if clean_text is not None else "",
                )

            except Exception as exc:
                error = f"Prediction failed: {exc}"

    try:
        history = get_all_predictions()
    except Exception as exc:
        if error is None:
            error = f"Database history load failed: {exc}"
        history = []

    # Build model summary from history
    temp = defaultdict(lambda: {"total": 0, "phishing": 0, "legitimate": 0})
    for row in history:
        model = row["model_used"]
        pred = str(row["prediction"]).lower()

        temp[model]["total"] += 1
        if pred == "phishing":
            temp[model]["phishing"] += 1
        elif pred == "legitimate":
            temp[model]["legitimate"] += 1

    model_summary = dict(temp)

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        clean_text=clean_text,
        error=error,
        models=AVAILABLE_MODELS,
        selected_model=selected_model,
        email_text=email_text,
        history=history,
        signals=signals,
        final_decision=final_decision,
        model_summary=model_summary,
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True, use_reloader=False)