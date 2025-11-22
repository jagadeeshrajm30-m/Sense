from pathlib import Path
from typing import Dict

import joblib

from src.data_utils import clean_text

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "baseline_logreg.joblib"

_model = None


def load_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model file not found at {MODEL_PATH}. "
                "Run `python -m src.train` from the project root first."
            )
        _model = joblib.load(MODEL_PATH)
    return _model


def predict_sentiment(text: str) -> Dict:
    model = load_model()
    cleaned = clean_text(text)
    proba = model.predict_proba([cleaned])[0]
    pred = int(proba.argmax())
    confidence = float(proba.max())

    if confidence < 0.6:       # threshold, you can tune this
        label_name = "neutral"
    else:
        label_name = "positive" if pred == 1 else "negative"

    return {
        "label": pred,
        "label_name": label_name,
        "confidence": confidence,
    }

