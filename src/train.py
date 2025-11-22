from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline

from src.data_utils import load_dataset, train_test_split_dataframe


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "reviews.csv"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "baseline_logreg.joblib"


def build_model() -> Pipeline:
    """
    Build a text classification pipeline: TF-IDF + Logistic Regression.
    """
    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=20000,
                    ngram_range=(1, 2),  # unigrams + bigrams
                    stop_words="english",
                ),
            ),
            ("logreg", LogisticRegression(max_iter=1000)),
        ]
    )
    return pipeline


def train():
    print(f"Loading dataset from: {DATA_PATH}")
    df = load_dataset(DATA_PATH)
    print(f"Loaded {len(df)} rows.")

    X_train, X_test, y_train, y_test = train_test_split_dataframe(df)

    print("Building model...")
    model = build_model()

    print("Training...")
    model.fit(X_train, y_train)

    print("Evaluating on test set...")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print("\nClassification report:\n")
    print(classification_report(y_test, y_pred))

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\nModel saved to: {MODEL_PATH}")


if __name__ == "__main__":
    train()
