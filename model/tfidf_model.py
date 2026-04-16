"""Train a TF-IDF + Logistic Regression model for email spam detection.

Usage::

    python tfidf_model.py

Artifacts produced
------------------
* ``spam_model.pkl``         – trained ``LogisticRegression``
* ``vectorizer.pkl``         – fitted ``TfidfVectorizer``
* ``tfidf_metrics.json``     – evaluation metrics snapshot

The script uses stratified splitting, 5-fold cross-validation on the
training set, ``class_weight='balanced'``, and exports a full
classification report, confusion matrix, and top feature importances.
"""

from __future__ import annotations

import json
import logging
import random

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import cross_val_score, train_test_split

from config import (
    BASE_DIR,
    DATASET_PATH,
    LOGREG_C,
    LOGREG_MAX_ITER,
    RANDOM_STATE,
    TEST_SIZE,
    TFIDF_MAX_FEATURES,
    TFIDF_MIN_DF,
    TFIDF_MODEL_PATH,
    TFIDF_NGRAM_RANGE,
    VECTORIZER_PATH,
)
from utils import clean_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

METRICS_PATH = BASE_DIR / "tfidf_metrics.json"


# ── Reproducibility ─────────────────────────────────────────────────────

def set_seeds(seed: int = RANDOM_STATE) -> None:
    """Set random seeds for Python and NumPy."""
    random.seed(seed)
    np.random.seed(seed)
    logger.info("Random seeds set to %d", seed)


# ── Data helpers ─────────────────────────────────────────────────────────

def load_dataset() -> pd.DataFrame:
    """Load and clean the Enron spam dataset.

    Returns
    -------
    pd.DataFrame
        Columns: ``text`` (cleaned), ``label`` (0 = ham, 1 = spam).

    Raises
    ------
    FileNotFoundError
        If the dataset CSV is missing.
    """
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")

    logger.info("Loading dataset from %s", DATASET_PATH)
    df = pd.read_csv(DATASET_PATH)

    df["text"] = df["Subject"].fillna("") + " " + df["Message"].fillna("")
    df["label"] = df["Spam/Ham"].str.lower().map({"ham": 0, "spam": 1})
    df = df[["text", "label"]].dropna()
    df["text"] = df["text"].astype(str).apply(clean_text)

    logger.info(
        "Dataset: %d samples  (ham=%d, spam=%d)",
        len(df),
        (df["label"] == 0).sum(),
        (df["label"] == 1).sum(),
    )
    return df


def augment_spam_samples(df: pd.DataFrame) -> pd.DataFrame:
    """Oversample spam with diverse synthetic templates.

    Repeats are sized to roughly balance ham and spam counts.
    ``class_weight='balanced'`` in the model handles residual imbalance.

    Note
    ----
    For production systems, consider SMOTE on TF-IDF features,
    synonym replacement, or back-translation.
    """
    templates = [
        "free money now",
        "win cash instantly",
        "claim your prize now",
        "someone told me about free money hacks",
        "i heard you can earn money fast online",
        "have you seen this trick to make money",
        "my friend showed me a way to get rich quick",
        "you should try this easy money method",
        "people are making money easily from this",
        "i found a way to earn money without effort",
        "click here to learn how to make money",
        "this is not a scam earn money fast",
        "earn money from home no experience needed",
        "you might want to check this offer",
        "just sharing this opportunity with you",
        "thought you might be interested in this offer",
    ]

    ham_count = int((df["label"] == 0).sum())
    spam_count = int((df["label"] == 1).sum())
    deficit = max(ham_count - spam_count, 0)
    repeats = max(deficit // len(templates), 1)

    logger.info(
        "Augmenting: %d templates × %d repeats → %d synthetic samples",
        len(templates),
        repeats,
        len(templates) * repeats,
    )

    extra = pd.DataFrame(
        {"text": templates * repeats, "label": [1] * (len(templates) * repeats)}
    )
    return pd.concat([df, extra], ignore_index=True)


# ── Feature importance ───────────────────────────────────────────────────

def top_features(
    model: LogisticRegression,
    vectorizer: TfidfVectorizer,
    n: int = 20,
) -> dict[str, list[dict[str, float]]]:
    """Extract the top-N most influential features for each class.

    Returns
    -------
    dict
        ``"ham"`` and ``"spam"`` keys mapping to lists of
        ``{"feature": str, "weight": float}`` dicts.
    """
    feature_names = np.array(vectorizer.get_feature_names_out())
    coefficients = model.coef_[0]

    top_spam_idx = np.argsort(coefficients)[-n:][::-1]
    top_ham_idx = np.argsort(coefficients)[:n]

    return {
        "spam": [
            {"feature": feature_names[i], "weight": round(float(coefficients[i]), 4)}
            for i in top_spam_idx
        ],
        "ham": [
            {"feature": feature_names[i], "weight": round(float(coefficients[i]), 4)}
            for i in top_ham_idx
        ],
    }


# ── Training pipeline ───────────────────────────────────────────────────

def main() -> None:
    """End-to-end: load → augment → split → train → evaluate → save."""

    set_seeds()

    df = load_dataset()
    df = augment_spam_samples(df)

    # Stratified 80 / 20 split
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["label"],
    )
    logger.info("Splits  →  train: %d  |  test: %d", len(X_train), len(X_test))

    # Vectorise — fit on training data only
    vectorizer = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=TFIDF_NGRAM_RANGE,
        min_df=TFIDF_MIN_DF,
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Train with class weighting
    model = LogisticRegression(
        max_iter=LOGREG_MAX_ITER,
        class_weight="balanced",
        C=LOGREG_C,
    )
    model.fit(X_train_vec, y_train)

    # 5-fold cross-validation
    cv_scores = cross_val_score(
        model, X_train_vec, y_train, cv=5, scoring="f1"
    )
    logger.info(
        "5-fold CV F1: %.4f ± %.4f", cv_scores.mean(), cv_scores.std()
    )

    # Evaluate on held-out test set
    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred).tolist()
    report = classification_report(y_test, y_pred, target_names=["HAM", "SPAM"])

    print("\n" + "=" * 55)
    print("  TF-IDF MODEL — EVALUATION REPORT")
    print("=" * 55)
    print(f"  Accuracy:       {acc:.4f}")
    print(f"  Precision:      {precision:.4f}")
    print(f"  Recall:         {recall:.4f}")
    print(f"  F1 Score:       {f1:.4f}")
    print(f"  5-Fold CV F1:   {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print()
    print(report)
    print("  Confusion Matrix:")
    print(f"    {cm[0]}")
    print(f"    {cm[1]}")
    print("=" * 55)

    # Feature importance
    features = top_features(model, vectorizer)
    print("\n  Top 10 SPAM indicators:")
    for f in features["spam"][:10]:
        print(f"    {f['weight']:+.4f}  {f['feature']}")
    print("\n  Top 10 HAM indicators:")
    for f in features["ham"][:10]:
        print(f"    {f['weight']:+.4f}  {f['feature']}")
    print()

    # Save model artifacts
    joblib.dump(model, TFIDF_MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    logger.info("Saved model      → %s", TFIDF_MODEL_PATH)
    logger.info("Saved vectorizer → %s", VECTORIZER_PATH)

    # Export metrics to JSON for CI / tracking
    metrics = {
        "model": "tfidf_logreg",
        "test_accuracy": round(acc, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "cv_f1_mean": round(float(cv_scores.mean()), 4),
        "cv_f1_std": round(float(cv_scores.std()), 4),
        "confusion_matrix": cm,
        "top_features": features,
        "hyperparameters": {
            "max_features": TFIDF_MAX_FEATURES,
            "ngram_range": list(TFIDF_NGRAM_RANGE),
            "min_df": TFIDF_MIN_DF,
            "C": LOGREG_C,
            "max_iter": LOGREG_MAX_ITER,
            "random_state": RANDOM_STATE,
        },
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    logger.info("Saved metrics    → %s", METRICS_PATH)


if __name__ == "__main__":
    main()