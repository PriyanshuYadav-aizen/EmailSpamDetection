"""Standalone TF-IDF inference for email spam detection.

Usage::

    python tfidf_predict.py

Loads the saved Logistic Regression model and TF-IDF vectoriser on
first prediction (lazy loading) and classifies the given email text.
"""

from __future__ import annotations

import logging
from functools import lru_cache

import joblib

from config import TFIDF_MODEL_PATH, VECTORIZER_PATH
from utils import clean_text

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load_artifacts():
    """Lazy-load the TF-IDF model and vectoriser (once, on first call)."""
    logger.info("Loading TF-IDF model from %s", TFIDF_MODEL_PATH)
    model = joblib.load(TFIDF_MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


def predict(text: str) -> str:
    """Classify email text as SPAM or HAM.

    Parameters
    ----------
    text : str
        Raw email text.

    Returns
    -------
    str
        ``"SPAM"`` or ``"HAM"``.
    """
    model, vectorizer = _load_artifacts()
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    return "SPAM" if pred == 1 else "HAM"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    user_input = input("\nEnter email text: ")
    result = predict(user_input)
    print(f"\nPrediction: {result}")