"""Standalone LSTM inference for email spam detection.

Usage::

    python lstm_predict.py

Loads the saved Keras model and tokeniser on first prediction
(lazy loading) and classifies the given email text.
"""

from __future__ import annotations

import logging
from functools import lru_cache

import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from config import LSTM_MODEL_PATH, MAX_SEQUENCE_LENGTH, TOKENIZER_PATH
from utils import clean_text

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load_artifacts():
    """Lazy-load the LSTM model and tokeniser (once, on first call)."""
    logger.info("Loading LSTM model from %s", LSTM_MODEL_PATH)
    model = load_model(LSTM_MODEL_PATH)
    tokenizer = joblib.load(TOKENIZER_PATH)
    return model, tokenizer


def predict(text: str) -> float:
    """Return a spam probability for the given email text.

    Parameters
    ----------
    text : str
        Raw email text.

    Returns
    -------
    float
        Probability in **[0.0, 1.0]** where values closer to 1.0
        indicate spam.
    """
    model, tokenizer = _load_artifacts()
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
    return float(model.predict(padded, verbose=0)[0][0])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    user_input = input("\nEnter email text: ")
    score = predict(user_input)
    print(f"\nPrediction score: {score:.4f}")
    print(f"Meaning: {'SPAM' if score > 0.5 else 'HAM'}")