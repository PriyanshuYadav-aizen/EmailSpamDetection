"""Centralised configuration for the spam-detection model service.

Values can be overridden via environment variables with the prefix
``SPAM_`` (e.g. ``SPAM_THRESHOLD=0.60``).  Defaults match the
hyperparameters used during training.

All file paths are resolved relative to **this file** so the project
works regardless of the caller's working directory.
"""

from __future__ import annotations

import os
from pathlib import Path

# ── Helpers ───────────────────────────────────────────────────────────────

def _env(name: str, default: str) -> str:
    """Read an environment variable with a ``SPAM_`` prefix."""
    return os.environ.get(f"SPAM_{name}", default)


def _env_float(name: str, default: float) -> float:
    return float(_env(name, str(default)))


def _env_int(name: str, default: int) -> int:
    return int(_env(name, str(default)))


# ── Paths ─────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent

LSTM_MODEL_PATH  = BASE_DIR / "lstm_model.keras"
TOKENIZER_PATH   = BASE_DIR / "tokenizer.pkl"
TFIDF_MODEL_PATH = BASE_DIR / "spam_model.pkl"
VECTORIZER_PATH  = BASE_DIR / "vectorizer.pkl"
DATASET_PATH     = BASE_DIR / "enron.csv"

# ── LSTM hyper-parameters ─────────────────────────────────────────────────

MAX_SEQUENCE_LENGTH = _env_int("MAX_SEQ_LEN", 100)
VOCAB_SIZE          = _env_int("VOCAB_SIZE", 10_000)
EMBEDDING_DIM       = _env_int("EMBEDDING_DIM", 128)
LSTM_UNITS          = _env_int("LSTM_UNITS", 64)
DROPOUT_RATE        = _env_float("DROPOUT_RATE", 0.4)
OOV_TOKEN           = "<OOV>"

# ── TF-IDF hyper-parameters ──────────────────────────────────────────────

TFIDF_MAX_FEATURES = _env_int("TFIDF_MAX_FEATURES", 50_000)
TFIDF_NGRAM_RANGE  = (1, 2)
TFIDF_MIN_DF       = _env_int("TFIDF_MIN_DF", 2)
LOGREG_C           = _env_float("LOGREG_C", 2.0)
LOGREG_MAX_ITER    = _env_int("LOGREG_MAX_ITER", 1_000)

# ── Ensemble weights ─────────────────────────────────────────────────────

RULE_WEIGHT    = _env_float("RULE_WEIGHT", 0.50)
TFIDF_WEIGHT   = _env_float("TFIDF_WEIGHT", 0.25)
LSTM_WEIGHT    = _env_float("LSTM_WEIGHT", 0.25)
SPAM_THRESHOLD = _env_float("THRESHOLD", 0.50)

# ── Training ─────────────────────────────────────────────────────────────

TEST_SIZE           = _env_float("TEST_SIZE", 0.2)
VALIDATION_SIZE     = _env_float("VAL_SIZE", 0.1)
RANDOM_STATE        = _env_int("RANDOM_STATE", 42)
BATCH_SIZE          = _env_int("BATCH_SIZE", 64)
EPOCHS              = _env_int("EPOCHS", 10)
EARLY_STOP_PATIENCE = _env_int("EARLY_STOP_PATIENCE", 3)

# ── Input constraints ────────────────────────────────────────────────────

MAX_INPUT_LENGTH = _env_int("MAX_INPUT_LENGTH", 50_000)  # characters

# ── Rule-engine keyword list ─────────────────────────────────────────────

SPAM_KEYWORDS: list[str] = [
    "free", "money", "win", "prize", "offer",
    "payment", "billing", "invoice", "transaction", "failed",
    "update payment", "billing update",
    "click", "urgent", "congratulations", "winner",
    "act now", "limited time", "no cost", "cash",
    "earn", "income", "lottery", "discount",
    "bank", "account", "verify", "suspend", "unusual activity",
    "security team", "confirm your details", "loss of access",
    "temporary suspension", "unauthorized login", "security alert",
    "validate", "action required", "notification", "official alert",
    "restricted access", "billing issue", "invoice", "payment failed",
    "package", "delivery", "shipping",
    "work from home", "job opportunity", "registration fee", "daily", "applicant",
    "$", "dollars", "opportunity", "exclusive", "limited", "🚨", "💼", "📦", "💰",
]
