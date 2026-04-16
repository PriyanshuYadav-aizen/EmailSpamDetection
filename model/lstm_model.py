"""Train a Bidirectional LSTM for email spam detection.

Usage::

    python lstm_model.py

Artifacts produced
------------------
* ``lstm_model.keras``       – trained Keras model
* ``tokenizer.pkl``          – fitted Keras tokeniser
* ``lstm_metrics.json``      – evaluation metrics snapshot

The script sets global random seeds for reproducibility, applies text
cleaning, balanced augmentation, a stratified train / validation / test
split, class weighting, early stopping, and learning-rate reduction on
plateau before saving the final model and its evaluation metrics.
"""

from __future__ import annotations

import json
import logging
import os
import random

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (
    Bidirectional,
    Dense,
    Dropout,
    Embedding,
    LSTM,
    SpatialDropout1D,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from config import (
    BATCH_SIZE,
    DATASET_PATH,
    DROPOUT_RATE,
    EARLY_STOP_PATIENCE,
    EMBEDDING_DIM,
    EPOCHS,
    LSTM_MODEL_PATH,
    LSTM_UNITS,
    MAX_SEQUENCE_LENGTH,
    OOV_TOKEN,
    RANDOM_STATE,
    TEST_SIZE,
    TOKENIZER_PATH,
    VALIDATION_SIZE,
    VOCAB_SIZE,
    BASE_DIR,
)
from utils import clean_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

METRICS_PATH = BASE_DIR / "lstm_metrics.json"


# ── Reproducibility ─────────────────────────────────────────────────────

def set_seeds(seed: int = RANDOM_STATE) -> None:
    """Set random seeds for Python, NumPy, and TensorFlow."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass
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

    The number of repeats is calculated so the augmented spam count
    roughly matches the ham count, keeping the dataset balanced.
    ``class_weight`` during training handles any remaining imbalance.

    Note
    ----
    For production systems, consider more advanced augmentation such as
    synonym replacement, back-translation, or contextual word embeddings.
    """
    templates = [
        # direct spam
        "free money now",
        "win cash instantly",
        "claim your prize now",
        # conversational spam
        "my friend told me about a way to make money online",
        "someone showed me a trick to earn money easily",
        "you should try this method to get rich quickly",
        "i heard people are making money from this",
        "my grandma told me about free money hacks",
        "i found a way to earn money without doing much",
        "this might help you earn money fast",
        "have you seen this easy money trick",
        # mixed tone
        "this is not a scam you can earn money online",
        "just sharing a way to make money from home",
        "thought you might like this earning opportunity",
        # real-like
        "dont you want to make money easily",
        "you can earn money from home without effort",
        "people are earning money from this simple trick",
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


# ── Model architecture ──────────────────────────────────────────────────

def build_model() -> Sequential:
    """Construct a Bidirectional LSTM binary classifier.

    Architecture
    ------------
    Embedding → SpatialDropout1D(0.2) → BiLSTM(64, dropout=0.2,
    recurrent_dropout=0.2) → Dropout(0.4) → Dense(32, relu)
    → Dropout(0.3) → Dense(1, sigmoid)
    """
    model = Sequential(
        [
            Embedding(
                input_dim=VOCAB_SIZE,
                output_dim=EMBEDDING_DIM,
                input_length=MAX_SEQUENCE_LENGTH,
            ),
            SpatialDropout1D(0.2),
            Bidirectional(
                LSTM(
                    LSTM_UNITS,
                    return_sequences=False,
                    dropout=0.2,
                    recurrent_dropout=0.2,
                )
            ),
            Dropout(DROPOUT_RATE),
            Dense(32, activation="relu"),
            Dropout(0.3),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        loss="binary_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
    )
    return model


# ── Training pipeline ───────────────────────────────────────────────────

def main() -> None:
    """End-to-end: load → augment → split → train → evaluate → save."""

    set_seeds()

    # 1. Data
    df = load_dataset()
    df = augment_spam_samples(df)

    # 2. Train / Validation / Test split  (stratified)
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["label"],
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_train_full,
    )
    logger.info(
        "Splits  →  train: %d  |  val: %d  |  test: %d",
        len(X_train),
        len(X_val),
        len(X_test),
    )

    # 3. Tokenise (fit on training vocabulary only)
    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token=OOV_TOKEN)
    tokenizer.fit_on_texts(X_train)

    X_train_pad = pad_sequences(
        tokenizer.texts_to_sequences(X_train), maxlen=MAX_SEQUENCE_LENGTH
    )
    X_val_pad = pad_sequences(
        tokenizer.texts_to_sequences(X_val), maxlen=MAX_SEQUENCE_LENGTH
    )
    X_test_pad = pad_sequences(
        tokenizer.texts_to_sequences(X_test), maxlen=MAX_SEQUENCE_LENGTH
    )

    # 4. Class weights to compensate for residual imbalance
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weight = dict(zip(classes.tolist(), weights.tolist()))
    logger.info("Class weights: %s", class_weight)

    # 5. Build model
    model = build_model()
    model.summary(print_fn=logger.info)

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOP_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            verbose=1,
        ),
    ]

    # 6. Train
    history = model.fit(
        X_train_pad,
        y_train,
        validation_data=(X_val_pad, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight,
        callbacks=callbacks,
    )

    # 7. Evaluate on held-out test set
    loss, acc = model.evaluate(X_test_pad, y_test, verbose=0)
    y_pred = (model.predict(X_test_pad, verbose=0) > 0.5).astype(int).flatten()

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred).tolist()
    report = classification_report(y_test, y_pred, target_names=["HAM", "SPAM"])

    print("\n" + "=" * 55)
    print("  LSTM MODEL — EVALUATION REPORT")
    print("=" * 55)
    print(f"  Test loss:      {loss:.4f}")
    print(f"  Test accuracy:  {acc:.4f}")
    print(f"  Precision:      {precision:.4f}")
    print(f"  Recall:         {recall:.4f}")
    print(f"  F1 Score:       {f1:.4f}")
    print()
    print(report)
    print("  Confusion Matrix:")
    print(f"    {cm[0]}")
    print(f"    {cm[1]}")
    print("=" * 55)

    # 8. Save artifacts
    model.save(LSTM_MODEL_PATH)
    joblib.dump(tokenizer, TOKENIZER_PATH)
    logger.info("Saved model     → %s", LSTM_MODEL_PATH)
    logger.info("Saved tokenizer → %s", TOKENIZER_PATH)

    # 9. Export metrics to JSON for CI / tracking
    metrics = {
        "model": "lstm",
        "test_loss": round(loss, 4),
        "test_accuracy": round(acc, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "confusion_matrix": cm,
        "epochs_completed": len(history.history["loss"]),
        "hyperparameters": {
            "vocab_size": VOCAB_SIZE,
            "embedding_dim": EMBEDDING_DIM,
            "lstm_units": LSTM_UNITS,
            "max_seq_len": MAX_SEQUENCE_LENGTH,
            "dropout": DROPOUT_RATE,
            "batch_size": BATCH_SIZE,
            "random_state": RANDOM_STATE,
        },
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    logger.info("Saved metrics   → %s", METRICS_PATH)


if __name__ == "__main__":
    main()