"""Ensemble spam-detection inference.

Combines three complementary signals to classify email text:

    1. **Rule engine** — keyword / URL heuristic         (weight 0.25)
    2. **TF-IDF + Logistic Regression**                  (weight 0.35)
    3. **Bidirectional LSTM**                             (weight 0.40)

The final score is a weighted average; a threshold of 0.55 decides
the label.  Individual confidences are returned alongside the
ensemble result for full transparency.

Models are loaded lazily on first prediction and cached for the
lifetime of the process.
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import Any
from urllib.parse import urlparse

import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from config import (
    LSTM_MODEL_PATH,
    LSTM_WEIGHT,
    MAX_INPUT_LENGTH,
    MAX_SEQUENCE_LENGTH,
    RULE_WEIGHT,
    SPAM_KEYWORDS,
    SPAM_THRESHOLD,
    TFIDF_MODEL_PATH,
    TFIDF_WEIGHT,
    TOKENIZER_PATH,
    VECTORIZER_PATH,
)
from exceptions import ModelLoadError, PredictionError
from utils import clean_text

logger = logging.getLogger(__name__)

TRUSTED_DOMAINS = (
    "amazon.com",
    "amazon.in",
    "google.com",
    "microsoft.com",
    "apple.com",
)
_URL_RE = re.compile(r"https?://[^\s]+|www\.[^\s]+", re.IGNORECASE)
FINANCIAL_KEYWORDS = (
    "payment",
    "billing",
    "invoice",
    "transaction",
    "money",
    "bank",
    "card",
)
ACTION_REQUEST_PHRASES = (
    "click",
    "verify",
    "confirm",
    "update",
    "pay now",
    "login",
    "review",
    "resolve",
    "reset",
)
URGENCY_PHRASES = (
    "urgent",
    "immediately",
    "within 24 hours",
    "act now",
    "final warning",
    "suspended",
    "suspension",
)
PHISHING_PHRASES = (
    "password reset",
    "reset your password",
    "verify your password",
    "verify your account",
    "confirm your account",
    "account will be suspended",
    "unusual activity",
    "login attempt",
    "secure your account",
)


def _count_hits(text: str, phrases: tuple[str, ...]) -> int:
    return sum(phrase in text for phrase in phrases)


def _extract_hosts(text: str) -> list[str]:
    """Extract normalized URL hosts from raw text."""
    hosts: list[str] = []
    for match in _URL_RE.findall(text):
        candidate = match if match.lower().startswith(("http://", "https://")) else f"http://{match}"
        try:
            host = (urlparse(candidate).hostname or "").lower().strip(".")
            if host:
                hosts.append(host)
        except Exception:
            continue
    return hosts


def _is_exact_or_subdomain(host: str, domain: str) -> bool:
    return host == domain or host.endswith(f".{domain}")


def _all_links_trusted(hosts: list[str]) -> bool:
    return bool(hosts) and all(
        any(_is_exact_or_subdomain(host, domain) for domain in TRUSTED_DOMAINS)
        for host in hosts
    )


def _has_brand_spoofed_link(hosts: list[str]) -> bool:
    for host in hosts:
        for domain in TRUSTED_DOMAINS:
            if domain in host and not _is_exact_or_subdomain(host, domain):
                return True
    return False


def _blend_with_dynamic_rule_weight(rule_conf: float, tfidf_prob: float, lstm_prob: float, rule_weight: float) -> float:
    """Blend model confidences while increasing rule influence on risky context."""
    other_total = TFIDF_WEIGHT + LSTM_WEIGHT
    if other_total <= 0:
        return rule_conf

    rule_w = max(0.0, min(1.0, rule_weight))
    other_scale = (1.0 - rule_w) / other_total
    tfidf_w = TFIDF_WEIGHT * other_scale
    lstm_w = LSTM_WEIGHT * other_scale

    return (rule_w * rule_conf) + (tfidf_w * tfidf_prob) + (lstm_w * lstm_prob)


def _apply_phishing_safety_layer(raw: str, rule_conf: float, weighted_score: float) -> tuple[float, float]:
    """Apply phishing-specific guardrails and return (final_score, adjusted_rule_conf)."""
    final_score = weighted_score
    adjusted_rule_conf = rule_conf

    hosts = _extract_hosts(raw)
    has_link = bool(hosts)
    all_trusted_links = _all_links_trusted(hosts)
    has_spoofed_brand_link = _has_brand_spoofed_link(hosts)

    financial_hits = _count_hits(raw, FINANCIAL_KEYWORDS)
    urgency_hits = _count_hits(raw, URGENCY_PHRASES)
    action_hits = _count_hits(raw, ACTION_REQUEST_PHRASES)
    phishing_phrase_hits = _count_hits(raw, PHISHING_PHRASES)

    has_financial_context = financial_hits >= 1
    has_urgency = urgency_hits >= 1
    has_action_request = action_hits >= 1

    # Requested pattern: (payment OR billing) + (failed OR update) + link => high spam
    has_payment_or_billing = ("payment" in raw) or ("billing" in raw)
    has_failed_or_update = ("failed" in raw) or ("update" in raw)
    payment_billing_pattern = has_link and has_payment_or_billing and has_failed_or_update

    # Requested rule: money/payment context + action request + link => spam
    financial_action_link_pattern = has_link and has_financial_context and has_action_request

    # Keep balance: identify likely legitimate financial notices.
    benign_trusted_financial_notice = (
        all_trusted_links
        and not has_spoofed_brand_link
        and has_financial_context
        and not has_urgency
        and not has_action_request
        and not has_failed_or_update
        and phishing_phrase_hits == 0
    )

    if benign_trusted_financial_notice:
        adjusted_rule_conf = min(adjusted_rule_conf, 0.45)
        final_score = min(final_score, 0.45)

    # Keep balance: dampen mild trusted-link messages when no phishing intent is present.
    if (
        all_trusted_links
        and not has_spoofed_brand_link
        and not benign_trusted_financial_notice
        and not payment_billing_pattern
        and not financial_action_link_pattern
        and phishing_phrase_hits == 0
        and adjusted_rule_conf < 0.8
    ):
        adjusted_rule_conf *= 0.6
        final_score *= 0.85

    # Strong signals must not be overridden by ML outputs.
    if has_spoofed_brand_link:
        final_score = max(final_score, 0.92)
    if payment_billing_pattern:
        final_score = max(final_score, 0.90)
    if financial_action_link_pattern:
        final_score = max(final_score, 0.88)
    if has_link and has_financial_context and has_action_request and has_urgency:
        final_score = max(final_score, 0.95)
    if phishing_phrase_hits >= 2 and has_link:
        final_score = max(final_score, 0.88)

    # Apply broad rule-based escalation only when non-benign risk context exists.
    non_benign_risk_context = (
        has_spoofed_brand_link
        or payment_billing_pattern
        or financial_action_link_pattern
        or (has_link and has_urgency)
        or (has_link and not all_trusted_links)
    )

    if non_benign_risk_context:
        if adjusted_rule_conf >= 0.8:
            final_score = max(0.95, final_score)
        elif adjusted_rule_conf >= 0.6:
            final_score = max(0.85, final_score)

    return final_score, adjusted_rule_conf


# ── Lazy model loading ───────────────────────────────────────────────────
#
# Models are loaded on *first call*, not at import time, so unit tests
# and CLI scripts that never call ``predict()`` don't pay the
# TensorFlow start-up cost.
# ──────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_tfidf_artifacts():
    """Load the TF-IDF vectoriser and Logistic Regression classifier.

    Raises
    ------
    ModelLoadError
        If the pickle files are missing or corrupt.
    """
    try:
        logger.info("Loading TF-IDF artifacts …")
        model = joblib.load(TFIDF_MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        logger.info("TF-IDF artifacts loaded successfully")
        return model, vectorizer
    except Exception as exc:
        raise ModelLoadError("tfidf", str(exc)) from exc


@lru_cache(maxsize=1)
def _load_lstm_artifacts():
    """Load the Keras LSTM model and its tokeniser.

    Raises
    ------
    ModelLoadError
        If the model file or tokeniser pickle is missing or corrupt.
    """
    try:
        logger.info("Loading LSTM artifacts …")
        model = load_model(LSTM_MODEL_PATH)
        tokenizer = joblib.load(TOKENIZER_PATH)
        logger.info("LSTM artifacts loaded successfully")
        return model, tokenizer
    except Exception as exc:
        raise ModelLoadError("lstm", str(exc)) from exc


def warmup() -> None:
    """Pre-load all model artifacts.

    Call this at application startup so the first real request isn't
    penalised by cold-load latency.
    """
    logger.info("Warming up models …")
    _load_tfidf_artifacts()
    _load_lstm_artifacts()
    logger.info("All models warm")


# ── Rule engine ──────────────────────────────────────────────────────────

def rule_engine(text: str) -> float:
    """Score email text against known spam indicators.

    Checks
    ------
    * Presence of configurable spam keywords (see ``config.SPAM_KEYWORDS``).
    * Raw URLs (``http`` / ``www``) that survive before cleaning.

    Parameters
    ----------
    text : str
        Lower-cased but otherwise **raw** email text (before
        ``clean_text`` is applied) so that URL signals are preserved.

    Returns
    -------
    float
        Confidence score clamped to **[0.0, 1.0]**.
    """
    # Keywords often found in legitimate emails
    HAM_KEYWORDS = ["unsubscribe", "sincerely", "regards", "best regards", "thank you", "customer support", "hello"]
    
    tokens = set(text.split())
    keyword_hits = sum(
        1 for kw in SPAM_KEYWORDS if kw in tokens or kw in text
    )
    
    # Legit signals (negative score)
    ham_hits = sum(
        1 for kw in HAM_KEYWORDS if kw in tokens or kw in text
    )
    
    # URL is a moderate signal (1), not a heavy one (2)
    url_bonus = 1 if ("http" in text or "www" in text) else 0
    
    # Calculate balance
    raw_score = keyword_hits + url_bonus - (ham_hits * 0.5)
    
    return max(0.0, min(raw_score / 4.0, 1.0))


# ── Ensemble prediction ─────────────────────────────────────────────────

def predict(text: str) -> dict[str, Any]:
    """Run the full ensemble pipeline on a single email.

    Parameters
    ----------
    text : str
        Raw (uncleaned) email text.

    Returns
    -------
    dict
        ``label``      – ``"SPAM"`` or ``"HAM"``
        ``score``      – weighted ensemble confidence
        ``rule_conf``  – rule-engine confidence
        ``tfidf_prob`` – TF-IDF model probability
        ``lstm_prob``  – LSTM model probability

    Raises
    ------
    PredictionError
        If any downstream model raises during inference.
    ValueError
        If *text* exceeds ``MAX_INPUT_LENGTH``.
    """
    if len(text) > MAX_INPUT_LENGTH:
        raise ValueError(
            f"Input text exceeds maximum length "
            f"({len(text):,} > {MAX_INPUT_LENGTH:,} chars)"
        )

    raw = text.lower()
    cleaned = clean_text(raw)

    # 1. Rule engine —————————————————————————————————————————————
    rule_conf = rule_engine(raw)

    # 2. TF-IDF + Logistic Regression ————————————————————————————
    try:
        tfidf_model, vectorizer = _load_tfidf_artifacts()
        vec = vectorizer.transform([cleaned])
        tfidf_prob = float(tfidf_model.predict_proba(vec)[0][1])
    except ModelLoadError:
        raise
    except Exception as exc:
        raise PredictionError("tfidf", str(exc)) from exc

    # 3. Bidirectional LSTM ———————————————————————————————————————
    try:
        lstm_model, tokenizer = _load_lstm_artifacts()
        seq = tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
        lstm_prob = float(lstm_model.predict(padded, verbose=0)[0][0])
    except ModelLoadError:
        raise
    except Exception as exc:
        raise PredictionError("lstm", str(exc)) from exc

    # 4. Weighted ensemble ———————————————————————————————————————
    # 4. Weighted ensemble with requested dynamic rule boost
    has_link = bool(_extract_hosts(raw))
    has_financial_context = _count_hits(raw, FINANCIAL_KEYWORDS) >= 1
    has_urgency = _count_hits(raw, URGENCY_PHRASES) >= 1

    dynamic_rule_weight = RULE_WEIGHT
    if has_link and has_financial_context and has_urgency:
        dynamic_rule_weight = max(RULE_WEIGHT, 0.70)

    weighted_score = _blend_with_dynamic_rule_weight(
        rule_conf=rule_conf,
        tfidf_prob=tfidf_prob,
        lstm_prob=lstm_prob,
        rule_weight=dynamic_rule_weight,
    )

    # 5. Heuristic safety layer
    final_score, rule_conf = _apply_phishing_safety_layer(raw, rule_conf, weighted_score)

    label = "SPAM" if final_score > SPAM_THRESHOLD else "HAM"

    logger.debug(
        "rule=%.3f  tfidf=%.3f  lstm=%.3f  → final=%.3f (%s)",
        rule_conf, tfidf_prob, lstm_prob, final_score, label,
    )

    return {
        "label": label,
        "score": round(final_score, 3),
        "rule_conf": round(rule_conf, 3),
        "tfidf_prob": round(tfidf_prob, 3),
        "lstm_prob": round(lstm_prob, 3),
    }


def predict_batch(texts: list[str]) -> list[dict[str, Any]]:
    """Run the ensemble on multiple emails.

    Internally vectorises and pads in batch for better throughput.

    Parameters
    ----------
    texts : list[str]
        List of raw email texts.

    Returns
    -------
    list[dict]
        One result dict per input text (same schema as ``predict``).
    """
    if not texts:
        return []

    for i, t in enumerate(texts):
        if len(t) > MAX_INPUT_LENGTH:
            raise ValueError(
                f"Input [{i}] exceeds max length "
                f"({len(t):,} > {MAX_INPUT_LENGTH:,} chars)"
            )

    raws = [t.lower() for t in texts]
    cleaned = [clean_text(r) for r in raws]

    # Rule engine (per-item)
    rule_confs = [rule_engine(r) for r in raws]

    # TF-IDF (batch)
    try:
        tfidf_model, vectorizer = _load_tfidf_artifacts()
        vecs = vectorizer.transform(cleaned)
        tfidf_probs = tfidf_model.predict_proba(vecs)[:, 1].tolist()
    except ModelLoadError:
        raise
    except Exception as exc:
        raise PredictionError("tfidf-batch", str(exc)) from exc

    # LSTM (batch)
    try:
        lstm_model, tokenizer = _load_lstm_artifacts()
        seqs = tokenizer.texts_to_sequences(cleaned)
        padded = pad_sequences(seqs, maxlen=MAX_SEQUENCE_LENGTH)
        lstm_probs = lstm_model.predict(padded, verbose=0).flatten().tolist()
    except ModelLoadError:
        raise
    except Exception as exc:
        raise PredictionError("lstm-batch", str(exc)) from exc

    # Assemble results
    results: list[dict[str, Any]] = []

    for raw, rc, tp, lp in zip(raws, rule_confs, tfidf_probs, lstm_probs):
        has_link = bool(_extract_hosts(raw))
        has_financial_context = _count_hits(raw, FINANCIAL_KEYWORDS) >= 1
        has_urgency = _count_hits(raw, URGENCY_PHRASES) >= 1

        dynamic_rule_weight = RULE_WEIGHT
        if has_link and has_financial_context and has_urgency:
            dynamic_rule_weight = max(RULE_WEIGHT, 0.70)

        weighted = _blend_with_dynamic_rule_weight(
            rule_conf=float(rc),
            tfidf_prob=float(tp),
            lstm_prob=float(lp),
            rule_weight=dynamic_rule_weight,
        )

        score, adjusted_rc = _apply_phishing_safety_layer(raw, float(rc), float(weighted))
            
        results.append({
            "label": "SPAM" if score > SPAM_THRESHOLD else "HAM",
            "score": round(float(score), 3),
            "rule_conf": round(float(adjusted_rc), 3),
            "tfidf_prob": round(float(tp), 3),
            "lstm_prob": round(float(lp), 3),
        })

    return results


# ── CLI ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    user_input = input("\nEnter email text: ")
    result = predict(user_input)

    print("\n" + "=" * 40)
    print("  ENSEMBLE PREDICTION RESULT")
    print("=" * 40)
    for key, value in result.items():
        print(f"  {key:>12}: {value}")
    print("=" * 40)