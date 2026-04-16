"""Unit tests for the ensemble prediction pipeline."""

from __future__ import annotations

import pytest

from predict_main import predict, predict_batch, rule_engine
from config import MAX_INPUT_LENGTH


class TestRuleEngine:
    """Tests for the keyword-based rule engine."""

    def test_no_keywords_returns_zero(self):
        assert rule_engine("hello how are you doing today") == 0.0

    def test_single_keyword(self):
        score = rule_engine("this is a free trial")
        assert score > 0.0

    def test_url_bonus(self):
        score_no_url = rule_engine("check this offer")
        score_with_url = rule_engine("check this http://example.com offer")
        assert score_with_url > score_no_url

    def test_capped_at_one(self):
        # Load up all the keywords
        text = "free money win prize offer click urgent cash earn lottery"
        score = rule_engine(text)
        assert score <= 1.0

    def test_empty_string(self):
        assert rule_engine("") == 0.0


class TestPredict:
    """Tests for the full ensemble ``predict()``."""

    def test_returns_expected_keys(self):
        result = predict("hello world")
        assert set(result.keys()) == {
            "label", "score", "rule_conf", "tfidf_prob", "lstm_prob",
        }

    def test_label_is_spam_or_ham(self):
        result = predict("some email text")
        assert result["label"] in ("SPAM", "HAM")

    def test_scores_are_bounded(self):
        result = predict("win free money now")
        assert 0.0 <= result["score"] <= 1.0
        assert 0.0 <= result["rule_conf"] <= 1.0
        assert 0.0 <= result["tfidf_prob"] <= 1.0
        assert 0.0 <= result["lstm_prob"] <= 1.0

    def test_obvious_spam_flagged(self, sample_spam_texts):
        for text in sample_spam_texts:
            result = predict(text)
            assert result["label"] == "SPAM", f"Expected SPAM for: {text!r}"

    def test_obvious_ham_flagged(self, sample_ham_texts):
        for text in sample_ham_texts:
            result = predict(text)
            assert result["label"] == "HAM", f"Expected HAM for: {text!r}"

    def test_rejects_oversized_input(self):
        long_text = "a " * (MAX_INPUT_LENGTH + 1)
        with pytest.raises(ValueError, match="exceeds maximum length"):
            predict(long_text)

    def test_high_confidence_phishing_boost(self):
        # A template matching multiple keywords and having a URL
        phishing_text = (
            "Subject: Security Alert: Unusual activity detected. "
            "Please verify your bank account here: http://secure-bank.com/verify"
        )
        result = predict(phishing_text)
        assert result["label"] == "SPAM"
        # We expect > 0.95 due to the boost logic
        assert result["score"] >= 0.95
        assert result["rule_conf"] == 1.0


class TestPredictBatch:
    """Tests for ``predict_batch()``."""

    def test_empty_list(self):
        assert predict_batch([]) == []

    def test_returns_list_of_dicts(self):
        results = predict_batch(["hello", "free money"])
        assert isinstance(results, list)
        assert len(results) == 2
        for r in results:
            assert "label" in r

    def test_batch_matches_single(self):
        texts = ["hello world", "free money click here"]
        batch = predict_batch(texts)
        singles = [predict(t) for t in texts]
        for b, s in zip(batch, singles):
            assert b["label"] == s["label"]
            assert abs(b["score"] - s["score"]) < 0.01
