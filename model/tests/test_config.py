"""Unit tests for configuration module."""

from __future__ import annotations

import os
from pathlib import Path

import pytest


class TestConfig:
    """Tests for ``config.py`` values and env-var overrides."""

    def test_base_dir_exists(self):
        from config import BASE_DIR
        assert isinstance(BASE_DIR, Path)
        assert BASE_DIR.is_dir()

    def test_default_threshold(self):
        from config import SPAM_THRESHOLD
        assert 0.0 < SPAM_THRESHOLD < 1.0

    def test_weights_sum_to_one(self):
        from config import RULE_WEIGHT, TFIDF_WEIGHT, LSTM_WEIGHT
        total = RULE_WEIGHT + TFIDF_WEIGHT + LSTM_WEIGHT
        assert abs(total - 1.0) < 1e-9

    def test_spam_keywords_non_empty(self):
        from config import SPAM_KEYWORDS
        assert len(SPAM_KEYWORDS) > 0
        assert all(isinstance(kw, str) for kw in SPAM_KEYWORDS)

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("SPAM_THRESHOLD", "0.99")
        # Force re-import
        import importlib
        import config
        importlib.reload(config)
        assert config.SPAM_THRESHOLD == 0.99
        # Restore
        monkeypatch.delenv("SPAM_THRESHOLD")
        importlib.reload(config)
