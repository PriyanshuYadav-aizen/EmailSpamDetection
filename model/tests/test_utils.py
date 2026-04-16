"""Unit tests for ``utils.clean_text``."""

from __future__ import annotations

import pytest

from utils import clean_text
from exceptions import PreprocessingError


class TestCleanText:
    """Tests for the shared text-cleaning pipeline."""

    def test_lowercases(self):
        assert clean_text("HELLO WORLD") == "hello world"

    def test_removes_http_urls(self):
        result = clean_text("visit http://example.com today")
        assert "http" not in result
        assert "example" not in result

    def test_removes_www_urls(self):
        result = clean_text("go to www.example.com now")
        assert "www" not in result

    def test_strips_punctuation(self):
        assert clean_text("hello, world!") == "hello world"

    def test_removes_digits(self):
        assert clean_text("order 12345 confirmed") == "order confirmed"

    def test_collapses_whitespace(self):
        assert clean_text("hello   world") == "hello world"

    def test_trims(self):
        assert clean_text("  hello  ") == "hello"

    def test_empty_string(self):
        assert clean_text("") == ""

    def test_full_pipeline(self):
        raw = "VISIT http://spam.com — win 500$ NOW!!!"
        result = clean_text(raw)
        assert result == "visit win now"

    def test_rejects_non_string(self):
        with pytest.raises(PreprocessingError):
            clean_text(42)  # type: ignore[arg-type]

    def test_rejects_none(self):
        with pytest.raises(PreprocessingError):
            clean_text(None)  # type: ignore[arg-type]
