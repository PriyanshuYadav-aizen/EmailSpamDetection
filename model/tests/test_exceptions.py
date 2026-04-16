"""Unit tests for custom exceptions."""

from __future__ import annotations

from exceptions import ModelLoadError, PredictionError, PreprocessingError, SpamDetectionError


class TestExceptionHierarchy:
    """Verify the exception class hierarchy."""

    def test_model_load_error_is_spam_detection_error(self):
        assert issubclass(ModelLoadError, SpamDetectionError)

    def test_prediction_error_is_spam_detection_error(self):
        assert issubclass(PredictionError, SpamDetectionError)

    def test_preprocessing_error_is_spam_detection_error(self):
        assert issubclass(PreprocessingError, SpamDetectionError)


class TestModelLoadError:
    def test_message_includes_artifact(self):
        exc = ModelLoadError("lstm", "file not found")
        assert "lstm" in str(exc)
        assert "file not found" in str(exc)

    def test_message_without_reason(self):
        exc = ModelLoadError("tfidf")
        assert "tfidf" in str(exc)


class TestPredictionError:
    def test_message_includes_model_name(self):
        exc = PredictionError("lstm", "bad input shape")
        assert "lstm" in str(exc)
        assert "bad input shape" in str(exc)
