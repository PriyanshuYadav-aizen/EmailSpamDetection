"""Custom exception hierarchy for the spam-detection service."""


class SpamDetectionError(Exception):
    """Base exception for all spam-detection errors."""


class ModelLoadError(SpamDetectionError):
    """Raised when a serialised model or artifact cannot be loaded."""

    def __init__(self, artifact: str, reason: str | None = None) -> None:
        self.artifact = artifact
        self.reason = reason
        msg = f"Failed to load artifact '{artifact}'"
        if reason:
            msg += f": {reason}"
        super().__init__(msg)


class PredictionError(SpamDetectionError):
    """Raised when inference fails for a given input."""

    def __init__(self, model_name: str, reason: str | None = None) -> None:
        self.model_name = model_name
        self.reason = reason
        msg = f"Prediction failed in '{model_name}'"
        if reason:
            msg += f": {reason}"
        super().__init__(msg)


class PreprocessingError(SpamDetectionError):
    """Raised when input text cannot be cleaned / tokenised."""
