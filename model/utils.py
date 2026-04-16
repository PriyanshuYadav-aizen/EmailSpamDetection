"""Shared text-preprocessing utilities for the spam detection pipeline.

Every module that needs to clean raw email text should import
``clean_text`` from here instead of re-implementing the pipeline.
"""

from __future__ import annotations

import re
import string

from exceptions import PreprocessingError


# Pre-compile regexes once at module level for performance
_RE_URL = re.compile(r"http\S+|www\S+")
_RE_DIGITS = re.compile(r"\d+")
_RE_WHITESPACE = re.compile(r"\s+")
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def clean_text(text: str) -> str:
    """Normalise and strip noise from raw email text.

    Pipeline
    --------
    1. Lower-case the entire string.
    2. Remove URLs (``http…`` / ``www…``).
    3. Strip all ASCII punctuation.
    4. Remove standalone digits.
    5. Collapse consecutive whitespace and trim.

    Parameters
    ----------
    text : str
        Raw email body or subject line.

    Returns
    -------
    str
        Cleaned, lower-cased text ready for vectorisation.

    Raises
    ------
    PreprocessingError
        If *text* is not a string.
    """
    if not isinstance(text, str):
        raise PreprocessingError(f"Expected str, got {type(text).__name__}")

    text = text.lower()
    text = _RE_URL.sub("", text)
    text = text.translate(_PUNCT_TABLE)
    text = _RE_DIGITS.sub("", text)
    text = _RE_WHITESPACE.sub(" ", text).strip()
    return text
