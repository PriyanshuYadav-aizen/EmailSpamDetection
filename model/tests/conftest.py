"""Shared fixtures for the spam-detection test suite."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Add the model directory to sys.path so imports work like they do at
# runtime (e.g., ``from utils import clean_text``).
MODEL_DIR = Path(__file__).resolve().parent.parent
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))


@pytest.fixture()
def sample_ham_texts() -> list[str]:
    """A small set of clearly legitimate email texts."""
    return [
        "Hi John, the meeting has been moved to 3 PM tomorrow.",
        "Please find the quarterly report attached.",
        "Can you review the pull request I submitted yesterday?",
        "Let's catch up over coffee next week.",
        "The project deadline has been extended to Friday.",
    ]


@pytest.fixture()
def sample_spam_texts() -> list[str]:
    """A small set of clearly spammy email texts."""
    return [
        "Congratulations! You have won a free lottery prize!",
        "URGENT: click here to claim your cash prize now!",
        "Make money fast! Free income opportunity!",
        "Win free cash — act now, limited time offer!",
        "Earn money from home, no effort needed, click here!",
    ]
