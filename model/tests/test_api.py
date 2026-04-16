"""Integration tests for the FastAPI application."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api import app

client = TestClient(app)


class TestHealthEndpoint:
    """Tests for ``GET /health``."""

    def test_returns_200(self):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_returns_ok(self):
        data = client.get("/health").json()
        assert data["status"] == "ok"


class TestPredictEndpoint:
    """Tests for ``POST /predict``."""

    def test_valid_request(self):
        resp = client.post("/predict", json={"content": "hello world"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["label"] in ("SPAM", "HAM")

    def test_spam_input(self):
        resp = client.post(
            "/predict",
            json={"content": "FREE money CLICK here WIN a PRIZE now!!"},
        )
        assert resp.status_code == 200
        assert resp.json()["label"] == "SPAM"

    def test_empty_content_rejected(self):
        resp = client.post("/predict", json={"content": ""})
        assert resp.status_code == 422  # Pydantic validation

    def test_missing_content_rejected(self):
        resp = client.post("/predict", json={})
        assert resp.status_code == 422

    def test_response_has_all_fields(self):
        resp = client.post("/predict", json={"content": "some text"})
        data = resp.json()
        for key in ("label", "score", "rule_conf", "tfidf_prob", "lstm_prob"):
            assert key in data

    def test_request_id_header(self):
        resp = client.post("/predict", json={"content": "test"})
        assert "x-request-id" in resp.headers


class TestBatchPredictEndpoint:
    """Tests for ``POST /predict/batch``."""

    def test_valid_batch(self):
        resp = client.post(
            "/predict/batch",
            json={"contents": ["hello", "free money"]},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["results"]) == 2

    def test_empty_list_rejected(self):
        resp = client.post("/predict/batch", json={"contents": []})
        assert resp.status_code == 422

    def test_all_blank_rejected(self):
        resp = client.post(
            "/predict/batch",
            json={"contents": ["   ", "  "]},
        )
        assert resp.status_code == 400
