"""FastAPI service for email spam detection.

Exposes REST endpoints that accept email text and return spam / ham
predictions from an ensemble of three models.

Features
--------
* Single and batch prediction endpoints
* Model warm-up at startup  (no cold-start penalty)
* CORS middleware for cross-origin front-end calls
* Structured JSON logging with request context
* Global exception handlers for custom error types
* Pydantic response schemas for automatic OpenAPI docs
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from exceptions import ModelLoadError, PredictionError, SpamDetectionError
from predict_main import predict, predict_batch, warmup

# ── Logging ───────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ── Application lifecycle ────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm up models on startup; log shutdown."""
    warmup()
    logger.info("Spam-detection API ready to serve")
    yield
    logger.info("Spam-detection API shutting down …")


app = FastAPI(
    title="Spam Detection API",
    description=(
        "Ensemble email spam classifier combining a rule engine, "
        "TF-IDF + Logistic Regression, and a Bidirectional LSTM."
    ),
    version="2.0.0",
    lifespan=lifespan,
)


# ── Middleware ────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    """Attach a unique request ID and log request timing."""
    request_id = str(uuid.uuid4())[:8]
    request.state.request_id = request_id
    start = time.perf_counter()

    response = await call_next(request)

    elapsed_ms = (time.perf_counter() - start) * 1_000
    logger.info(
        "[%s] %s %s → %d  (%.1f ms)",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    response.headers["X-Request-ID"] = request_id
    return response


# ── Exception handlers ───────────────────────────────────────────────────

@app.exception_handler(ModelLoadError)
async def handle_model_load_error(request: Request, exc: ModelLoadError):
    """Return 503 when a model artifact is unavailable."""
    logger.error("Model load failure: %s", exc)
    return JSONResponse(
        status_code=503,
        content={"detail": str(exc), "type": "model_load_error"},
    )


@app.exception_handler(PredictionError)
async def handle_prediction_error(request: Request, exc: PredictionError):
    """Return 500 when inference fails."""
    logger.error("Prediction failure: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc), "type": "prediction_error"},
    )


@app.exception_handler(SpamDetectionError)
async def handle_generic_error(request: Request, exc: SpamDetectionError):
    """Catch-all for any other custom exception."""
    logger.error("Spam detection error: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc), "type": "spam_detection_error"},
    )


# ── Schemas ───────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    """Incoming single-prediction payload."""

    content: str = Field(
        min_length=1,
        description="Email body text to classify as spam or ham.",
    )


class BatchPredictRequest(BaseModel):
    """Incoming batch-prediction payload."""

    contents: list[str] = Field(
        min_length=1,
        max_length=100,
        description="List of email texts (max 100 per request).",
    )


class PredictResponse(BaseModel):
    """Prediction result with per-model confidence breakdown."""

    label: str
    score: float
    rule_conf: float
    tfidf_prob: float
    lstm_prob: float


class BatchPredictResponse(BaseModel):
    """Batch prediction wrapper."""

    results: list[PredictResponse]


class HealthResponse(BaseModel):
    """Health-check response."""

    status: str


# ── Routes ────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health_check():
    """Liveness probe — returns 200 when the service is up."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse, tags=["inference"])
def predict_email(payload: PredictRequest):
    """Classify a single email as **SPAM** or **HAM**.

    Returns the ensemble label, composite score, and individual
    model confidences so the caller can inspect each signal.
    """
    content = payload.content.strip()
    if not content:
        raise HTTPException(status_code=400, detail="content must not be blank")

    logger.info("Single prediction request (length=%d chars)", len(content))
    result = predict(content)
    logger.info("Result: %s (score=%.3f)", result["label"], result["score"])
    return result


@app.post(
    "/predict/batch",
    response_model=BatchPredictResponse,
    tags=["inference"],
)
def predict_email_batch(payload: BatchPredictRequest):
    """Classify up to 100 emails in a single request.

    Vectorisation and LSTM inference are batched internally for
    better throughput compared to repeated single calls.
    """
    cleaned = [c.strip() for c in payload.contents if c.strip()]
    if not cleaned:
        raise HTTPException(status_code=400, detail="all contents are blank")

    logger.info("Batch prediction request (%d emails)", len(cleaned))
    results = predict_batch(cleaned)
    return {"results": results}
