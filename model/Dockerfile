# ── Build stage ───────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Runtime stage ─────────────────────────────────────────────────────────
FROM python:3.11-slim

LABEL maintainer="Email Spam Detection Team"
LABEL description="Ensemble spam classifier API (rule + TF-IDF + LSTM)"

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code and model artifacts
COPY . .

# Non-root user for security
RUN useradd --create-home appuser
USER appuser

EXPOSE ${PORT:-5001}

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://localhost:{os.environ.get(\"PORT\",5001)}/health')"

CMD uvicorn api:app --host 0.0.0.0 --port ${PORT:-5001}
