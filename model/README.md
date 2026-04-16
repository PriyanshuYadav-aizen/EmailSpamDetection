# Spam Detection — Model Service

Ensemble email spam classifier combining three complementary signals for
robust, interpretable predictions.

| Signal | Model | Weight | Purpose |
| ------ | ----- | ------ | ------- |
| Rule engine | Keyword + URL heuristic | 25 % | Fast, deterministic baseline |
| TF-IDF | Logistic Regression (`sklearn`) | 35 % | Strong on word / bigram patterns |
| Deep learning | Bidirectional LSTM (`Keras`) | 40 % | Captures sequential context |

## Architecture

```
┌──────────────┐
│  Raw Email   │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  clean_text  │  (utils.py — single source of truth)
└──────┬───────┘
       │
       ├──────────────────┬──────────────────┐
       ▼                  ▼                  ▼
┌─────────────┐   ┌──────────────┐   ┌─────────────┐
│ Rule Engine │   │   TF-IDF +   │   │ Bi-LSTM     │
│ (keywords)  │   │   LogReg     │   │ (Keras)     │
└──────┬──────┘   └──────┬───────┘   └──────┬──────┘
       │ 0.25            │ 0.35             │ 0.40
       └─────────────────┼──────────────────┘
                         ▼
                ┌─────────────────┐
                │ Weighted Score  │
                │ → SPAM / HAM   │
                └─────────────────┘
```

## Project Structure

```
model/
├── __init__.py            # Package marker with version
├── config.py              # Centralised hyperparameters & env-var overrides
├── utils.py               # Shared text-preprocessing pipeline
├── exceptions.py          # Custom exception hierarchy
│
├── tfidf_model.py         # Training — TF-IDF + Logistic Regression
├── lstm_model.py          # Training — Bidirectional LSTM
│
├── tfidf_predict.py       # Standalone TF-IDF inference CLI
├── lstm_predict.py        # Standalone LSTM inference CLI
├── predict_main.py        # Ensemble inference (rule + TF-IDF + LSTM)
├── api.py                 # FastAPI REST service
│
├── tests/                 # Pytest test suite
│   ├── conftest.py        #   Shared fixtures
│   ├── test_utils.py      #   Preprocessing tests
│   ├── test_predict.py    #   Prediction pipeline tests
│   ├── test_api.py        #   API integration tests
│   ├── test_config.py     #   Configuration tests
│   └── test_exceptions.py #   Exception hierarchy tests
│
├── spam_model.pkl         # Saved LogisticRegression
├── vectorizer.pkl         # Saved TfidfVectorizer
├── lstm_model.keras       # Saved Keras LSTM
├── tokenizer.pkl          # Saved Keras Tokenizer
│
├── Dockerfile             # Multi-stage container build
├── .dockerignore
├── requirements.txt       # Pinned Python dependencies
├── .gitignore
└── README.md
```

## Quick Start

### 1. Install dependencies

```bash
py -3.11 -m pip install -r requirements.txt
```

### 2. Run the API

```bash
py -3.11 -m uvicorn api:app --host 0.0.0.0 --port 5001
```

### 3. Health check

```
GET http://localhost:5001/health
→ {"status": "ok"}
```

## API Reference

### `POST /predict`

Classify a single email.

**Request**

```json
{ "content": "free money now" }
```

**Response**

```json
{
  "label": "SPAM",
  "score": 0.931,
  "rule_conf": 0.4,
  "tfidf_prob": 0.912,
  "lstm_prob": 0.945
}
```

### `POST /predict/batch`

Classify up to **100 emails** in one request (vectorisation is batched
internally for better throughput).

**Request**

```json
{ "contents": ["free money", "meeting at 3 PM"] }
```

**Response**

```json
{
  "results": [
    { "label": "SPAM", "score": 0.87, ... },
    { "label": "HAM",  "score": 0.12, ... }
  ]
}
```

### Response Fields

| Field | Type | Description |
| ----- | ---- | ----------- |
| `label` | `str` | `"SPAM"` or `"HAM"` |
| `score` | `float` | Weighted ensemble confidence (0 – 1) |
| `rule_conf` | `float` | Rule-engine confidence |
| `tfidf_prob` | `float` | TF-IDF model probability |
| `lstm_prob` | `float` | LSTM model probability |

## Configuration

All hyperparameters live in `config.py` and can be overridden with
environment variables using the **`SPAM_`** prefix:

```bash
SPAM_THRESHOLD=0.60    # raise decision boundary
SPAM_MAX_INPUT_LENGTH=100000
SPAM_RULE_WEIGHT=0.30
```

## Testing

```bash
py -3.11 -m pytest tests/ -v
```

Tests cover:
- **Preprocessing** — every step of the `clean_text` pipeline
- **Exceptions** — hierarchy and message formatting
- **Configuration** — defaults, bounds, env-var overrides
- **Prediction** — schema, score bounds, known-input accuracy
- **API** — endpoints, validation, error codes, batch mode

## Re-training

> The training scripts expect `enron.csv` in this directory (git-ignored).

```bash
# TF-IDF + Logistic Regression
py -3.11 tfidf_model.py
# → spam_model.pkl, vectorizer.pkl, tfidf_metrics.json

# Bidirectional LSTM
py -3.11 lstm_model.py
# → lstm_model.keras, tokenizer.pkl, lstm_metrics.json
```

Both scripts:
- Set global random seeds for **reproducibility**
- Use **stratified** splits (LSTM also has a validation set)
- Apply **class weighting** to handle imbalance
- Print a full **classification report** and **confusion matrix**
- Export metrics to **JSON** for CI / experiment tracking
- The TF-IDF script also reports **5-fold cross-validation** and
  **top feature importances**

## Docker

```bash
docker build -t spam-detector .
docker run -p 5001:5001 spam-detector
```

The image uses a multi-stage build, runs as a non-root user, and
includes a built-in health check.

## Design Decisions

| Decision | Rationale |
| -------- | --------- |
| **Ensemble** of rule + ML + DL | Speed, accuracy, and interpretability |
| **Lazy model loading** (`@lru_cache`) | Tests and CLIs that don't call `predict()` skip TF startup |
| **Model warmup on API start** | First real request isn't penalised by cold-load latency |
| **Centralised `config.py`** | No magic numbers scattered; env-var overrides for prod |
| **Shared `utils.py`** | Single source of truth for preprocessing — zero duplication |
| **Custom exceptions** | Typed error handling; API returns 503 for model-load failures |
| **Batch endpoint** | Amortises TF/sklearn overhead over N inputs |
| **Request-ID middleware** | Every response carries `X-Request-ID` for tracing |
| **Pre-compiled regexes** | `utils.py` compiles patterns once at import time |
| **JSON metrics export** | Machine-readable output for CI pipelines |
| **Multi-stage Dockerfile** | Slim runtime image; build deps don't ship |
| **Reproducibility seeds** | Python, NumPy, TensorFlow seeded before training |

## Full-Stack Startup

From the repository root on Windows:

```powershell
.\start-dev.ps1
```

| Service | URL |
| ------- | --- |
| Model API | `http://localhost:5001` |
| Backend | `http://localhost:5000` |
| Frontend | `http://localhost:5173` |

```powershell
.\stop-model.ps1   # stop only the model service
```
