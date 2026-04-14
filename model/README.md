## Model Service

This folder contains the trained spam detector and its inference API.

### Run the model API

1. Install dependencies:

	```bash
	py -3.11 -m pip install -r requirements.txt
	```

2. Start the service from this folder:

	```bash
	py -3.11 -m uvicorn api:app --host 0.0.0.0 --port 5001
	```

3. Health check:

	```bash
	GET http://localhost:5001/health
	```

### API

`POST /predict`

Request body:

```json
{ "content": "free money now" }
```

Response shape:

```json
{
  "label": "SPAM",
  "score": 0.931,
  "rule_conf": 0.4,
  "tfidf_prob": 0.912,
  "lstm_prob": 0.945
}
```

## Full Stack Startup

From the repository root on Windows, run:

```powershell
.\start-dev.ps1
```

This opens three shells:

1. Model API on `http://localhost:5001`
2. Backend on `http://localhost:5000`
3. Frontend on `http://localhost:5173`

Backend health check:

```bash
GET http://localhost:5000/health
```

To stop only the model service from the repository root:

```powershell
.\stop-model.ps1
```
