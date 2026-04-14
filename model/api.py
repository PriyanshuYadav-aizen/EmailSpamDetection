from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from predict_main import predict


app = FastAPI(title="Spam Model API", version="1.0.0")


class PredictRequest(BaseModel):
    content: str = Field(min_length=1)


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict")
def predict_email(payload: PredictRequest):
    content = payload.content.strip()
    if not content:
        raise HTTPException(status_code=400, detail="content is required")

    result = predict(content)
    return result
