import time
import uuid
from decimal import Decimal
from datetime import datetime, timezone

from fastapi import FastAPI
from pydantic import BaseModel

from api.model_loader import load_production_model
from api.database import log_prediction, update_feedback

app = FastAPI(
    title="Toxic Comment Moderation API",
    version="1.0.0",
)

model = load_production_model()

LABEL_COLUMNS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]


class CommentRequest(BaseModel):
    text: str

class FeedbackRequest(BaseModel):
    request_id: str
    is_correct: bool

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict")
def predict(request: CommentRequest):
    start_time = time.perf_counter()

    prediction = model.predict([request.text])[0]

    latency_ms = (time.perf_counter() - start_time) * 1000

    result = {
        label: int(value)
        for label, value in zip(LABEL_COLUMNS, prediction)
    }

    request_id = str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc).isoformat()

    prediction_record = {
        "request_id": request_id,
        "timestamp": timestamp,
        "text": request.text,
        "prediction": result,
        "latency_ms": Decimal(str(round(latency_ms, 2))),
    }

    log_prediction(prediction_record)

    return {
        "request_id": request_id,
        "text": request.text,
        "prediction": result,
        "latency_ms": round(latency_ms, 2),
    }

@app.post("/feedback")
def submit_feedback(request: FeedbackRequest):
    update_feedback(
        request_id=request.request_id,
        is_correct=request.is_correct,
    )

    return {
        "status": "feedback recorded",
        "request_id": request.request_id,
        "is_correct": request.is_correct,
    }

