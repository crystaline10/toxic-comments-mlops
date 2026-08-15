from fastapi import FastAPI
from pydantic import BaseModel

from api.model_loader import load_production_model


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


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict")
def predict(request: CommentRequest):
    prediction = model.predict([request.text])[0]

    result = {
        label: int(value)
        for label, value in zip(LABEL_COLUMNS, prediction)
    }

    return {
        "text": request.text,
        "prediction": result,
    }