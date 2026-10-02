from contextlib import asynccontextmanager
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import yaml

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Strutture Dati Pydantic per I/O validato
class PredictionRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=2,
        json_schema_extra={
            "example": "I love this sentiment analysis API!"
        }
    )

class PredictionResponse(BaseModel):
    text: str
    label: str
    confidence: float
    model_id: str

class HealthResponse(BaseModel):
    status: str
    model_id: str
    labels_supported: list[str]


classifier_pipeline = None
active_model_id = ""


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Caricamento asincrono del modello in memoria all'avvio del container."""
    global classifier_pipeline, active_model_id
    token = os.getenv("HF_TOKEN")
    hub_id = config["model"]["hub_id"]
    base_id = config["model"]["base_name"]

    try:
        tokenizer = AutoTokenizer.from_pretrained(hub_id, token=token)
        model = AutoModelForSequenceClassification.from_pretrained(hub_id, token=token)
        active_model_id = hub_id
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(base_id)
        model = AutoModelForSequenceClassification.from_pretrained(base_id)
        active_model_id = base_id

    classifier_pipeline = pipeline(
        "sentiment-analysis",
        model=model,
        tokenizer=tokenizer,
        device=-1,  # CPU serving
        truncation=True,
        max_length=128,
    )
    yield
    classifier_pipeline = None


app = FastAPI(
    title="Twitter Sentiment Analysis Microservice",
    description="Serving REST API per inferenza online del modello di Sentiment Analysis",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health_check():
    """Endpoint di stato per Kubernetes, Docker e Load Balancer."""
    if classifier_pipeline is None:
        raise HTTPException(status_code=503, detail="Modello non ancora inizializzato")
    return HealthResponse(
        status="healthy",
        model_id=active_model_id,
        labels_supported=["negative", "neutral", "positive"],
    )


@app.post("/predict", response_model=PredictionResponse)
def predict_sentiment(payload: PredictionRequest):
    """Calcola il sentiment sul testo fornito."""
    if classifier_pipeline is None:
        raise HTTPException(status_code=503, detail="Modello non pronto per l'inferenza")

    try:
        raw_pred = classifier_pipeline(payload.text)[0]
        mapping = {"negative": "negative", "neutral": "neutral", "positive": "positive"}
        lbl = raw_pred["label"].lower()
        if lbl.startswith("label_"):
            idx = int(lbl.split("_")[-1])
            clean_label = ["negative", "neutral", "positive"][idx]
        else:
            clean_label = mapping.get(lbl, lbl)

        return PredictionResponse(
            text=payload.text,
            label=clean_label,
            confidence=round(float(raw_pred["score"]), 4),
            model_id=active_model_id,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore durante l'inferenza: {str(e)}")
