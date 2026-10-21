from contextlib import asynccontextmanager
import os
from typing import Optional, Union
from fastapi import FastAPI, HTTPException
from prometheus_client import Counter
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import yaml

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Metrica personalizzata Prometheus per Prediction Drift
PREDICTION_COUNTER = Counter(
    "sentiment_predictions_total",
    "Totale predizioni di sentiment suddivise per classe",
    ["label"],
)


# Strutture Dati Pydantic per I/O validato
class PredictionRequest(BaseModel):
    text: Optional[str] = Field(None, example="I love this MLOps service!")
    texts: Optional[list[str]] = Field(None, example=["Great service!", "Terrible delay."])


class SinglePrediction(BaseModel):
    label: str
    confidence: float


class PredictionResponse(BaseModel):
    predictions: Union[SinglePrediction, list[SinglePrediction]]
    model_id: str
    fallback_active: bool


class HealthResponse(BaseModel):
    status: str  # healthy | degraded
    model_id: str
    fallback_active: bool
    fallback_reason: Optional[str] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Caricamento asincrono del modello in memoria all'avvio del container."""
    token = os.getenv("HF_TOKEN")
    hub_id = config["model"]["hub_id"]
    base_id = config["model"]["base_name"]

    app.state.classifier = None
    app.state.model_id = None
    app.state.fallback_active = False
    app.state.fallback_reason = None

    # Tentativo 1: Carica Champion da HF Model Hub
    try:
        app.state.fallback_active = False
        tokenizer = AutoTokenizer.from_pretrained(hub_id, token=token)
        model = AutoModelForSequenceClassification.from_pretrained(hub_id, token=token)
        app.state.classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
        # app.state.classifier = pipeline(
        # "sentiment-analysis", model=model, tokenizer=tokenizer, 
        # device=-1, truncation=True, max_length=128
        # )
        app.state.model_id = hub_id
    except Exception as e_champion:
        app.state.fallback_active = True
        app.state.fallback_reason = str(e_champion)
        # Tentativo 2: Fallback su modello base
        try:
            tokenizer = AutoTokenizer.from_pretrained(base_id)
            model = AutoModelForSequenceClassification.from_pretrained(base_id)
            app.state.classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
            # app.state.classifier = pipeline(
            # "sentiment-analysis", model=model, tokenizer=tokenizer, 
            # device=-1, truncation=True, max_length=128
            # )
            app.state.model_id = base_id
        except Exception as e_base:
            # Fallimento irreversibile: esce e blocca l'avvio
            raise RuntimeError(
                f"Avvio fallito: impossibile caricare né Champion ({e_champ}) né Base ({e_base})"
            )

    yield
    app.state.classifier = None


app = FastAPI(
    title="Sentiment Analysis API",
    description="REST API per inferenza e monitoraggio del modello di Sentiment Analysis",
    version="2.0.0",
    lifespan=lifespan,
)

# Espone l'endpoint standard /metrics per Prometheus
Instrumentator().instrument(app).expose(app)


@app.get("/health", response_model=HealthResponse)
def health_check():
    """Endpoint di stato per Kubernetes, Docker e Load Balancer."""
    if not hasattr(app.state, "classifier") or app.state.classifier is None:
        raise HTTPException(status_code=503, detail="Modello non inizializzato")

    current_status = "degraded" if app.state.fallback_active else "healthy"
    return HealthResponse(
        status=current_status,
        model_id=app.state.model_id,
        fallback_active=app.state.fallback_active,
        fallback_reason=app.state.fallback_reason,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict_sentiment(payload: PredictionRequest):
    """Calcola il sentiment sul testo fornito."""
    if not hasattr(app.state, "classifier") or app.state.classifier is None:
        raise HTTPException(status_code=503, detail="Modello non pronto per l'inferenza")

    is_single = False
    if payload.text is not None:
        texts = [payload.text]
        is_single = True
    elif payload.texts is not None and len(payload.texts) > 0:
        texts = payload.texts
    else:
        raise HTTPException(status_code=422, detail="Fornire 'text' oppure 'texts'")

    try:
        raw_preds = app.state.classifier(texts, batch_size=16)
        results = []
        for p in raw_preds:
            lbl = p["label"].lower()
            if lbl.startswith("label_"):
                idx = int(lbl.split("_")[-1])
                clean_lbl = ["negative", "neutral", "positive"][idx]
            else:
                clean_lbl = lbl

            # Incremento metrica Prometheus per ciascuna predizione
            PREDICTION_COUNTER.labels(label=clean_lbl).inc()

            results.append(SinglePrediction(label=clean_lbl, confidence=round(float(p["score"]), 4)))

        response_content = results[0] if is_single else results
        return PredictionResponse(
            predictions=response_content,
            model_id=app.state.model_id,
            fallback_active=app.state.fallback_active,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore durante l'inferenza: {str(e)}")
