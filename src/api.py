from contextlib import asynccontextmanager
import os
from typing import Optional, Union
from fastapi import FastAPI, HTTPException, Query
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
    text: Optional[str] = Field(None, json_schema_extra={"example": "I love this MLOps service!"})
    texts: Optional[list[str]] = Field(None, json_schema_extra={"example": ["Great service!", "Terrible delay."]})


class SinglePrediction(BaseModel):
    label: str
    confidence: float


class PredictionResponse(BaseModel):
    predictions: Union[SinglePrediction, list[SinglePrediction]]
    model_id: str
    model_revision: Optional[str]
    fallback_active: bool


class HealthResponse(BaseModel):
    status: str  # healthy | degraded
    model_id: str
    model_revision: Optional[str]
    fallback_active: bool
    fallback_reason: Optional[str] = None


def load_model_pipeline(revision: Optional[str] = None):
    """Caricamento del modello in memoria."""
    token = os.getenv("HF_TOKEN")
    hub_id = config["model"]["hub_id"]
    base_id = config["model"]["base_name"]

    # Tentativo 1: Carica Champion da HF Model Hub
    try:
        tok = AutoTokenizer.from_pretrained(hub_id, token=token, revision=revision)
        mod = AutoModelForSequenceClassification.from_pretrained(hub_id, token=token, revision=revision)
        pipe = pipeline("sentiment-analysis", model=mod, tokenizer=tok,)  # device=-1, truncation=True, max_length=128)
        return pipe, hub_id, revision or "latest", False, None
    except Exception as e_champ:
        # Tentativo 2: Fallback su modello base
        try:
            tok = AutoTokenizer.from_pretrained(base_id)
            mod = AutoModelForSequenceClassification.from_pretrained(base_id)
            pipe = pipeline("sentiment-analysis", model=mod, tokenizer=tok,)  # device=-1, truncation=True, max_length=128)
            return pipe, base_id, "base-fallback", True, str(e_champ)
        except Exception as e_base:
            # Fallimento irreversibile: esce e blocca l'avvio
            raise RuntimeError(f"Avvio fallito: impossibile caricare né Champion ({e_champ}) né Base ({e_base})")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Caricamento asincrono del modello in memoria all'avvio del container."""
    pipe, model_id, rev, fallback_act, fallback_rsn = load_model_pipeline()
    app.state.classifier = pipe
    app.state.model_id = model_id
    app.state.model_revision = rev
    app.state.fallback_active = fallback_act
    app.state.fallback_reason = fallback_rsn
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
        raise HTTPException(status_code=503, detail="Modello non pronto")

    status = "degraded" if app.state.fallback_active else "healthy"
    return HealthResponse(
        status=status,
        model_id=app.state.model_id,
        model_revision=app.state.model_revision,
        fallback_active=app.state.fallback_active,
        fallback_reason=app.state.fallback_reason,
    )


@app.post("/reload", response_model=HealthResponse)
def reload_model(revision: Optional[str] = Query(None, description="Commit SHA o revisione da caricare")):
    try:
        pipe, model_id, rev, fallback_act, fallback_rsn = load_model_pipeline(revision=revision)
        app.state.classifier = pipe
        app.state.model_id = model_id
        app.state.model_revision = rev
        app.state.fallback_active = fallback_act
        app.state.fallback_reason = fallback_rsn
        return health_check()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Hot-reload fallito: {str(e)}")


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
        raise HTTPException(status_code=422, detail="Specificare 'text' o 'texts'")

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
            model_revision=app.state.model_revision,
            fallback_active=app.state.fallback_active,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore durante l'inferenza: {str(e)}")
