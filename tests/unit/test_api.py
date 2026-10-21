from fastapi.testclient import TestClient
from src.api import app


def test_health_endpoint():
    with TestClient(app) as client:
        res = client.get("/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] in ["healthy", "degraded"]
        assert "model_id" in data
        assert isinstance(data["fallback_active"], bool)


def test_predict_single_text():
    with TestClient(app) as client:
        payload = {"text": "I love this product!"}
        res = client.post("/predict", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert "predictions" in data
        assert data["predictions"]["label"] in ["positive", "neutral", "negative"]
        assert 0.0 <= data["predictions"]["confidence"] <= 1.0


def test_predict_batch_texts():
    with TestClient(app) as client:
        payload = {"texts": ["Great movie", "Worst food ever"]}
        res = client.post("/predict", json=payload)
        assert res.status_code == 200
        data = res.json()
        preds = data["predictions"]
        assert isinstance(preds, list)
        assert len(preds) == 2
