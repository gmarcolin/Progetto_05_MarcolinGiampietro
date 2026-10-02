from fastapi.testclient import TestClient
from src.api import app


def test_api_health_endpoint():
    with TestClient(app) as client:
        res = client.get("/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "healthy"
        assert "model_id" in data


def test_api_predict_endpoint():
    with TestClient(app) as client:
        payload = {"text": "I really love this fast and clean MLOps pipeline!"}
        res = client.post("/predict", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["label"] in ["positive", "neutral", "negative"]
        assert 0.0 <= data["confidence"] <= 1.0
        assert data["text"] == payload["text"]
