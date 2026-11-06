import os
import pytest
import requests


def test_live_container_endpoints():
    api_url = os.environ.get("TEST_API_URL", "http://localhost:8000")
    try:
        health_res = requests.get(f"{api_url}/health", timeout=10)
        assert health_res.status_code == 200
        data = health_res.json()
        assert data["status"] in ["healthy", "degraded"]
        assert "model_revision" in data

        pred_res = requests.post(f"{api_url}/predict", json={"text": "Live test"}, timeout=10)
        assert pred_res.status_code == 200
    except requests.exceptions.ConnectionError:
        pytest.skip(f"API Live non raggiungibile su {api_url}")
