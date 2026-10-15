
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


def test_prediction_schema():
    response = client.post(
        "/predict",
        json={"features": [0.0] * 30}
    )
    assert response.status_code == 200

    data = response.json()
    assert isinstance(data["prediction"], int)
    assert isinstance(data["probabilities"], list)
    assert "model_name" in data


def test_invalid_feature_count():
    response = client.post(
        "/predict",
        json={"features": [0.0] * 5}
    )
    assert response.status_code == 422
