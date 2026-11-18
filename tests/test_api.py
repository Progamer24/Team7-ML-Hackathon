import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_api_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "PathSnap-API"


def test_api_list_trips():
    response = client.get("/trips")
    assert response.status_code == 200
    data = response.json()
    assert "available_trips" in data
    assert len(data["available_trips"]) == 10


def test_api_predict():
    response = client.post("/predict", json={"trip_id": "00000000"})
    assert response.status_code == 200
    data = response.json()
    assert data["trip_id"] == "00000000"
    assert data["status"] == "success"
    assert len(data["predicted_arcs"]) == data["num_observations"]
    assert 0.0 <= data["route_continuity"] <= 1.0


def test_api_evaluate():
    response = client.post("/evaluate", json={"trip_id": "00000000"})
    assert response.status_code == 200
    data = response.json()
    assert data["trip_id"] == "00000000"
    assert 0.0 <= data["observation_top1_accuracy"] <= 1.0
    assert 0.0 <= data["route_continuity"] <= 1.0
