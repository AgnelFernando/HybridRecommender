import json, os
from fastapi.testclient import TestClient
from src.serve.api import app

client = TestClient(app)

def test_health():
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert data["items"] > 0

def test_recommendations():
    res = client.post("/recommendations", json={"user_id": 999999, "topk": 5})
    assert res.status_code == 200
    data = res.json()
    assert "items" in data
    assert len(data["items"]) <= 5
