from fastapi.testclient import TestClient

from rag.api import app


def test_health_reports_chunks():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["chunks"] > 0


def test_ask_returns_structured_response():
    with TestClient(app) as client:
        response = client.post(
            "/ask",
            json={"question": "vLLM 的核心优化技术是什么？", "top_k": 3},
        )
        assert response.status_code == 200
        body = response.json()
        assert "answer" in body
        assert body["latency_ms"] >= 0
