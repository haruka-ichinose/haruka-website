import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api import chat
from core.config import settings
from core.guardrails import RateLimiter


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(settings, "rag_mode", "local")
    monkeypatch.setattr(settings, "rag_proxy_token", "secret")
    monkeypatch.setattr(chat, "_limiter", RateLimiter(per_minute=1, per_day=10, global_per_day=10))
    app = FastAPI()
    app.include_router(chat.router)
    return TestClient(app, raise_server_exceptions=False)


def test_missing_or_wrong_token_is_rejected(client):
    assert client.post("/chat", json={"message": "hi"}).status_code == 401
    bad = {"Authorization": "Bearer nope"}
    assert client.post("/chat", json={"message": "hi"}, headers=bad).status_code == 401


def test_rate_limit_keys_on_forwarded_ip(client):
    auth = {"Authorization": "Bearer secret"}
    # First request passes the limiter (then errors later: no RAG clients here).
    first = client.post("/chat", json={"message": "hi"}, headers={**auth, "x-client-ip": "1.1.1.1"})
    assert first.status_code != 429
    again = client.post("/chat", json={"message": "hi"}, headers={**auth, "x-client-ip": "1.1.1.1"})
    assert again.status_code == 429
    other = client.post("/chat", json={"message": "hi"}, headers={**auth, "x-client-ip": "2.2.2.2"})
    assert other.status_code != 429


def test_oversized_message_is_rejected(client):
    auth = {"Authorization": "Bearer secret"}
    assert client.post("/chat", json={"message": "x" * 5000}, headers=auth).status_code == 422
