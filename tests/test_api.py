"""API smoke tests: health, CORS, auth, and error hiding.

These run without Supabase, the ML models or an LLM key. Database access is
replaced with a stub, so the tests only check how the API behaves around it.
"""
import pytest
from fastapi.testclient import TestClient

import agents.memory
from api.dependencies import get_current_user
from api.main import app

ALLOWED = "http://localhost:3000"
FAKE_USER = {"patient_id": "test-patient", "email": "test@example.com"}


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_root_points_to_docs(client):
    assert client.get("/").json()["docs"] == "/docs"


def test_cors_allows_configured_frontend(client):
    resp = client.options(
        "/health",
        headers={"Origin": ALLOWED, "Access-Control-Request-Method": "GET"},
    )
    assert resp.headers.get("access-control-allow-origin") == ALLOWED


def test_cors_rejects_unknown_origin(client):
    resp = client.options(
        "/health",
        headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "GET"},
    )
    assert "access-control-allow-origin" not in resp.headers


@pytest.mark.parametrize("path", ["/alerts", "/alerts/all", "/patients/me"])
def test_protected_routes_need_a_token(client, path):
    resp = client.get(path)
    assert resp.status_code == 401


def test_internal_errors_are_not_leaked(client, monkeypatch):
    secret = "connection to db at 10.0.0.5 failed: password=hunter2"

    def broken_memory():
        raise RuntimeError(secret)

    monkeypatch.setattr(agents.memory, "get_memory", broken_memory)
    app.dependency_overrides[get_current_user] = lambda: FAKE_USER

    resp = client.get("/alerts")
    assert resp.status_code == 500
    assert resp.json()["detail"] == "Could not load alerts"
    assert secret not in resp.text
