import pytest
from fastapi.testclient import TestClient
from src.api.app import app
from src.config import Settings


def test_cors_preflight_options():
    """
    Verify browser preflight OPTIONS request returns 200 and required CORS headers.
    """
    client = TestClient(app)
    headers = {
        "Origin": "http://localhost:3000",
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "authorization,content-type",
    }
    response = client.options("/chat", headers=headers)
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "http://localhost:3000"
    assert "POST" in response.headers.get("access-control-allow-methods", "")
    assert "authorization" in response.headers.get("access-control-allow-headers", "").lower()


def test_cors_headers_on_standard_request():
    """
    Verify standard requests with Origin header receive Access-Control-Allow-Origin response header.
    """
    client = TestClient(app)
    response = client.get("/health", headers={"Origin": "http://localhost:5173"})
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "http://localhost:5173"


def test_cors_origins_pydantic_parsing():
    """
    Verify Settings parses comma-separated string, JSON list, or defaults.
    """
    # Default
    s1 = Settings()
    assert s1.cors_origins == ["*"]

    # Comma-separated string
    s2 = Settings(cors_origins="http://localhost:3000, https://internal.travel.corp")
    assert s2.cors_origins == ["http://localhost:3000", "https://internal.travel.corp"]

    # JSON list string
    s3 = Settings(cors_origins='["https://app1.com", "https://app2.com"]')
    assert s3.cors_origins == ["https://app1.com", "https://app2.com"]
