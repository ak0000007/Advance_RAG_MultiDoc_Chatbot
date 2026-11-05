import asyncio
import pytest
import httpx
from unittest.mock import patch, MagicMock

from src.clients.salesforce_client import (
    SalesforceAsyncClient,
    _DEFAULT_LIMITS,
    _TIMEOUT,
)


@pytest.fixture
def mock_sf_client(tmp_path):
    key_file = tmp_path / "test_key.pem"
    key_file.write_text("fake_key_content")
    with patch.object(SalesforceAsyncClient, "_load_private_key", return_value="fake_pem_key"):
        client = SalesforceAsyncClient(
            client_id="test_client_id",
            login_url="https://login.salesforce.com",
            private_key_path=str(key_file),
            domain="https://test.my.salesforce.com",
        )
        yield client
        # Clean teardown
        if client._client and not client._client.is_closed:
            asyncio.run(client.aclose())


@pytest.mark.asyncio
async def test_persistent_client_reuse(mock_sf_client):
    """Verify that multiple accesses return the exact same httpx.AsyncClient instance."""
    client_instance_1 = mock_sf_client.http_client
    client_instance_2 = mock_sf_client.http_client

    assert client_instance_1 is client_instance_2
    assert not client_instance_1.is_closed
    assert isinstance(client_instance_1, httpx.AsyncClient)
    assert mock_sf_client._owns_client is True


@pytest.mark.asyncio
async def test_connection_limits_and_timeout(mock_sf_client):
    """Verify that connection pool limits are properly configured."""
    client = mock_sf_client.http_client
    assert _DEFAULT_LIMITS.max_keepalive_connections == 20
    assert _DEFAULT_LIMITS.max_connections == 50
    assert _DEFAULT_LIMITS.keepalive_expiry == 30.0
    assert client.timeout.connect == _TIMEOUT.connect


@pytest.mark.asyncio
async def test_aclose_lifecycle(mock_sf_client):
    """Verify that aclose() cleanly closes the owned httpx.AsyncClient."""
    http_client = mock_sf_client.http_client
    assert not http_client.is_closed

    await mock_sf_client.aclose()
    assert http_client.is_closed


@pytest.mark.asyncio
async def test_auto_recovery_if_client_closed(mock_sf_client):
    """Verify that accessing http_client after a close creates a new active client."""
    first_client = mock_sf_client.http_client
    await mock_sf_client.aclose()
    assert first_client.is_closed

    # Second access should safely allocate a fresh active client
    second_client = mock_sf_client.http_client
    assert second_client is not first_client
    assert not second_client.is_closed
    await mock_sf_client.aclose()


@pytest.mark.asyncio
async def test_injected_external_client(tmp_path):
    """Verify that externally injected client is not closed on aclose() (ownership principle)."""
    key_file = tmp_path / "test_key.pem"
    key_file.write_text("fake_key_content")
    
    external_http_client = httpx.AsyncClient()
    try:
        with patch.object(SalesforceAsyncClient, "_load_private_key", return_value="fake_pem_key"):
            sf = SalesforceAsyncClient(
                client_id="id",
                login_url="https://login.salesforce.com",
                private_key_path=str(key_file),
                domain="https://test.my.salesforce.com",
                http_client=external_http_client,
            )
            assert sf._owns_client is False
            assert sf.http_client is external_http_client

            # Closing sf should NOT close caller's external client
            await sf.aclose()
            assert not external_http_client.is_closed
    finally:
        await external_http_client.aclose()


@pytest.mark.asyncio
async def test_async_context_manager(tmp_path):
    """Verify async context manager protocol enters and exits cleanly."""
    key_file = tmp_path / "test_key.pem"
    key_file.write_text("fake_key_content")

    with patch.object(SalesforceAsyncClient, "_load_private_key", return_value="fake_pem_key"):
        async with SalesforceAsyncClient(
            client_id="id",
            login_url="https://login.salesforce.com",
            private_key_path=str(key_file),
            domain="https://test.my.salesforce.com",
        ) as sf:
            client = sf.http_client
            assert not client.is_closed

        assert client.is_closed


@pytest.mark.asyncio
async def test_pooled_requests_use_shared_client(mock_sf_client):
    """Verify that API requests pass through the persistent pooled client."""
    mock_response = httpx.Response(
        status_code=200,
        json={"isSuccess": True, "data": [{"Id": "a01xx000001", "Name": "BK-100"}]},
        request=httpx.Request("GET", "https://test.my.salesforce.com"),
    )

    with patch.object(mock_sf_client, "get_access_token", return_value="fake_access_token"):
        with patch.object(mock_sf_client.http_client, "get", return_value=mock_response) as mock_get:
            res = await mock_sf_client.get_booking(username="agent@travel.com", booking_id_or_name="a01xx0000001AAAAAA")
            assert res["isSuccess"] is True
            mock_get.assert_called_once()
            assert "Authorization" in mock_get.call_args.kwargs["headers"]
            assert mock_get.call_args.kwargs["headers"]["Authorization"] == "Bearer fake_access_token"
