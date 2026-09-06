from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

import backend.modules.ai.providers as providers
from backend.modules.ai.providers import (
    OpenAIProvider,
    ProviderGenerateRequest,
    close_ai_provider_http_clients,
)


@pytest.fixture(autouse=True)
def reset_http_clients() -> None:
    providers._openai_http_client = None
    providers._anthropic_http_client = None
    yield
    providers._openai_http_client = None
    providers._anthropic_http_client = None


def test_openai_provider_reuses_shared_http_client(monkeypatch) -> None:
    created_clients: list[MagicMock] = []

    class FakeAsyncClient:
        is_closed = False

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            created_clients.append(self)

        async def post(self, *args, **kwargs):
            response = MagicMock()
            response.status_code = 200
            response.json.return_value = {
                "data": [{"embedding": [0.1, 0.2]}],
                "choices": [{"message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            }
            return response

    monkeypatch.setattr(providers.httpx, "AsyncClient", FakeAsyncClient)
    monkeypatch.setattr(providers.settings, "OPENAI_API_KEY", "test-key")

    async def _run() -> None:
        provider = OpenAIProvider()
        await provider.embed_texts(["one", "two"])
        await provider.generate(
            ProviderGenerateRequest(
                model="gpt-test",
                system_prompt="system",
                user_prompt="user",
                response_format="text",
                temperature=0.2,
            )
        )

    asyncio.run(_run())
    assert len(created_clients) == 1


def test_post_with_retry_retries_transient_status_codes(monkeypatch) -> None:
    responses = [
        MagicMock(status_code=429, text="rate limited"),
        MagicMock(status_code=200, text="ok"),
    ]
    client = MagicMock()
    client.post = AsyncMock(side_effect=responses)
    sleep_mock = AsyncMock()
    monkeypatch.setattr(providers.asyncio, "sleep", sleep_mock)

    result = asyncio.run(providers._post_with_retry(client, "/chat/completions", json={}))

    assert result.status_code == 200
    assert client.post.await_count == 2
    sleep_mock.assert_awaited_once_with(1)


def test_close_ai_provider_http_clients_closes_open_clients() -> None:
    open_client = MagicMock()
    open_client.is_closed = False
    open_client.aclose = AsyncMock()
    providers._openai_http_client = open_client

    asyncio.run(close_ai_provider_http_clients())

    open_client.aclose.assert_awaited_once()
    assert providers._openai_http_client is None
