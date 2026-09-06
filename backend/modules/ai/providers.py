from __future__ import annotations

import asyncio
import hashlib
import json
import math
from dataclasses import dataclass

import httpx
from fastapi import HTTPException

from backend.core.config import settings
from backend.lib.vectors import estimate_tokens

_openai_http_client: httpx.AsyncClient | None = None
_anthropic_http_client: httpx.AsyncClient | None = None
_RETRYABLE_STATUS_CODES = frozenset({429, 502, 503, 504})
_MAX_HTTP_RETRIES = 3


def _get_openai_http_client() -> httpx.AsyncClient:
    global _openai_http_client
    if _openai_http_client is None or _openai_http_client.is_closed:
        _openai_http_client = httpx.AsyncClient(
            timeout=60.0,
            base_url=settings.OPENAI_BASE_URL,
        )
    return _openai_http_client


def _get_anthropic_http_client() -> httpx.AsyncClient:
    global _anthropic_http_client
    if _anthropic_http_client is None or _anthropic_http_client.is_closed:
        _anthropic_http_client = httpx.AsyncClient(
            timeout=60.0,
            base_url=settings.ANTHROPIC_BASE_URL,
        )
    return _anthropic_http_client


async def close_ai_provider_http_clients() -> None:
    global _openai_http_client, _anthropic_http_client
    for client in (_openai_http_client, _anthropic_http_client):
        if client is not None and not client.is_closed:
            await client.aclose()
    _openai_http_client = None
    _anthropic_http_client = None


async def _post_with_retry(client: httpx.AsyncClient, url: str, **kwargs) -> httpx.Response:
    last_response: httpx.Response | None = None
    for attempt in range(_MAX_HTTP_RETRIES):
        response = await client.post(url, **kwargs)
        last_response = response
        if response.status_code not in _RETRYABLE_STATUS_CODES or attempt == _MAX_HTTP_RETRIES - 1:
            return response
        await asyncio.sleep(min(2**attempt, 8))
    assert last_response is not None
    return last_response


@dataclass(slots=True)
class ProviderGenerateRequest:
    model: str
    system_prompt: str
    user_prompt: str
    response_format: str
    temperature: float


@dataclass(slots=True)
class ProviderGenerateResult:
    provider_key: str
    model: str
    output_text: str
    output_json: dict | None
    input_tokens: int
    output_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class BaseAiProvider:
    key = "base"

    async def generate(self, request: ProviderGenerateRequest) -> ProviderGenerateResult:
        raise NotImplementedError

    async def embed_texts(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        raise NotImplementedError


def _hash_embedding(text: str, dimensions: int = 32) -> list[float]:
    values: list[float] = []
    for index in range(dimensions):
        digest = hashlib.sha256(f"{index}:{text}".encode()).digest()
        integer = int.from_bytes(digest[:8], "big")
        values.append(((integer % 2000) / 1000.0) - 1.0)
    norm = math.sqrt(sum(value * value for value in values)) or 1.0
    return [value / norm for value in values]


class LocalHeuristicProvider(BaseAiProvider):
    key = "local"

    async def generate(self, request: ProviderGenerateRequest) -> ProviderGenerateResult:
        prompt = request.user_prompt.strip()
        system = request.system_prompt.strip()
        summary = prompt[:1500]
        if request.response_format == "json":
            payload = {
                "provider": self.key,
                "model": request.model,
                "summary": summary,
                "system_context": system[:400],
            }
            output_text = json.dumps(payload, indent=2)
            output_json = payload
        else:
            output_text = (
                f"[{request.model}] Heuristic response\n\n"
                f"System context:\n{system[:400] or 'None'}\n\n"
                f"User prompt:\n{summary}"
            )
            output_json = None
        return ProviderGenerateResult(
            provider_key=self.key,
            model=request.model,
            output_text=output_text,
            output_json=output_json,
            input_tokens=estimate_tokens(f"{system}\n{prompt}"),
            output_tokens=estimate_tokens(output_text),
        )

    async def embed_texts(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        return [
            _hash_embedding(text, dimensions=settings.RAG_EMBEDDING_DIMENSIONS) for text in texts
        ]


class OpenAIProvider(BaseAiProvider):
    key = "openai"

    async def generate(self, request: ProviderGenerateRequest) -> ProviderGenerateResult:
        if not settings.OPENAI_API_KEY:
            raise HTTPException(status_code=422, detail="OPENAI_API_KEY is not configured")
        client = _get_openai_http_client()
        response = await _post_with_retry(
            client,
            "/chat/completions",
            headers={"Authorization": f"Bearer {settings.OPENAI_API_KEY}"},
            json={
                "model": request.model,
                "temperature": request.temperature,
                "response_format": {"type": "json_object"}
                if request.response_format == "json"
                else {"type": "text"},
                "messages": [
                    {"role": "system", "content": request.system_prompt},
                    {"role": "user", "content": request.user_prompt},
                ],
            },
        )
        if response.status_code >= 400:
            raise HTTPException(
                status_code=502,
                detail=f"OpenAI request failed: {response.text[:300]}",
            )
        payload = response.json()
        output_text = payload["choices"][0]["message"]["content"]
        output_json = None
        if request.response_format == "json":
            try:
                output_json = json.loads(output_text)
            except json.JSONDecodeError:
                output_json = {"raw": output_text}
        usage = payload.get("usage", {})
        return ProviderGenerateResult(
            provider_key=self.key,
            model=request.model,
            output_text=output_text,
            output_json=output_json,
            input_tokens=int(usage.get("prompt_tokens", 0)),
            output_tokens=int(usage.get("completion_tokens", 0)),
        )

    async def embed_texts(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        if not settings.OPENAI_API_KEY:
            raise HTTPException(status_code=422, detail="OPENAI_API_KEY is not configured")
        client = _get_openai_http_client()
        response = await _post_with_retry(
            client,
            "/embeddings",
            headers={"Authorization": f"Bearer {settings.OPENAI_API_KEY}"},
            json={
                "model": model or settings.OPENAI_EMBEDDING_MODEL,
                "input": texts,
            },
        )
        if response.status_code >= 400:
            raise HTTPException(
                status_code=502,
                detail=f"OpenAI embeddings request failed: {response.text[:300]}",
            )
        payload = response.json()
        return [item["embedding"] for item in payload.get("data", [])]


class AnthropicProvider(BaseAiProvider):
    key = "anthropic"

    async def generate(self, request: ProviderGenerateRequest) -> ProviderGenerateResult:
        if not settings.ANTHROPIC_API_KEY:
            raise HTTPException(status_code=422, detail="ANTHROPIC_API_KEY is not configured")
        client = _get_anthropic_http_client()
        response = await _post_with_retry(
            client,
            "/messages",
            headers={
                "x-api-key": settings.ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
            },
            json={
                "model": request.model,
                "system": request.system_prompt,
                "temperature": request.temperature,
                "max_tokens": settings.AI_MAX_OUTPUT_TOKENS,
                "messages": [{"role": "user", "content": request.user_prompt}],
            },
        )
        if response.status_code >= 400:
            raise HTTPException(
                status_code=502,
                detail=f"Anthropic request failed: {response.text[:300]}",
            )
        payload = response.json()
        content_blocks = payload.get("content", [])
        output_text = "\n".join(
            block.get("text", "") for block in content_blocks if block.get("type") == "text"
        )
        output_json = None
        if request.response_format == "json":
            try:
                output_json = json.loads(output_text)
            except json.JSONDecodeError:
                output_json = {"raw": output_text}
        usage = payload.get("usage", {})
        return ProviderGenerateResult(
            provider_key=self.key,
            model=request.model,
            output_text=output_text,
            output_json=output_json,
            input_tokens=int(usage.get("input_tokens", 0)),
            output_tokens=int(usage.get("output_tokens", 0)),
        )

    async def embed_texts(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        if settings.AI_EMBEDDING_PROVIDER == "local":
            return await LocalHeuristicProvider().embed_texts(texts)
        raise HTTPException(
            status_code=422,
            detail="Anthropic embeddings are not configured. Use the local embedding provider.",
        )


class AiProviderRegistry:
    def __init__(self):
        self._providers = {
            provider.key: provider
            for provider in (
                LocalHeuristicProvider(),
                OpenAIProvider(),
                AnthropicProvider(),
            )
        }

    def get(self, key: str | None) -> BaseAiProvider:
        provider_key = (key or settings.AI_DEFAULT_PROVIDER).strip().lower()
        provider = self._providers.get(provider_key)
        if not provider:
            raise HTTPException(status_code=404, detail=f"Unknown AI provider: {provider_key}")
        return provider

    def embedding_provider_and_model(self) -> tuple[BaseAiProvider, str]:
        provider = self.get(settings.AI_EMBEDDING_PROVIDER)
        model = (
            settings.OPENAI_EMBEDDING_MODEL
            if provider.key == "openai"
            else settings.AI_LOCAL_MODEL_NAME
        )
        return provider, model

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        provider, model = self.embedding_provider_and_model()
        return await provider.embed_texts(texts, model=model)

    async def embed_retrieval_queries(self, texts: list[str]) -> list[list[float]]:
        from backend.lib.embedding_cache import embed_texts_with_cache

        provider, model = self.embedding_provider_and_model()
        return await embed_texts_with_cache(
            provider=provider.key,
            model=model,
            texts=texts,
            embed_fn=lambda miss: provider.embed_texts(miss, model=model),
            dimensions=settings.RAG_EMBEDDING_DIMENSIONS,
            ttl_seconds=settings.CACHE_QUERY_EMBEDDING_TTL_SECONDS,
        )
