"""Async HTTP client for the llama.cpp server OpenAI-compatible API."""

from __future__ import annotations

import json
from typing import AsyncIterator, Any

import httpx

from llmmanager.constants import HTTP_CONNECT_TIMEOUT, HTTP_READ_TIMEOUT, HTTP_QUICK_INFER_TIMEOUT
from llmmanager.exceptions import ServerNotRunningError


class LlamaCppAPIClient:
    def __init__(self, host: str, port: int) -> None:
        self._base = f"http://{host}:{port}"
        self._client = httpx.AsyncClient(
            base_url=self._base,
            timeout=httpx.Timeout(
                connect=HTTP_CONNECT_TIMEOUT,
                read=HTTP_READ_TIMEOUT,
                write=30.0,
                pool=5.0,
            ),
        )

    async def health(self) -> bool:
        try:
            r = await self._client.get("/health")
            if r.status_code == 200:
                data = r.json()
                # llama-server returns {"status": "ok"} or {"status": "loading model"}
                return data.get("status") in ("ok", "no slot available")
            return False
        except httpx.ConnectError:
            return False

    async def list_models(self) -> list[dict[str, Any]]:
        try:
            r = await self._client.get("/v1/models")
            r.raise_for_status()
            return r.json().get("data", [])
        except httpx.ConnectError as exc:
            raise ServerNotRunningError("llama.cpp server is not reachable") from exc

    async def chat_stream(
        self,
        model: str,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
            **kwargs,
        }
        async with self._client.stream(
            "POST",
            "/v1/chat/completions",
            json=payload,
            timeout=httpx.Timeout(
                connect=HTTP_CONNECT_TIMEOUT,
                read=HTTP_QUICK_INFER_TIMEOUT,
                write=30.0,
                pool=5.0,
            ),
        ) as resp:
            if resp.status_code >= 400:
                body = await resp.aread()
                try:
                    err_msg = json.loads(body).get("error", {})
                    if isinstance(err_msg, dict):
                        err_msg = err_msg.get("message", body.decode())
                except Exception:
                    err_msg = body.decode() or f"HTTP {resp.status_code}"
                raise httpx.HTTPStatusError(
                    f"llama.cpp error {resp.status_code}: {err_msg}",
                    request=resp.request,
                    response=resp,
                )
            async for line in resp.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        content = (
                            chunk.get("choices", [{}])[0]
                            .get("delta", {})
                            .get("content", "")
                        )
                        if content:
                            yield content
                    except json.JSONDecodeError:
                        continue

    async def close(self) -> None:
        await self._client.aclose()
