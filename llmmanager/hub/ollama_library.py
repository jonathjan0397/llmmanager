"""Ollama model library browser."""

from __future__ import annotations

import httpx

from llmmanager.exceptions import HubConnectionError
from llmmanager.models.llm_model import LLMModel, ModelSource


async def search_models(query: str = "", limit: int = 50) -> list[LLMModel]:
    """
    Fetch models from the Ollama library API.
    Returns a list of LLMModel objects with available metadata.
    """
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            params: dict = {"limit": limit}
            if query:
                params["q"] = query
            r = await client.get("https://ollama.ai/api/models", params=params)
            r.raise_for_status()
            data = r.json()
    except httpx.ConnectError as exc:
        raise HubConnectionError("Cannot reach ollama.ai") from exc
    except Exception as exc:
        raise HubConnectionError(f"Ollama library request failed: {exc}") from exc

    models: list[LLMModel] = []
    for item in data.get("models", []):
        models.append(LLMModel(
            model_id=item.get("name", ""),
            display_name=item.get("name", ""),
            source=ModelSource.OLLAMA_LIBRARY,
            description=item.get("description", ""),
            tags=item.get("tags", []),
            ollama_tag=item.get("name", ""),
        ))
    return models
