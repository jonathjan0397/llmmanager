"""HuggingFace Hub model browser — uses HF REST API directly via httpx."""

from __future__ import annotations

import httpx

from llmmanager.exceptions import HubConnectionError
from llmmanager.models.llm_model import LLMModel, ModelSource


_HF_API = "https://huggingface.co/api/models"


async def search_models(
    query: str = "",
    limit: int = 30,
    hf_token: str | None = None,
) -> list[LLMModel]:
    """
    Search HuggingFace Hub for GGUF models via the public REST API.
    No huggingface_hub package required.
    """
    params: dict = {
        "filter": "gguf",
        "sort": "downloads",
        "direction": -1,
        "limit": limit,
        "full": "false",
    }
    if query:
        params["search"] = query

    headers = {}
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.get(_HF_API, params=params, headers=headers)
            r.raise_for_status()
            items = r.json()
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code in (401, 403):
            raise HubConnectionError("HuggingFace token required or invalid") from exc
        raise HubConnectionError(f"HuggingFace API error: {exc}") from exc
    except Exception as exc:
        raise HubConnectionError(f"HuggingFace request failed: {exc}") from exc

    models: list[LLMModel] = []
    for item in items:
        repo_id = item.get("id", "")
        if not repo_id:
            continue
        downloads = item.get("downloads", 0)
        tags = [t for t in (item.get("tags") or []) if not t.startswith("arxiv:")]
        models.append(LLMModel(
            model_id=repo_id,
            display_name=repo_id,
            source=ModelSource.HUGGINGFACE,
            tags=tags[:8],
            description=f"{downloads:,} downloads",
            hf_repo_id=repo_id,
        ))
    return models
