"""HuggingFace Hub model browser."""

from __future__ import annotations

from llmmanager.exceptions import HubAuthError, HubConnectionError
from llmmanager.models.llm_model import LLMModel, ModelSource


async def search_models(
    query: str,
    limit: int = 30,
    hf_token: str | None = None,
) -> list[LLMModel]:
    """
    Search HuggingFace Hub for GGUF/safetensors models.
    Returns a list of LLMModel objects.
    """
    try:
        import asyncio
        from huggingface_hub import HfApi
        api = HfApi(token=hf_token)
        results = await asyncio.to_thread(
            lambda: list(api.list_models(
                search=query,
                filter="gguf",
                sort="downloads",
                direction=-1,
                limit=limit,
            ))
        )
    except ImportError as exc:
        raise HubConnectionError("huggingface-hub package not installed") from exc
    except Exception as exc:
        if "401" in str(exc) or "403" in str(exc):
            raise HubAuthError("HuggingFace token required or invalid") from exc
        raise HubConnectionError(f"HuggingFace Hub request failed: {exc}") from exc

    models: list[LLMModel] = []
    for m in results:
        models.append(LLMModel(
            model_id=m.id,
            display_name=m.id,
            source=ModelSource.HUGGINGFACE,
            tags=list(m.tags or []),
            hf_repo_id=m.id,
        ))
    return models
