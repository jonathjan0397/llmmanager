"""Ollama model library browser."""

from __future__ import annotations

import httpx

from llmmanager.exceptions import HubConnectionError
from llmmanager.models.llm_model import LLMModel, ModelSource


# Popular curated models shown when the live API is unavailable.
_POPULAR_MODELS: list[dict] = [
    {"name": "llama3.2",        "tags": ["1b", "3b"],                  "description": "Meta's Llama 3.2 lightweight models"},
    {"name": "llama3.1",        "tags": ["8b", "70b"],                 "description": "Meta's Llama 3.1 instruction-tuned models"},
    {"name": "llama3",          "tags": ["8b", "70b"],                 "description": "Meta's Llama 3 models"},
    {"name": "mistral",         "tags": ["7b"],                        "description": "Mistral 7B instruction model"},
    {"name": "mixtral",         "tags": ["8x7b", "8x22b"],             "description": "Mistral MoE models"},
    {"name": "gemma2",          "tags": ["2b", "9b", "27b"],           "description": "Google Gemma 2 models"},
    {"name": "gemma",           "tags": ["2b", "7b"],                  "description": "Google Gemma models"},
    {"name": "phi4",            "tags": ["14b"],                       "description": "Microsoft Phi-4"},
    {"name": "phi3",            "tags": ["3.8b", "14b"],               "description": "Microsoft Phi-3 models"},
    {"name": "qwen2.5",         "tags": ["0.5b", "1.5b", "3b", "7b", "14b", "32b"], "description": "Alibaba Qwen 2.5"},
    {"name": "qwen2.5-coder",   "tags": ["1.5b", "7b", "14b", "32b"], "description": "Qwen 2.5 code-specialized"},
    {"name": "deepseek-r1",     "tags": ["1.5b", "7b", "8b", "14b", "32b"], "description": "DeepSeek R1 reasoning models"},
    {"name": "deepseek-coder-v2", "tags": ["16b", "236b"],            "description": "DeepSeek Coder V2"},
    {"name": "codellama",       "tags": ["7b", "13b", "34b"],          "description": "Meta Code Llama"},
    {"name": "nomic-embed-text","tags": ["latest"],                    "description": "Nomic text embedding model"},
    {"name": "mxbai-embed-large", "tags": ["latest"],                  "description": "MixedBread large embedding model"},
    {"name": "starcoder2",      "tags": ["3b", "7b", "15b"],           "description": "Starcoder2 code models"},
    {"name": "neural-chat",     "tags": ["7b"],                        "description": "Intel Neural Chat"},
    {"name": "dolphin-mistral", "tags": ["7b"],                        "description": "Dolphin Mistral uncensored"},
    {"name": "vicuna",          "tags": ["7b", "13b"],                 "description": "Vicuna fine-tuned on ShareGPT"},
]


async def search_models(query: str = "", limit: int = 50) -> list[LLMModel]:
    """
    Fetch models from the Ollama library API.
    Falls back to a curated popular-models list if the live API is unreachable.
    """
    try:
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
            params: dict = {"limit": limit}
            if query:
                params["q"] = query
            r = await client.get("https://ollama.com/api/models", params=params)
            r.raise_for_status()
            data = r.json()

        items = data.get("models", data if isinstance(data, list) else [])
        if items:
            return [
                LLMModel(
                    model_id=item.get("name", ""),
                    display_name=item.get("name", ""),
                    source=ModelSource.OLLAMA_LIBRARY,
                    description=item.get("description", ""),
                    tags=item.get("tags", []),
                    ollama_tag=item.get("name", ""),
                )
                for item in items
                if item.get("name")
            ]

    except Exception:
        pass  # Fall through to curated list

    # Curated fallback — filter by query if provided
    q = query.lower()
    results: list[LLMModel] = []
    for item in _POPULAR_MODELS:
        name = item.get("name", "")
        if q and q not in name.lower() and q not in item.get("description", "").lower():
            continue
        results.append(LLMModel(
            model_id=name,
            display_name=name,
            source=ModelSource.OLLAMA_LIBRARY,
            description=item.get("description", ""),
            tags=item.get("tags", []) if isinstance(item.get("tags"), list) else [],
            ollama_tag=name,
        ))
        if len(results) >= limit:
            break

    return results
