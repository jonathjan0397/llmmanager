"""Static cloud model catalogs for OpenAI, Anthropic, and Groq."""

from __future__ import annotations

from llmmanager.models.llm_model import LLMModel, ModelSource

# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------

OPENAI_MODELS: list[LLMModel] = [
    LLMModel(model_id="gpt-4o",           display_name="GPT-4o",           source=ModelSource.OPENAI, description="Most capable multimodal model", context_length=128_000),
    LLMModel(model_id="gpt-4o-mini",      display_name="GPT-4o Mini",      source=ModelSource.OPENAI, description="Fast and affordable GPT-4o",    context_length=128_000),
    LLMModel(model_id="gpt-4-turbo",      display_name="GPT-4 Turbo",      source=ModelSource.OPENAI, description="Previous GPT-4 Turbo",           context_length=128_000),
    LLMModel(model_id="o1",               display_name="o1",                source=ModelSource.OPENAI, description="Advanced reasoning model",       context_length=200_000),
    LLMModel(model_id="o1-mini",          display_name="o1-mini",           source=ModelSource.OPENAI, description="Faster reasoning model",         context_length=128_000),
    LLMModel(model_id="o3-mini",          display_name="o3-mini",           source=ModelSource.OPENAI, description="Latest reasoning model",         context_length=200_000),
    LLMModel(model_id="gpt-3.5-turbo",   display_name="GPT-3.5 Turbo",    source=ModelSource.OPENAI, description="Fast and cheap legacy model",     context_length=16_385),
]

# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------

ANTHROPIC_MODELS: list[LLMModel] = [
    LLMModel(model_id="claude-opus-4-6",          display_name="Claude Opus 4.6",    source=ModelSource.ANTHROPIC, description="Most capable Claude model",   context_length=200_000),
    LLMModel(model_id="claude-sonnet-4-6",        display_name="Claude Sonnet 4.6",  source=ModelSource.ANTHROPIC, description="Balanced performance",         context_length=200_000),
    LLMModel(model_id="claude-haiku-4-5-20251001",display_name="Claude Haiku 4.5",   source=ModelSource.ANTHROPIC, description="Fast and lightweight",          context_length=200_000),
    LLMModel(model_id="claude-3-5-sonnet-20241022",display_name="Claude 3.5 Sonnet", source=ModelSource.ANTHROPIC, description="Previous generation Sonnet",    context_length=200_000),
    LLMModel(model_id="claude-3-opus-20240229",   display_name="Claude 3 Opus",      source=ModelSource.ANTHROPIC, description="Previous generation Opus",      context_length=200_000),
]

# ---------------------------------------------------------------------------
# Groq
# ---------------------------------------------------------------------------

GROQ_MODELS: list[LLMModel] = [
    LLMModel(model_id="llama-3.3-70b-versatile",  display_name="Llama 3.3 70B",       source=ModelSource.GROQ, description="Meta Llama 3.3 70B on Groq",  context_length=128_000),
    LLMModel(model_id="llama-3.1-8b-instant",     display_name="Llama 3.1 8B Instant",source=ModelSource.GROQ, description="Ultra-fast Llama 3.1 8B",      context_length=128_000),
    LLMModel(model_id="deepseek-r1-distill-llama-70b", display_name="DeepSeek R1 70B",source=ModelSource.GROQ, description="DeepSeek reasoning on Groq",   context_length=128_000),
    LLMModel(model_id="mixtral-8x7b-32768",        display_name="Mixtral 8x7B",        source=ModelSource.GROQ, description="Mistral MoE on Groq",           context_length=32_768),
    LLMModel(model_id="gemma2-9b-it",              display_name="Gemma 2 9B",           source=ModelSource.GROQ, description="Google Gemma 2 9B on Groq",    context_length=8_192),
]


def get_cloud_models(
    openai_key: str | None,
    anthropic_key: str | None,
    groq_key: str | None,
) -> list[LLMModel]:
    """Return available cloud models based on which API keys are configured."""
    models: list[LLMModel] = []
    if openai_key:
        models.extend(OPENAI_MODELS)
    if anthropic_key:
        models.extend(ANTHROPIC_MODELS)
    if groq_key:
        models.extend(GROQ_MODELS)
    return models
