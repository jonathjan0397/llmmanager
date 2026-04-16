"""Flag definitions for LM Studio (connection settings only — LM Studio is GUI-configured)."""

from llmmanager.config.schema import FlagDefinition

LMSTUDIO_FLAGS: list[FlagDefinition] = [
    FlagDefinition(
        name="--api-key",
        type="str",
        default="",
        description=(
            "API key required by LM Studio's local server. "
            "Set this in LM Studio under Local Server → API Key, "
            "then enter the same value here."
        ),
        category="Connection",
        requires_restart=False,
    ),
    FlagDefinition(
        name="--port",
        type="int",
        default=1234,
        description="Port LM Studio's local server is listening on (default: 1234).",
        category="Connection",
        requires_restart=False,
    ),
]
