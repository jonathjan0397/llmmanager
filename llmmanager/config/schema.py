"""Pydantic config schemas — the single source of truth for all persisted settings."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field


class FlagDefinition(BaseModel):
    """Describes one CLI flag for dynamic FlagForm generation in the TUI."""

    name: str
    """CLI flag name, e.g. '--port'"""

    env_var: str | None = None
    """Corresponding env var, e.g. 'OLLAMA_PORT'"""

    type: Literal["int", "float", "str", "bool", "path", "choice"] = "str"
    """Value type — drives the input widget rendered in FlagForm."""

    choices: list[str] | None = None
    """Valid choices when type='choice'."""

    default: Any = None
    """Default value shown in the form."""

    description: str = ""
    """Human-readable explanation shown as help text in the TUI."""

    category: str = "General"
    """Groups flags into collapsible sections in FlagForm."""

    requires_restart: bool = True
    """Show restart-required banner when this flag is changed."""


class ServerConfig(BaseModel):
    """Per-server instance configuration persisted to TOML."""

    server_type: str
    """Identifies the backend: 'ollama' | 'vllm' | 'lmstudio'"""

    enabled: bool = True
    auto_start: bool = False
    host: str = "127.0.0.1"
    port: int = 0
    """0 means use the server's own default port."""

    flags: dict[str, Any] = Field(default_factory=dict)
    """CLI flag overrides: flag_name -> value. Passed verbatim at launch."""

    extra_env: dict[str, str] = Field(default_factory=dict)
    """Additional environment variables injected into the server process."""

    log_file: Path | None = None
    """If set, server stdout/stderr is tee'd to this path."""


class ProfileConfig(BaseModel):
    """A named snapshot of server configs — switch between use-case setups."""

    name: str
    description: str = ""
    servers: dict[str, ServerConfig] = Field(default_factory=dict)
    """Keyed by server_type."""

    created_at: str = ""
    """ISO-8601 timestamp."""

    updated_at: str = ""


class NotificationConfig(BaseModel):
    low_vram_threshold_pct: float = 10.0
    """Warn when free VRAM drops below this percentage."""

    crash_auto_restart: bool = False
    """Automatically restart a server if crash is detected."""

    desktop_notify: bool = False
    """Send OS desktop notifications (requires notify-send on Linux)."""


class BenchmarkDefaults(BaseModel):
    quick_prompt: str = "Explain what a transformer model is in one sentence."
    n_tokens: int = 200
    n_runs: int = 3
    warm_up: bool = True
    max_concurrency: int = 128
    """Ceiling for concurrency ramp test."""

    concurrency_levels: list[int] = Field(
        default_factory=lambda: [1, 2, 4, 8, 16, 32, 64, 128]
    )
    safety_max_p99_latency_ms: int = 30_000
    safety_max_error_rate_pct: float = 10.0
    sustained_duration_s: int = 60


class AppConfig(BaseModel):
    """Root config object serialized to ~/.config/llmmanager/config.toml."""

    version: int = 1
    """Schema version — used by ConfigManager to apply migrations."""

    poll_interval_ms: int = 2000
    log_tail_lines: int = 500
    theme: Literal["dark", "light"] = "dark"
    active_profile: str | None = None

    servers: dict[str, ServerConfig] = Field(default_factory=dict)
    """Keyed by server_type."""

    profiles: dict[str, ProfileConfig] = Field(default_factory=dict)
    notifications: NotificationConfig = Field(default_factory=NotificationConfig)
    benchmark: BenchmarkDefaults = Field(default_factory=BenchmarkDefaults)

    hf_token: str | None = None
    """HuggingFace API token for gated model access."""

    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    groq_api_key: str | None = None
