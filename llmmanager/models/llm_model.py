"""LLM model domain models."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ModelSource(str, Enum):
    OLLAMA_LIBRARY = "ollama_library"
    HUGGINGFACE    = "huggingface"
    LOCAL          = "local"


class CompatibilityTier(str, Enum):
    """Hardware compatibility classification shown in Model Management."""
    COMFORTABLE = "comfortable"
    """Fits in VRAM with headroom."""
    LIMITED     = "limited"
    """Fits but little headroom; performance may vary."""
    TOO_LARGE   = "too_large"
    """Will not fit in available VRAM."""
    UNKNOWN     = "unknown"


@dataclass
class LLMModel:
    model_id: str
    display_name: str
    source: ModelSource
    size_gb: float | None = None
    parameter_count_b: float | None = None
    """Parameter count in billions."""
    quantization: str | None = None
    """e.g. 'Q4_K_M', 'fp16', 'awq'"""
    format: str | None = None
    """'gguf' | 'safetensors' | 'awq'"""
    context_length: int | None = None
    tags: list[str] = field(default_factory=list)
    description: str = ""
    is_downloaded: bool = False
    is_loaded: bool = False
    vram_estimate_mb: float | None = None
    compatibility: CompatibilityTier = CompatibilityTier.UNKNOWN
    hf_repo_id: str | None = None
    ollama_tag: str | None = None


@dataclass
class DownloadProgress:
    model_id: str
    server_type: str
    total_bytes: int | None
    downloaded_bytes: int
    speed_bps: float
    eta_seconds: float | None
    status: str
    """'downloading' | 'verifying' | 'extracting' | 'complete' | 'error'"""
    error: str | None = None

    @property
    def progress_pct(self) -> float:
        if self.total_bytes and self.total_bytes > 0:
            return min(self.downloaded_bytes / self.total_bytes * 100, 100.0)
        return 0.0
