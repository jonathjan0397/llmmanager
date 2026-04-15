"""Abstract server interface — every LLM server backend implements this."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncIterator

from llmmanager.config.schema import FlagDefinition, ServerConfig
from llmmanager.models.llm_model import LLMModel
from llmmanager.models.server import EndpointInfo, ServerInfo, ServerStatus


class AbstractServer(ABC):
    """
    Contract for all LLM server backends.

    Instances are created by ServerRegistry and held for the app lifetime.
    All I/O methods are async — never block the Textual event loop.
    """

    name: str
    """Identifier used as dict key: 'ollama' | 'vllm' | 'lmstudio'"""

    display_name: str
    """Human-readable name shown in the TUI."""

    def __init__(self, config: ServerConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @abstractmethod
    async def start(self) -> None:
        """Start the server process."""
        ...

    @abstractmethod
    async def stop(self) -> None:
        """Stop the server process gracefully."""
        ...

    @abstractmethod
    async def restart(self) -> None:
        """Stop then start."""
        ...

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    @abstractmethod
    async def get_status(self) -> ServerStatus:
        """Return current status, re-attaching to an existing PID if needed."""
        ...

    @abstractmethod
    async def get_info(self) -> ServerInfo:
        """Return full server info including status."""
        ...

    # ------------------------------------------------------------------
    # Models
    # ------------------------------------------------------------------

    @abstractmethod
    async def list_loaded_models(self) -> list[LLMModel]:
        """Return models currently loaded/available in this server."""
        ...

    @abstractmethod
    async def load_model(self, model_id: str) -> None:
        """Load (pull if needed) the named model into the server."""
        ...

    @abstractmethod
    async def unload_model(self, model_id: str) -> None:
        """Unload the model from memory (keep on disk)."""
        ...

    @abstractmethod
    async def delete_model(self, model_id: str) -> None:
        """Delete the model from disk."""
        ...

    # ------------------------------------------------------------------
    # Endpoints
    # ------------------------------------------------------------------

    @abstractmethod
    async def get_endpoints(self) -> list[EndpointInfo]:
        """Return list of active API endpoints."""
        ...

    # ------------------------------------------------------------------
    # Logs
    # ------------------------------------------------------------------

    @abstractmethod
    async def stream_logs(self) -> AsyncIterator[str]:
        """Async generator yielding log lines as they arrive."""
        ...

    # ------------------------------------------------------------------
    # Quick inference (API panel test)
    # ------------------------------------------------------------------

    @abstractmethod
    async def quick_infer(
        self, model_id: str, prompt: str
    ) -> AsyncIterator[str]:
        """Stream a response for a single prompt. Used by API panel."""
        ...

    # ------------------------------------------------------------------
    # Installation
    # ------------------------------------------------------------------

    @abstractmethod
    async def is_installed(self) -> bool:
        """Return True if the server binary/package is present."""
        ...

    @abstractmethod
    async def get_installed_version(self) -> str | None:
        """Return the installed version string, or None if not installed."""
        ...

    @abstractmethod
    async def list_available_versions(self) -> list[str]:
        """Return available versions from the upstream release channel."""
        ...

    @abstractmethod
    async def install(self, version: str = "latest") -> AsyncIterator[str]:
        """Stream installation output lines. Raises ServerInstallError on failure."""
        ...

    @abstractmethod
    async def uninstall(self) -> AsyncIterator[str]:
        """Stream uninstallation output lines."""
        ...

    # ------------------------------------------------------------------
    # Flag metadata (drives FlagForm dynamic UI)
    # ------------------------------------------------------------------

    @classmethod
    @abstractmethod
    def get_flag_definitions(cls) -> list[FlagDefinition]:
        """Return ordered list of CLI flag definitions for this server."""
        ...

    # ------------------------------------------------------------------
    # Pre-flight checks (used by Setup Wizard)
    # ------------------------------------------------------------------

    @abstractmethod
    async def preflight_checks(self) -> list[tuple[str, bool, str]]:
        """
        Return a list of (check_name, passed, detail) tuples.
        Used by the Setup Wizard to show a checklist before installing.
        """
        ...
