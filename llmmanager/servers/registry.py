"""ServerRegistry — discovers and holds all server backend instances."""

from __future__ import annotations

from llmmanager.config.manager import ConfigManager
from llmmanager.config.defaults import SERVER_DEFAULTS
from llmmanager.servers.base import AbstractServer


class ServerRegistry:
    """
    Holds one AbstractServer instance per configured server type.
    Created once at app startup and held for the app lifetime.
    """

    def __init__(self, config_manager: ConfigManager) -> None:
        self._config_manager = config_manager
        self._servers: dict[str, AbstractServer] = {}

    def initialize(self) -> None:
        """Instantiate all enabled server backends from config."""
        # Import here to avoid circular imports at module load time
        from llmmanager.servers.ollama.server import OllamaServer
        from llmmanager.servers.vllm.server import VLLMServer
        from llmmanager.servers.lmstudio.server import LMStudioServer

        _BACKEND_MAP: dict[str, type[AbstractServer]] = {
            "ollama": OllamaServer,
            "vllm": VLLMServer,
            "lmstudio": LMStudioServer,
        }

        cfg = self._config_manager.config

        for server_type, backend_cls in _BACKEND_MAP.items():
            server_cfg = cfg.servers.get(server_type, SERVER_DEFAULTS.get(server_type))
            if server_cfg is None:
                continue
            self._servers[server_type] = backend_cls(server_cfg)

    def get(self, server_type: str) -> AbstractServer | None:
        return self._servers.get(server_type)

    def all(self) -> list[AbstractServer]:
        return list(self._servers.values())

    def all_enabled(self) -> list[AbstractServer]:
        return [s for s in self._servers.values() if s.config.enabled]
