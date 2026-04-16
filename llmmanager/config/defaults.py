"""Factory defaults for each server type."""

from llmmanager.config.schema import ServerConfig
from llmmanager.constants import (
    OLLAMA_DEFAULT_PORT,
    VLLM_DEFAULT_PORT,
    LMSTUDIO_DEFAULT_PORT,
    LLAMACPP_DEFAULT_PORT,
)


def default_ollama_config() -> ServerConfig:
    return ServerConfig(
        server_type="ollama",
        port=OLLAMA_DEFAULT_PORT,
        enabled=True,
        auto_start=False,
    )


def default_vllm_config() -> ServerConfig:
    return ServerConfig(
        server_type="vllm",
        port=VLLM_DEFAULT_PORT,
        enabled=True,
        auto_start=False,
    )


def default_lmstudio_config() -> ServerConfig:
    return ServerConfig(
        server_type="lmstudio",
        port=LMSTUDIO_DEFAULT_PORT,
        enabled=True,
        auto_start=False,
    )


def default_llamacpp_config() -> ServerConfig:
    return ServerConfig(
        server_type="llamacpp",
        port=LLAMACPP_DEFAULT_PORT,
        enabled=True,
        auto_start=False,
    )


SERVER_DEFAULTS: dict[str, ServerConfig] = {
    "ollama": default_ollama_config(),
    "vllm": default_vllm_config(),
    "lmstudio": default_lmstudio_config(),
    "llamacpp": default_llamacpp_config(),
}
