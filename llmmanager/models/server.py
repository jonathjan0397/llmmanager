"""Server domain models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class ServerState(str, Enum):
    RUNNING  = "running"
    STOPPED  = "stopped"
    STARTING = "starting"
    STOPPING = "stopping"
    ERROR    = "error"
    UNKNOWN  = "unknown"


@dataclass
class EndpointInfo:
    url: str
    protocol: str
    """'openai-compat' | 'ollama-native' | 'grpc'"""
    description: str


@dataclass
class ServerStatus:
    state: ServerState
    pid: int | None = None
    uptime_seconds: float | None = None
    active_model: str | None = None
    loaded_models: list[str] = field(default_factory=list)
    endpoints: list[EndpointInfo] = field(default_factory=list)
    last_checked: datetime = field(default_factory=datetime.utcnow)
    error_message: str | None = None


@dataclass
class ServerInfo:
    server_type: str
    display_name: str
    version: str | None
    host: str
    port: int
    status: ServerStatus
