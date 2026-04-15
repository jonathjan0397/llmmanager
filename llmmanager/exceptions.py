"""Custom exception hierarchy for LLMManager."""


class LLMManagerError(Exception):
    """Base exception for all LLMManager errors."""


# --- Server errors ---

class ServerError(LLMManagerError):
    """Base for server-related errors."""

class ServerNotInstalledError(ServerError):
    """Server binary/package is not installed."""

class ServerAlreadyRunningError(ServerError):
    """Attempted to start a server that is already running."""

class ServerNotRunningError(ServerError):
    """Attempted an operation on a server that is not running."""

class ServerStartError(ServerError):
    """Server failed to start."""

class ServerStopError(ServerError):
    """Server failed to stop cleanly."""

class ServerInstallError(ServerError):
    """Installation of a server failed."""

class ServerVersionError(ServerError):
    """Version detection or constraint failure."""

class PortConflictError(ServerError):
    """Requested port is already in use."""
    def __init__(self, port: int, pid: int | None = None) -> None:
        self.port = port
        self.pid = pid
        msg = f"Port {port} is already in use"
        if pid:
            msg += f" by PID {pid}"
        super().__init__(msg)


# --- Model errors ---

class ModelError(LLMManagerError):
    """Base for model-related errors."""

class ModelNotFoundError(ModelError):
    """Model does not exist locally or in the registry."""

class ModelLoadError(ModelError):
    """Failed to load a model into a server."""

class ModelDownloadError(ModelError):
    """Model download failed or was interrupted."""

class InsufficientVRAMError(ModelError):
    """Not enough VRAM to load the requested model."""
    def __init__(self, required_mb: float, available_mb: float) -> None:
        self.required_mb = required_mb
        self.available_mb = available_mb
        super().__init__(
            f"Insufficient VRAM: need {required_mb:.0f}MB, have {available_mb:.0f}MB free"
        )


# --- GPU errors ---

class GPUError(LLMManagerError):
    """Base for GPU telemetry errors."""

class GPUNotAvailableError(GPUError):
    """No supported GPU vendor detected."""

class GPUQueryError(GPUError):
    """Failed to query GPU statistics."""


# --- Config errors ---

class ConfigError(LLMManagerError):
    """Base for configuration errors."""

class ConfigLoadError(ConfigError):
    """Failed to load or parse config file."""

class ConfigSaveError(ConfigError):
    """Failed to write config file."""

class ConfigMigrationError(ConfigError):
    """Config migration to a newer schema version failed."""

class ProfileNotFoundError(ConfigError):
    """Named profile does not exist."""


# --- Benchmark errors ---

class BenchmarkError(LLMManagerError):
    """Base for benchmark errors."""

class BenchmarkAbortedError(BenchmarkError):
    """Benchmark was aborted due to safety cutoff or user cancellation."""
    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"Benchmark aborted: {reason}")


# --- Hub errors ---

class HubError(LLMManagerError):
    """Base for model hub errors."""

class HubConnectionError(HubError):
    """Cannot reach the model hub (network error)."""

class HubAuthError(HubError):
    """Authentication required or token invalid."""
