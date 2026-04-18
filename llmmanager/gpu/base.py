"""Abstract GPU provider interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from llmmanager.models.gpu import GPUInfo, GPUProcess, GPUVendor


class AbstractGPUProvider(ABC):
    vendor: GPUVendor

    @classmethod
    @abstractmethod
    def is_available(cls) -> bool:
        """Cheap synchronous check — does not raise, returns bool."""
        ...

    @abstractmethod
    async def initialize(self) -> None:
        """One-time async setup (e.g. nvmlInit). Called once at app start."""
        ...

    @abstractmethod
    async def get_all_gpus(self) -> list[GPUInfo]:
        """Return current stats for all detected GPUs of this vendor."""
        ...

    @abstractmethod
    async def shutdown(self) -> None:
        """Clean up resources (e.g. nvmlShutdown)."""
        ...

    async def get_processes(self) -> list[GPUProcess]:
        """Return running GPU processes across all devices. Default: not supported."""
        return []

    async def set_fan_speed(self, gpu_index: int, speed_pct: int) -> tuple[bool, str]:
        """
        Set fan to a fixed speed (0-100 %).
        Returns (success, message).
        Default: not supported — subclasses override where possible.
        Requires root on Linux; not available on Windows.
        """
        return False, "Fan control is not supported for this GPU vendor."

    async def set_fan_auto(self, gpu_index: int) -> tuple[bool, str]:
        """Return fan to automatic/temperature-driven control. Returns (success, message)."""
        return False, "Fan control is not supported for this GPU vendor."

    async def set_fan_speed_sudo(self, gpu_index: int, speed_pct: int, sudo_password: str) -> tuple[bool, str]:
        """set_fan_speed via sudo fallback. Subclasses implement where supported."""
        return False, "Fan control is not supported for this GPU vendor."

    async def set_fan_auto_sudo(self, gpu_index: int, sudo_password: str) -> tuple[bool, str]:
        """set_fan_auto via sudo fallback. Subclasses implement where supported."""
        return False, "Fan control is not supported for this GPU vendor."
