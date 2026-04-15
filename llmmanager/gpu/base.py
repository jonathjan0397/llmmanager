"""Abstract GPU provider interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from llmmanager.models.gpu import GPUInfo, GPUVendor


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
