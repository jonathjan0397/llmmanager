"""Auto-detect available GPU vendor and return the appropriate provider."""

from __future__ import annotations

from llmmanager.gpu.base import AbstractGPUProvider
from llmmanager.gpu.amd import AMDProvider
from llmmanager.gpu.cpu_only import CPUOnlyProvider
from llmmanager.gpu.intel import IntelProvider
from llmmanager.gpu.nvidia import NvidiaProvider

# Probe order: most capable / common first, CPU-only always last
_PROBE_ORDER: list[type[AbstractGPUProvider]] = [
    NvidiaProvider,
    AMDProvider,
    IntelProvider,
    CPUOnlyProvider,
]


def detect_gpu_provider() -> AbstractGPUProvider:
    """
    Probe GPU vendors in order and return the first available provider instance.
    Never raises — CPUOnlyProvider is always available as the final fallback.
    """
    for cls in _PROBE_ORDER:
        if cls.is_available():
            return cls()
    # Should never be reached because CPUOnlyProvider.is_available() == True
    return CPUOnlyProvider()
