"""
Smoke tests — run on the server via:
    ~/.llmmanager-venv/bin/python tests/smoke_test.py
"""
import sys

PASS = 0
FAIL = 0


def check(label: str, fn):
    global PASS, FAIL
    try:
        result = fn()
        print(f"  [PASS] {label}: {result}")
        PASS += 1
    except Exception as exc:
        print(f"  [FAIL] {label}: {exc}")
        FAIL += 1


# Version
check("version", lambda: __import__("llmmanager").__version__)

# Core deps
def _deps():
    import textual, pydantic, psutil, aiofiles, httpx
    return f"textual={textual.__version__} pydantic={pydantic.__version__}"
check("core deps", _deps)

# GPU detection
def _gpu():
    from llmmanager.gpu.detector import detect_gpu_provider
    p = detect_gpu_provider()
    return f"vendor={p.vendor.value}"
check("GPU detection", _gpu)

# Config load
def _config():
    from llmmanager.config.manager import ConfigManager
    cm = ConfigManager()
    cfg = cm.load()
    return f"version={cfg.version}"
check("config load/save", _config)

# Server registry
def _registry():
    from llmmanager.config.manager import ConfigManager
    from llmmanager.servers.registry import ServerRegistry
    cm = ConfigManager()
    cm.load()
    r = ServerRegistry(cm)
    r.initialize()
    return [s.name for s in r.all()]
check("server registry", _registry)

# VRAM estimator
def _vram():
    from llmmanager.hub.vram_estimator import estimate_vram_mb, fits_in_vram
    mb = estimate_vram_mb(7.0, "Q4_K_M")
    fits, est = fits_in_vram(7.0, "Q4_K_M", 24_000)
    return f"{mb:.0f}MB estimated, fits_in_24GB={fits}"
check("VRAM estimator", _vram)

# Benchmark config
def _bench():
    from llmmanager.models.benchmark import BenchmarkConfig, BenchmarkProfile
    from llmmanager.constants import BENCHMARK_CONCURRENCY_LEVELS
    cfg = BenchmarkConfig(server_type="ollama", model_id="llama3.2:3b")
    assert cfg.concurrency_levels == BENCHMARK_CONCURRENCY_LEVELS
    return f"levels={cfg.concurrency_levels}"
check("benchmark config", _bench)

# Exception hierarchy
def _exc():
    from llmmanager.exceptions import (
        PortConflictError, InsufficientVRAMError, BenchmarkAbortedError,
        ServerNotInstalledError, GPUQueryError,
    )
    e = PortConflictError(11434, 1234)
    assert "11434" in str(e)
    return "exception hierarchy OK"
check("exception hierarchy", _exc)

# Ollama flags
def _flags():
    from llmmanager.servers.ollama.flags import OLLAMA_FLAGS
    from llmmanager.servers.vllm.flags import VLLM_FLAGS
    return f"ollama={len(OLLAMA_FLAGS)} flags, vllm={len(VLLM_FLAGS)} flags"
check("flag definitions", _flags)

# Notification manager
def _notif():
    from llmmanager.config.schema import AppConfig, NotificationConfig
    from llmmanager.notifications.manager import NotificationManager, Severity
    cfg = AppConfig()
    nm = NotificationManager(cfg)
    nm.add("Test", "body", Severity.INFO)
    assert nm.unread_count == 1
    nm.mark_all_read()
    assert nm.unread_count == 0
    return "notification bus OK"
check("notification manager", _notif)

# Config profiles
def _profiles():
    from llmmanager.config.manager import ConfigManager
    from llmmanager.config.schema import ProfileConfig
    from datetime import datetime, timezone
    cm = ConfigManager()
    cm.load()
    now = datetime.now(timezone.utc).isoformat()
    cm.config.profiles["test"] = ProfileConfig(
        name="test", created_at=now, updated_at=now
    )
    cm.save()
    cm2 = ConfigManager()
    cm2.load()
    assert "test" in cm2.config.profiles
    # cleanup
    del cm2.config.profiles["test"]
    cm2.save()
    return "profile save/load OK"
check("config profiles", _profiles)

print()
print(f"Results: {PASS} passed, {FAIL} failed")
sys.exit(0 if FAIL == 0 else 1)
