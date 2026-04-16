"""App-wide constants, default paths, and timeouts."""

from pathlib import Path
from platformdirs import user_config_dir, user_data_dir

APP_NAME = "llmmanager"
APP_VERSION = "0.1.9"
APP_DISPLAY_NAME = "LLM Manager"

# Config and data directories
CONFIG_DIR = Path(user_config_dir(APP_NAME))
DATA_DIR = Path(user_data_dir(APP_NAME))
CONFIG_FILE = CONFIG_DIR / "config.toml"
BENCHMARK_DIR = DATA_DIR / "benchmarks"
VENV_DIR = DATA_DIR / "venvs"
LOG_DIR = DATA_DIR / "logs"

# Polling
DEFAULT_POLL_INTERVAL_MS = 2000
MIN_POLL_INTERVAL_MS = 500
MAX_POLL_INTERVAL_MS = 30000

# Log tailing
DEFAULT_LOG_TAIL_LINES = 500
MAX_LOG_TAIL_LINES = 5000

# Benchmark concurrency levels
BENCHMARK_CONCURRENCY_LEVELS = [1, 2, 4, 8, 16, 32, 64, 128]

# Benchmark safety cutoffs
BENCHMARK_MAX_P99_LATENCY_MS = 30_000   # stop climbing if p99 > 30s
BENCHMARK_MAX_ERROR_RATE_PCT = 10.0     # stop climbing if error rate > 10%
BENCHMARK_SUSTAINED_DURATION_S = 60    # sustained concurrency test duration

# Benchmark quality probe prompt sets
BENCHMARK_PROBE_SETS = ["coding", "reasoning", "instruction", "chat"]

# VRAM warning threshold
DEFAULT_LOW_VRAM_THRESHOLD_PCT = 10.0

# Server defaults
OLLAMA_DEFAULT_PORT = 11434
VLLM_DEFAULT_PORT = 8000
LMSTUDIO_DEFAULT_PORT = 1234
LLAMACPP_DEFAULT_PORT = 8080

# HTTP timeouts (seconds)
HTTP_CONNECT_TIMEOUT = 5.0
HTTP_READ_TIMEOUT = 120.0
HTTP_QUICK_INFER_TIMEOUT = 300.0

# Process re-attach
PROCESS_HEALTH_CHECK_RETRIES = 3
PROCESS_HEALTH_CHECK_DELAY_S = 1.0

# Install
OLLAMA_INSTALL_SCRIPT = "https://ollama.ai/install.sh"
