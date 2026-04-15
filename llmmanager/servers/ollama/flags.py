"""Ollama CLI flag definitions for the FlagForm dynamic UI."""

from llmmanager.config.schema import FlagDefinition

OLLAMA_FLAGS: list[FlagDefinition] = [
    # --- General ---
    FlagDefinition(
        name="--host",
        env_var="OLLAMA_HOST",
        type="str",
        default="127.0.0.1:11434",
        description="IP:port the Ollama server listens on.",
        category="General",
    ),
    FlagDefinition(
        name="--origins",
        env_var="OLLAMA_ORIGINS",
        type="str",
        default="",
        description="Comma-separated list of allowed CORS origins. '*' allows all.",
        category="General",
    ),
    FlagDefinition(
        name="--models",
        env_var="OLLAMA_MODELS",
        type="path",
        default="~/.ollama/models",
        description="Directory where model blobs are stored.",
        category="General",
    ),

    # --- Memory / Loading ---
    FlagDefinition(
        name="--keep-alive",
        env_var="OLLAMA_KEEP_ALIVE",
        type="str",
        default="5m",
        description=(
            "How long a model stays loaded after the last request. "
            "Use '0' to unload immediately, '-1' to never unload. "
            "Supports duration strings: '5m', '1h', '30s'."
        ),
        category="Memory",
    ),
    FlagDefinition(
        name="--max-loaded-models",
        env_var="OLLAMA_MAX_LOADED_MODELS",
        type="int",
        default=1,
        description=(
            "Maximum number of models that can be simultaneously loaded. "
            "Increase for multi-model concurrency at the cost of VRAM."
        ),
        category="Memory",
    ),
    FlagDefinition(
        name="--max-queue",
        env_var="OLLAMA_MAX_QUEUE",
        type="int",
        default=512,
        description="Maximum number of queued requests before new ones are rejected.",
        category="Memory",
    ),

    # --- Performance ---
    FlagDefinition(
        name="--num-parallel",
        env_var="OLLAMA_NUM_PARALLEL",
        type="int",
        default=1,
        description=(
            "Maximum number of parallel requests processed per model. "
            "Higher values increase throughput but require more VRAM."
        ),
        category="Performance",
    ),
    FlagDefinition(
        name="--flash-attention",
        env_var="OLLAMA_FLASH_ATTENTION",
        type="bool",
        default=False,
        description=(
            "Enable Flash Attention for supported models. "
            "Reduces VRAM usage and improves speed on compatible hardware."
        ),
        category="Performance",
    ),
    FlagDefinition(
        name="--kv-cache-type",
        env_var="OLLAMA_KV_CACHE_TYPE",
        type="choice",
        choices=["f16", "q8_0", "q4_0"],
        default="f16",
        description=(
            "KV cache quantization. 'q8_0' and 'q4_0' reduce VRAM at a small quality cost. "
            "Requires Flash Attention to be enabled."
        ),
        category="Performance",
    ),
    FlagDefinition(
        name="--gpu-overhead",
        env_var="OLLAMA_GPU_OVERHEAD",
        type="int",
        default=0,
        description=(
            "Amount of VRAM (bytes) to reserve for the OS and other applications. "
            "Helps prevent OOM kills on GPUs shared with a desktop environment."
        ),
        category="Performance",
    ),

    # --- Compute ---
    FlagDefinition(
        name="--num-gpu",
        env_var="OLLAMA_NUM_GPU",
        type="int",
        default=-1,
        description=(
            "Number of GPU layers to offload. -1 = auto (offload all possible layers). "
            "Set to 0 to force CPU inference."
        ),
        category="Compute",
    ),
    FlagDefinition(
        name="--main-gpu",
        env_var="OLLAMA_MAIN_GPU",
        type="int",
        default=0,
        description="Index of the primary GPU for multi-GPU setups.",
        category="Compute",
    ),
    FlagDefinition(
        name="--low-vram",
        env_var="OLLAMA_LOW_VRAM",
        type="bool",
        default=False,
        description=(
            "Enable low-VRAM mode: disables the KV cache to save memory "
            "at the cost of significantly reduced performance."
        ),
        category="Compute",
    ),

    # --- Networking ---
    FlagDefinition(
        name="--noprune",
        env_var="OLLAMA_NOPRUNE",
        type="bool",
        default=False,
        description="Disable automatic pruning of old model versions on startup.",
        category="Storage",
        requires_restart=True,
    ),

    # --- Debug ---
    FlagDefinition(
        name="--debug",
        env_var="OLLAMA_DEBUG",
        type="bool",
        default=False,
        description="Enable verbose debug logging to stdout.",
        category="Debug",
    ),
    FlagDefinition(
        name="--log-level",
        env_var="OLLAMA_LOG_LEVEL",
        type="choice",
        choices=["debug", "info", "warn", "error"],
        default="info",
        description="Minimum log level emitted by the server.",
        category="Debug",
    ),
]
