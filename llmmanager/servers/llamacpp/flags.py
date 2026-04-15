"""llama.cpp server CLI flag definitions for the FlagForm dynamic UI."""

from llmmanager.config.schema import FlagDefinition

LLAMACPP_FLAGS: list[FlagDefinition] = [
    # --- Model ---
    FlagDefinition(
        name="--model",
        type="str",
        default="",
        description="Path to a GGUF model file. Required.",
        category="Model",
    ),
    FlagDefinition(
        name="--model-alias",
        type="str",
        default="",
        description=(
            "Alias name for the model reported in /v1/models. "
            "Defaults to the filename."
        ),
        category="Model",
    ),
    FlagDefinition(
        name="--ctx-size",
        type="int",
        default=2048,
        description=(
            "Context window size in tokens. Larger values use more memory. "
            "Set to 0 to use the model's native context length."
        ),
        category="Model",
    ),
    FlagDefinition(
        name="--n-predict",
        type="int",
        default=-1,
        description="Max tokens to generate per request. -1 = unlimited.",
        category="Model",
    ),

    # --- GPU offload ---
    FlagDefinition(
        name="--n-gpu-layers",
        type="int",
        default=0,
        description=(
            "Number of model layers to offload to GPU. "
            "0 = CPU only. -1 = offload all layers."
        ),
        category="GPU",
    ),
    FlagDefinition(
        name="--main-gpu",
        type="int",
        default=0,
        description="GPU index to use as the primary device for multi-GPU setups.",
        category="GPU",
    ),
    FlagDefinition(
        name="--split-mode",
        type="choice",
        choices=["none", "layer", "row"],
        default="layer",
        description=(
            "How to split the model across multiple GPUs. "
            "'layer' splits by layer (recommended), 'row' splits by tensor row."
        ),
        category="GPU",
    ),
    FlagDefinition(
        name="--flash-attn",
        type="bool",
        default=False,
        description=(
            "Enable flash attention for faster inference and lower VRAM usage. "
            "Requires compatible GPU and build."
        ),
        category="GPU",
    ),

    # --- Performance ---
    FlagDefinition(
        name="--threads",
        type="int",
        default=None,
        description=(
            "Number of CPU threads for generation. "
            "Defaults to physical core count. Reduce to leave headroom for other tasks."
        ),
        category="Performance",
    ),
    FlagDefinition(
        name="--threads-batch",
        type="int",
        default=None,
        description=(
            "CPU threads for prompt evaluation (batched processing). "
            "Defaults to --threads."
        ),
        category="Performance",
    ),
    FlagDefinition(
        name="--batch-size",
        type="int",
        default=512,
        description="Prompt processing batch size. Larger = faster prompt eval, more RAM.",
        category="Performance",
    ),
    FlagDefinition(
        name="--ubatch-size",
        type="int",
        default=512,
        description="Physical batch size for prompt processing. Must be ≤ batch-size.",
        category="Performance",
    ),
    FlagDefinition(
        name="--cont-batching",
        type="bool",
        default=True,
        description=(
            "Enable continuous batching for higher throughput under concurrent load. "
            "Allows new requests to be inserted into running batches."
        ),
        category="Performance",
    ),

    # --- Memory ---
    FlagDefinition(
        name="--mlock",
        type="bool",
        default=False,
        description=(
            "Lock the model in RAM to prevent swapping. "
            "Improves consistency at the cost of reserved physical memory."
        ),
        category="Memory",
    ),
    FlagDefinition(
        name="--no-mmap",
        type="bool",
        default=False,
        description=(
            "Disable memory-mapped model loading. "
            "Use when the filesystem does not support mmap or to force full RAM load."
        ),
        category="Memory",
    ),

    # --- Serving ---
    FlagDefinition(
        name="--parallel",
        type="int",
        default=1,
        description=(
            "Number of parallel request slots. "
            "Each slot reserves a full context window of KV cache. "
            "Increase for multi-user deployments."
        ),
        category="Serving",
    ),
    FlagDefinition(
        name="--timeout",
        type="int",
        default=600,
        description="Request timeout in seconds before the server aborts generation.",
        category="Serving",
    ),

    # --- Sampling defaults ---
    FlagDefinition(
        name="--temp",
        type="float",
        default=0.8,
        description="Default sampling temperature (0 = greedy). Overridden per request.",
        category="Sampling",
    ),
    FlagDefinition(
        name="--repeat-penalty",
        type="float",
        default=1.1,
        description="Penalty multiplier for repeating tokens. 1.0 = disabled.",
        category="Sampling",
    ),
    FlagDefinition(
        name="--seed",
        type="int",
        default=-1,
        description="RNG seed. -1 = random each run.",
        category="Sampling",
    ),
]
