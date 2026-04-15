"""vLLM CLI flag definitions for the FlagForm dynamic UI."""

from llmmanager.config.schema import FlagDefinition

VLLM_FLAGS: list[FlagDefinition] = [
    # --- Model ---
    FlagDefinition(
        name="--model",
        type="str",
        default="",
        description="HuggingFace model ID or local path to serve. Required.",
        category="Model",
    ),
    FlagDefinition(
        name="--tokenizer",
        type="str",
        default="",
        description="HuggingFace tokenizer ID or path. Defaults to the model's tokenizer.",
        category="Model",
    ),
    FlagDefinition(
        name="--revision",
        type="str",
        default="",
        description="Specific model revision (branch, tag, or commit hash) to load.",
        category="Model",
    ),
    FlagDefinition(
        name="--trust-remote-code",
        type="bool",
        default=False,
        description=(
            "Allow executing custom model code from HuggingFace. "
            "Only enable for models you trust."
        ),
        category="Model",
    ),
    FlagDefinition(
        name="--max-model-len",
        type="int",
        default=None,
        description=(
            "Maximum sequence length (context window). "
            "Defaults to the model's config value. Reduce to save VRAM."
        ),
        category="Model",
    ),

    # --- Quantization ---
    FlagDefinition(
        name="--quantization",
        type="choice",
        choices=["awq", "gptq", "squeezellm", "fp8", "bitsandbytes", "none"],
        default="none",
        description="Quantization method to apply when loading the model.",
        category="Quantization",
    ),
    FlagDefinition(
        name="--load-format",
        type="choice",
        choices=["auto", "pt", "safetensors", "npcache", "dummy", "gguf"],
        default="auto",
        description="Weight loading format. 'auto' detects from file extension.",
        category="Quantization",
    ),
    FlagDefinition(
        name="--dtype",
        type="choice",
        choices=["auto", "half", "float16", "bfloat16", "float", "float32"],
        default="auto",
        description=(
            "Model weight dtype. 'auto' uses bfloat16 for Ampere+ GPUs, float16 otherwise."
        ),
        category="Quantization",
    ),

    # --- Memory ---
    FlagDefinition(
        name="--gpu-memory-utilization",
        type="float",
        default=0.90,
        description=(
            "Fraction of GPU VRAM to allocate for the model and KV cache (0.0–1.0). "
            "Lower this if you share the GPU with other workloads."
        ),
        category="Memory",
    ),
    FlagDefinition(
        name="--swap-space",
        type="int",
        default=4,
        description="CPU swap space in GiB for KV cache blocks that don't fit in VRAM.",
        category="Memory",
    ),
    FlagDefinition(
        name="--kv-cache-dtype",
        type="choice",
        choices=["auto", "fp8", "fp8_e5m2", "fp8_e4m3"],
        default="auto",
        description="Dtype for KV cache. 'fp8' variants reduce memory at minimal quality cost.",
        category="Memory",
    ),
    FlagDefinition(
        name="--block-size",
        type="choice",
        choices=["8", "16", "32"],
        default="16",
        description="Token block size for contiguous batching. Affects KV cache fragmentation.",
        category="Memory",
    ),
    FlagDefinition(
        name="--enable-prefix-caching",
        type="bool",
        default=False,
        description=(
            "Cache and reuse KV state for shared prompt prefixes. "
            "Speeds up multi-turn conversations and RAG pipelines significantly."
        ),
        category="Memory",
    ),

    # --- Serving ---
    FlagDefinition(
        name="--host",
        type="str",
        default="127.0.0.1",
        description="IP address the API server binds to.",
        category="Serving",
    ),
    FlagDefinition(
        name="--port",
        type="int",
        default=8000,
        description="Port the API server listens on.",
        category="Serving",
    ),
    FlagDefinition(
        name="--max-num-seqs",
        type="int",
        default=256,
        description="Maximum number of sequences processed in a single iteration.",
        category="Serving",
    ),
    FlagDefinition(
        name="--max-num-batched-tokens",
        type="int",
        default=None,
        description=(
            "Maximum total tokens per batch. "
            "Defaults to max_model_len. Reduce for lower latency at the cost of throughput."
        ),
        category="Serving",
    ),
    FlagDefinition(
        name="--disable-log-requests",
        type="bool",
        default=False,
        description="Suppress per-request logging to reduce log noise in production.",
        category="Serving",
    ),
    FlagDefinition(
        name="--api-key",
        type="str",
        default="",
        description="API key required in the Authorization header. Leave blank to disable auth.",
        category="Serving",
    ),

    # --- Parallelism ---
    FlagDefinition(
        name="--tensor-parallel-size",
        type="int",
        default=1,
        description=(
            "Number of GPUs for tensor parallelism. "
            "Set to the number of GPUs for multi-GPU inference."
        ),
        category="Parallelism",
    ),
    FlagDefinition(
        name="--pipeline-parallel-size",
        type="int",
        default=1,
        description=(
            "Number of pipeline stages for pipeline parallelism. "
            "Useful for very large models across multiple nodes."
        ),
        category="Parallelism",
    ),
    FlagDefinition(
        name="--worker-use-ray",
        type="bool",
        default=False,
        description="Use Ray for distributed workers instead of the default multiprocessing.",
        category="Parallelism",
    ),

    # --- Speculative Decoding ---
    FlagDefinition(
        name="--speculative-model",
        type="str",
        default="",
        description=(
            "Draft model for speculative decoding. "
            "Use a smaller model to speed up generation of the main model."
        ),
        category="Performance",
    ),
    FlagDefinition(
        name="--num-speculative-tokens",
        type="int",
        default=None,
        description="Number of tokens the draft model predicts per step.",
        category="Performance",
    ),

    # --- Logging ---
    FlagDefinition(
        name="--log-level",
        type="choice",
        choices=["debug", "info", "warning", "error", "critical"],
        default="info",
        description="Minimum logging level.",
        category="Debug",
    ),
]
