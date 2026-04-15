# LLMManager

A terminal-based control panel for managing LLM server stacks on Linux and Windows.

Install, configure, run, and benchmark **Ollama**, **vLLM**, **LM Studio**, and **llama.cpp** from a single TUI — no manual config files or CLI juggling required.

```
+-----------------------------------------------------------------------------+
|  LLM Manager v0.2.0                          [N]Notifs  [P]Profile  [?]Help |
|-----------------------------------------------------------------------------|
|  [1]Dashboard  [2]Servers  [3]Models  [4]Logs  [5]Bench  [6]Profiles  [7]API|
+-----------------------------------------------------------------------------+
|                                                                             |
|  Ollama * Running    vLLM - Stopped    llama.cpp * Running    LM Studio -   |
|                                                                             |
|  GPU 0: RTX 4090  VRAM [############........] 12.3/24.0 GB  Util 34%       |
|  CPU  [####................] 18%   RAM [########....] 18.2/64.0 GB          |
+-----------------------------------------------------------------------------+
```

---

## Quick install

```sh
curl -fsSL https://raw.githubusercontent.com/jonathjan0397/llmmanager/master/install.sh | sh
```

Then run:

```sh
llmmanager
```

The installer will:
- Detect your distro and install Python 3.11+ if needed
- Install [pipx](https://pipx.pypa.io) if not present
- Install LLMManager into an isolated environment
- Check for optional GPU tools (`nvidia-smi`, `rocm-smi`, `xpu-smi`) and clipboard support

---

## Manual install

**Requirements:** Linux (primary), Windows (supported), Python 3.11+

```sh
# Via pipx (recommended — isolated, adds to PATH)
pipx install git+https://github.com/jonathjan0397/llmmanager

# Via pip (into active venv)
pip install git+https://github.com/jonathjan0397/llmmanager

# From source
git clone https://github.com/jonathjan0397/llmmanager
cd llmmanager
pip install -e .
```

### llama.cpp server (optional)

The llama.cpp backend installs `llama-cpp-python[server]` into its own isolated venv
(`~/.llmmanager-venvs/llamacpp/`) when you click **Install** on the Server Management screen.

The installer auto-detects your GPU and sets the right build flags:

| Hardware | Build backend | What LLMManager does |
|----------|--------------|----------------------|
| NVIDIA CUDA | `LLAMA_CUDA=on` | Detects `nvcc` in PATH, sets `CMAKE_ARGS` automatically |
| AMD ROCm | `LLAMA_HIP=on` | Detects `hipcc` in PATH |
| Apple Silicon | `LLAMA_METAL=on` | Detected on macOS/ARM |
| CPU-only | *(default)* | No extra flags needed |

If you want to manage the build yourself, install `llama-cpp-python[server]` manually into
`~/.llmmanager-venvs/llamacpp/` before hitting Install in the UI — LLMManager will detect it.

### GPU telemetry dependencies

| Vendor | Library | Notes |
|--------|---------|-------|
| NVIDIA | `pynvml` (auto-installed as `nvidia-ml-py`) | Requires NVIDIA drivers |
| AMD | `rocm-smi` CLI | Requires ROCm installation |
| Intel | `xpu-smi` CLI | Requires Intel oneAPI |
| CPU-only | `psutil` | Always available — shows system RAM |

---

## Features

### Server Management
- **Install** Ollama, vLLM, and llama.cpp with one click — live output streamed to the TUI
- **Start / Stop / Restart** servers from the UI
- **Configure** all server CLI flags with descriptions and type validation
- **Pre-flight checks** before starting (disk space, port availability, binary present)
- **Auto-start** toggle per server
- **Port conflict detection**
- **Version management** — pin to specific releases

### Model Management
- Browse the **Ollama library** (gemma4, gemma3, llama4, llama3.3, mistral-small, phi4-mini,
  qwq, deepseek-v3, codegemma, tinyllama, smollm2, command-r, and more) from inside the app
- Browse **HuggingFace Hub** for GGUF and safetensor models
- **Version picker** — when a model has multiple tags (e.g. `7b`, `13b`, `70b`) a dialog
  lets you choose which variant to download before the transfer starts
- **Download, load, unload, and delete** models
- **VRAM estimator** — see if a model fits your hardware before downloading
- **Hardware compatibility tier** per model: Comfortable / Limited / Too Large
- Import local GGUF/safetensors files

### Benchmark Suite
- **Multi-model runs** — select any number of models with checkboxes; LLMManager runs each
  one sequentially and then presents a unified comparison
- **Benchmark categories:**
  - *Throughput* — sustained tokens/sec
  - *Latency* — TTFT, p50/p95/p99 per request
  - *Memory* — actual vs estimated VRAM delta
  - *Concurrency ramp* — 1 → 2 → 4 → 8 → 16 → 32 → 64 → 128 parallel requests
    (stops automatically if p99 latency > 30 s or error rate > 10 %)
  - *Context scaling* — TPS at 1 K / 8 K / 32 K / 128 K token context lengths
  - *Quality probes* — standardised prompt sets: coding, reasoning, instruction, chat
- **Benchmark profiles:** Quick (~1 min/model), Standard (~5 min/model), Stress (full ramp)
- **Comparison charts** — bar charts, concurrency/context scaling tables, and Unicode
  sparklines shown side-by-side for all tested models after a run
- **Report export** — every completed run is saved as a human-readable plain-text file:
  - Location: `~/.local/share/llmmanager/benchmarks/reports/`
  - Filename: `YYYYMMDD_HHMMSS_model1_model2_model3.txt`
  - Contents: per-model summary, full scaling tables, and a side-by-side comparison table
- JSON result files saved to `~/.local/share/llmmanager/benchmarks/` for history and scripting

### Live Dashboard
- Per-server status cards with uptime
- Quick Load widget — dropdown of currently available models with one-click load/unload and
  a refresh button; no manual typing required
- GPU utilisation and VRAM meters (NVIDIA / AMD / Intel / CPU-only)
- Temperature, power draw, fan speed
- CPU and RAM usage

### Chat
- Multi-turn conversation with any loaded model on any running server
- Server and model dropdowns populated live from running instances
- **Enter key** sends the message; Shift+Enter or the Send button also work
- Streaming response rendered in the terminal as tokens arrive

### API Panel
- All active endpoints listed with one-click copy
- Quick inference test — send a prompt, see streaming response and latency

### Profiles
- Save named configuration snapshots (e.g. "Coding", "Chat", "High-throughput")
- Switch between profiles with a single keypress

### Notifications
- Server crash detection with optional auto-restart
- Low VRAM warnings (configurable threshold)
- Download completion alerts

---

## Keyboard shortcuts

| Key | Action |
|-----|--------|
| `1`–`7` | Switch screens |
| `s` | Start selected server |
| `S` | Stop selected server |
| `r` | Restart selected server |
| `i` | Install server |
| `d` | Download selected model |
| `b` | Benchmark selected model |
| `Delete` | Delete selected model (with confirmation) |
| `/` | Search / filter |
| `c` | Copy endpoint URL |
| `n` | Notifications panel |
| `p` | Profile switcher |
| `F5` | Force refresh |
| `F1` / `?` | Help |
| `q` / `F10` | Quit |

---

## Supported servers

| Server | Install | Configure | Start/Stop | Models |
|--------|---------|-----------|------------|--------|
| **Ollama** | Auto | Full flags | Yes | Full (pull / unload / delete) |
| **vLLM** | Auto (venv) | Full flags | Yes | Full |
| **llama.cpp** | Auto (venv, GPU-aware) | Full flags | Yes | Full (GGUF path required) |
| **LM Studio** | Manual (GUI app) | GUI only | GUI only | Read (list loaded) |

LM Studio must be started manually; LLMManager detects and connects to its local server
automatically once it is running.

---

## Configuration

Config lives at `~/.config/llmmanager/config.toml` — human-editable TOML.

```toml
[servers.ollama]
server_type = "ollama"
port = 11434
auto_start = false

[servers.ollama.flags]
keep-alive = "10m"
num-parallel = 4
flash-attention = true

[servers.llamacpp]
server_type = "llamacpp"
port = 8080
auto_start = false

[servers.llamacpp.flags]
model = "/path/to/your/model.gguf"
n-gpu-layers = 35
ctx-size = 4096
cont-batching = true

[notifications]
low_vram_threshold_pct = 10.0
crash_auto_restart = false
```

---

## Development

```sh
git clone https://github.com/jonathjan0397/llmmanager
cd llmmanager
make venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows
make dev        # editable install
make run        # launch the TUI
make check      # lint + typecheck
make test       # run tests
make help       # show all targets
```

---

## Releasing

```sh
git tag v0.2.0
git push origin v0.2.0
```

GitHub Actions will build and publish to PyPI automatically (requires
[trusted publisher](https://docs.pypi.org/trusted-publishers/) setup on pypi.org).

---

## License

MIT
