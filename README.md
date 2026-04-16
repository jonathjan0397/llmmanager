# LLMManager

A terminal-based control panel for managing LLM server stacks on Linux and Windows.

Install, configure, run, and benchmark **Ollama**, **vLLM**, **LM Studio**, and **llama.cpp** from a single TUI — no manual config files or CLI juggling required.

```
+-----------------------------------------------------------------------------+
|  LLM Manager v0.1.1                          [N]Notifs  [P]Profile  [?]Help |
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
| **Ollama** | Auto | Full flags | Yes | Full (pull / load / unload / delete) |
| **vLLM** | Auto (venv) | Full flags | Yes | Full |
| **llama.cpp** | Auto (venv, GPU-aware) | Full flags | Yes | Full (GGUF path required) |
| **LM Studio** | Manual (GUI app) | Connection only | GUI only | List all / load / unload |

---

## Server notes

### Ollama

- **Install / Uninstall** uses `sudo` on Linux. LLMManager will prompt for your sudo password
  in-app — it is never stored.
- **Pre-loading**: after Start or Restart, LLMManager sends a short warm-up prompt to the
  default model so the first real request is fast.
- **keep-alive**: set `keep-alive = "0"` in flags to unload models from VRAM immediately after
  each request; useful when you share the GPU with other workloads.

### vLLM

- **Runs in an isolated venv** at `~/.local/share/llmmanager/venvs/vllm/` — it will not
  conflict with other Python environments on your system.
- **One model at a time**: vLLM loads a single model at startup via the `--model` flag. Set
  your model in the Model field before clicking Start.
- **Model visibility**: LLMManager shows models from three sources: the running server,
  the `--model` flag in your saved config, and your local HuggingFace cache
  (`~/.cache/huggingface/hub/`). If a model appears in the list but is greyed out it is
  cached but not currently loaded.
- **CUDA required** for GPU inference. CPU-only inference is possible but very slow.
- **HuggingFace token**: if a model requires authentication, set `HF_TOKEN` in your
  environment before launching LLMManager, or pass it via the `--tokenizer` / env flags.

### llama.cpp

- **GGUF path required**: set the full path to a `.gguf` file in the Model field before
  clicking Start. The server will not start without it.
- **GPU layers (`--n-gpu-layers`)**: set to `-1` to offload all layers, or a specific number
  to keep part of the model in VRAM and the rest in RAM (useful for models larger than your VRAM).
- **Context size (`--ctx-size`)**: defaults to 512; increase to 4096–32768 for longer
  conversations. Larger contexts use more VRAM.
- **Continuous batching**: enable `--cont-batching` for better throughput when running multiple
  concurrent requests.
- **Build flags** are auto-detected at install time (CUDA / ROCm / Metal / CPU). If you
  upgrade your GPU drivers after installing, re-install llama.cpp from the Server Management
  screen to rebuild with the correct flags.

### LM Studio

LM Studio is a GUI desktop application. LLMManager **cannot install, start, or stop it** —
these controls are disabled when LM Studio is selected.

**Setup:**
1. Download and install LM Studio from [lmstudio.ai](https://lmstudio.ai)
2. Open LM Studio → go to **Local Server** (the `<->` icon in the left sidebar)
3. Click **Start Server**
4. *(Optional)* Set an API key under Local Server settings if you want to secure access

**In LLMManager:**
- Go to **Servers → LM Studio**
- Set the **port** if you changed it from the default (1234)
- Set the **API key** if you enabled one in LM Studio
- Click **Save & Poll** — LLMManager will verify the connection and show what models are loaded
- LM Studio will then appear as **Running** on the Dashboard and be available in Chat,
  Benchmarks, and the API panel

**Model loading**: use the model picker on the Server Management screen or the Quick Load
widget on the Dashboard. LLMManager sends load/unload requests to LM Studio's local API —
you do not need to use the LM Studio GUI to switch models.

**Polling**: LLMManager polls LM Studio every 2 seconds (same as other servers). If LM Studio
is closed, the Dashboard card will switch to Stopped automatically.

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
git tag v0.1.1
git push origin v0.1.1
```

GitHub Actions will build and publish to PyPI automatically (requires
[trusted publisher](https://docs.pypi.org/trusted-publishers/) setup on pypi.org).

---

## License

MIT
