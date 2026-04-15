# LLMManager

A terminal-based control panel for managing LLM server stacks on Linux.

Install, configure, run, and benchmark **Ollama**, **vLLM**, and **LM Studio** from a single TUI — no manual config files or CLI juggling required.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  LLM Manager v0.1.0                          [N]Notifs  [P]Profile  [?]Help │
│─────────────────────────────────────────────────────────────────────────────│
│  [1]Dashboard  [2]Servers  [3]Models  [4]Logs  [5]Bench  [6]Profiles  [7]API│
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Ollama ● Running    vLLM ○ Stopped    LM Studio ○ Not detected            │
│                                                                             │
│  GPU 0: RTX 4090  VRAM [████████████░░░░░░░] 12.3/24.0 GB  Util 34%       │
│  CPU  [████░░░░░░░░░░░░░░░░] 18%   RAM [████████░░░░] 18.2/64.0 GB        │
│─────────────────────────────────────────────────────────────────────────────│
│  GPU: RTX 4090  VRAM 12.3/24GB  CPU: 18%  RAM: 18.2/64GB  Ollama: Running  │
└─────────────────────────────────────────────────────────────────────────────┘
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
- Check for optional GPU tools (nvidia-smi, rocm-smi, xpu-smi) and clipboard support

---

## Manual install

**Requirements:** Linux, Python 3.11+

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

---

## Features

### Server Management
- **Install** Ollama and vLLM with one click — live output streamed to the TUI
- **Start / Stop / Restart** servers from the UI
- **Configure** all server CLI flags with descriptions and type validation
- **Pre-flight checks** before installing (disk space, port availability, CUDA)
- **Auto-start** toggle per server
- **Port conflict detection**
- **Version management** — pin to specific releases

### Model Management
- Browse the **Ollama library** and **HuggingFace Hub** from inside the app
- **Download, load, unload, and delete** models
- **VRAM estimator** — see if a model fits your hardware before downloading
- **Hardware compatibility tier** per model: Comfortable / Limited / Too Large
- Import local GGUF/safetensors files

### Benchmark Suite
- **Throughput** — tokens/sec sustained
- **Latency** — TTFT, p50/p95/p99 per request
- **Memory** — actual vs estimated VRAM delta
- **Concurrency ramp** — 1 → 2 → 4 → 8 → 16 → 32 → 64 → 128 parallel requests
  - Safety cutoff: stops if p99 latency > 30s or error rate > 10%
  - Sustained mode at high concurrency levels
  - Outputs **recommended max concurrency** for your hardware
- **Context scaling** — performance at 1k / 8k / 32k / 128k tokens
- **Quality probes** — standardised prompt sets: coding, reasoning, instruction, chat
- **Benchmark profiles**: Quick (~2 min), Standard (~10 min), Stress (full ramp)
- Results saved to `~/.local/share/llmmanager/benchmarks/` for history comparison

### Live Dashboard
- GPU utilisation and VRAM meters (NVIDIA / AMD / Intel / CPU-only)
- Temperature, power draw, fan speed
- CPU and RAM usage
- Per-server status cards with uptime

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

## GPU support

| Vendor | Tool | Notes |
|--------|------|-------|
| NVIDIA | `pynvml` (auto-installed) | Requires NVIDIA drivers |
| AMD | `rocm-smi` | Requires ROCm installation |
| Intel | `xpu-smi` | Requires Intel oneAPI |
| CPU-only | `psutil` | Always available, shows system RAM |

---

## Supported servers

| Server | Install | Configure | Start/Stop | Models |
|--------|---------|-----------|-----------|--------|
| **Ollama** | ✅ Auto | ✅ Full flags | ✅ | ✅ |
| **vLLM** | ✅ Auto (venv) | ✅ Full flags | ✅ | ✅ |
| **LM Studio** | ⚠️ Manual | ❌ GUI only | ❌ GUI only | ✅ Read |

LM Studio must be started manually; LLMManager detects and connects to its local server automatically.

---

## Development

```sh
git clone https://github.com/jonathjan0397/llmmanager
cd llmmanager
make venv
source .venv/bin/activate
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

GitHub Actions will build and publish to PyPI automatically (requires [trusted publisher](https://docs.pypi.org/trusted-publishers/) setup on pypi.org).

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

[notifications]
low_vram_threshold_pct = 10.0
crash_auto_restart = false
```

---

## License

MIT
