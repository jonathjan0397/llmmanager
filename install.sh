#!/usr/bin/env sh
# LLMManager installer
# Usage: curl -fsSL https://raw.githubusercontent.com/jonathjan0397/llmmanager/master/install.sh | sh
set -e

REPO="https://github.com/jonathjan0397/llmmanager"
MIN_PYTHON_MINOR=11
APP_NAME="llmmanager"

# ── Colours ────────────────────────────────────────────────────────────────────
if [ -t 1 ]; then
    RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
    CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'
else
    RED=''; GREEN=''; YELLOW=''; CYAN=''; BOLD=''; RESET=''
fi

info()    { printf "${CYAN}[llmmanager]${RESET} %s\n" "$*"; }
success() { printf "${GREEN}[llmmanager]${RESET} %s\n" "$*"; }
warn()    { printf "${YELLOW}[llmmanager] WARNING:${RESET} %s\n" "$*"; }
die()     { printf "${RED}[llmmanager] ERROR:${RESET} %s\n" "$*" >&2; exit 1; }

# ── Detect distro ──────────────────────────────────────────────────────────────
detect_distro() {
    if [ -f /etc/os-release ]; then
        # shellcheck disable=SC1091
        . /etc/os-release
        echo "${ID:-unknown}"
    elif command -v lsb_release >/dev/null 2>&1; then
        lsb_release -si | tr '[:upper:]' '[:lower:]'
    else
        echo "unknown"
    fi
}

detect_pkg_manager() {
    if   command -v apt-get >/dev/null 2>&1; then echo "apt"
    elif command -v dnf     >/dev/null 2>&1; then echo "dnf"
    elif command -v pacman  >/dev/null 2>&1; then echo "pacman"
    elif command -v zypper  >/dev/null 2>&1; then echo "zypper"
    else echo "unknown"
    fi
}

# ── Python version check ───────────────────────────────────────────────────────
get_python_minor() {
    # Returns the minor version of the best available python3.x binary
    for cmd in python3.13 python3.12 python3.11 python3; do
        if command -v "$cmd" >/dev/null 2>&1; then
            minor=$("$cmd" -c "import sys; print(sys.version_info.minor)" 2>/dev/null || echo 0)
            major=$("$cmd" -c "import sys; print(sys.version_info.major)" 2>/dev/null || echo 0)
            if [ "$major" -eq 3 ] && [ "$minor" -ge "$MIN_PYTHON_MINOR" ]; then
                echo "$cmd"
                return 0
            fi
        fi
    done
    echo ""
}

install_python_apt() {
    info "Installing Python 3.11 via deadsnakes PPA..."
    sudo apt-get update -qq
    sudo apt-get install -y software-properties-common
    sudo add-apt-repository -y ppa:deadsnakes/ppa
    sudo apt-get update -qq
    sudo apt-get install -y python3.11 python3.11-venv python3.11-distutils
    success "Python 3.11 installed."
}

install_python_dnf() {
    info "Installing Python 3.11 via dnf..."
    sudo dnf install -y python3.11 python3.11-pip
    success "Python 3.11 installed."
}

install_python_pacman() {
    info "Installing Python via pacman (Arch always ships latest)..."
    sudo pacman -Sy --noconfirm python
    success "Python installed."
}

install_python_pyenv() {
    warn "Could not install Python automatically for your distro."
    info "Install pyenv and Python 3.11+ manually:"
    printf "  curl https://pyenv.run | bash\n"
    printf "  pyenv install 3.11\n"
    printf "  pyenv global 3.11\n"
    die "Please install Python 3.11+ and re-run this script."
}

ensure_python() {
    PYTHON_CMD=$(get_python_minor)
    if [ -n "$PYTHON_CMD" ]; then
        VER=$("$PYTHON_CMD" --version 2>&1)
        success "Found $VER ($PYTHON_CMD)"
        return 0
    fi

    warn "Python 3.${MIN_PYTHON_MINOR}+ not found. Attempting to install..."
    PKG_MGR=$(detect_pkg_manager)
    case "$PKG_MGR" in
        apt)    install_python_apt ;;
        dnf)    install_python_dnf ;;
        pacman) install_python_pacman ;;
        *)      install_python_pyenv ;;
    esac

    PYTHON_CMD=$(get_python_minor)
    [ -n "$PYTHON_CMD" ] || die "Python 3.${MIN_PYTHON_MINOR}+ still not found after install attempt."
    success "Python ready: $("$PYTHON_CMD" --version 2>&1)"
}

# ── pipx ───────────────────────────────────────────────────────────────────────
ensure_pipx() {
    if command -v pipx >/dev/null 2>&1; then
        success "pipx $(pipx --version) already installed."
        return 0
    fi

    info "Installing pipx..."
    PKG_MGR=$(detect_pkg_manager)
    case "$PKG_MGR" in
        apt)
            sudo apt-get install -y pipx 2>/dev/null || \
                "$PYTHON_CMD" -m pip install --user pipx
            ;;
        dnf)
            sudo dnf install -y pipx 2>/dev/null || \
                "$PYTHON_CMD" -m pip install --user pipx
            ;;
        pacman)
            sudo pacman -Sy --noconfirm python-pipx 2>/dev/null || \
                "$PYTHON_CMD" -m pip install --user pipx
            ;;
        *)
            "$PYTHON_CMD" -m pip install --user pipx
            ;;
    esac

    # Ensure pipx is on PATH
    "$PYTHON_CMD" -m pipx ensurepath 2>/dev/null || true

    # Reload PATH for this script
    export PATH="$HOME/.local/bin:$PATH"

    command -v pipx >/dev/null 2>&1 || die "pipx install failed. Try: pip install --user pipx"
    success "pipx $(pipx --version) installed."
}

# ── Install llmmanager ─────────────────────────────────────────────────────────
install_app() {
    info "Installing ${APP_NAME}..."
    if pipx list 2>/dev/null | grep -q "$APP_NAME"; then
        info "Upgrading existing installation..."
        pipx upgrade "$APP_NAME" 2>/dev/null || \
            pipx install "git+${REPO}" --force
    else
        pipx install "git+${REPO}"
    fi
    success "${APP_NAME} installed."
}

# ── Optional system deps ───────────────────────────────────────────────────────
check_system_deps() {
    printf "\n${BOLD}Optional system dependencies:${RESET}\n"
    PKG_MGR=$(detect_pkg_manager)

    # Clipboard support
    if ! command -v xclip >/dev/null 2>&1 && ! command -v xdotool >/dev/null 2>&1; then
        warn "xclip not found — clipboard copy in API panel will be limited."
        case "$PKG_MGR" in
            apt)    printf "  Fix: sudo apt-get install -y xclip\n" ;;
            dnf)    printf "  Fix: sudo dnf install -y xclip\n" ;;
            pacman) printf "  Fix: sudo pacman -S xclip\n" ;;
        esac
    else
        success "Clipboard support: OK"
    fi

    # NVIDIA GPU
    if command -v nvidia-smi >/dev/null 2>&1; then
        success "NVIDIA GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
    else
        info "NVIDIA GPU: not detected (install NVIDIA drivers if you have one)"
    fi

    # AMD GPU
    if command -v rocm-smi >/dev/null 2>&1; then
        success "AMD GPU (ROCm): OK"
    else
        info "AMD GPU: rocm-smi not found (install ROCm if you have an AMD GPU)"
        printf "  https://rocm.docs.amd.com/en/latest/deploy/linux/index.html\n"
    fi

    # Intel GPU
    if command -v xpu-smi >/dev/null 2>&1; then
        success "Intel GPU: OK"
    else
        info "Intel GPU: xpu-smi not found (install Intel oneAPI if you have an Intel GPU)"
    fi

    # Disk space
    FREE_GB=$(df "$HOME" | awk 'NR==2 {printf "%.0f", $4/1024/1024}')
    if [ "$FREE_GB" -ge 20 ]; then
        success "Disk space: ${FREE_GB} GB free"
    else
        warn "Disk space: only ${FREE_GB} GB free — large models need 5–50+ GB"
    fi
}

# ── Shell completion hint ──────────────────────────────────────────────────────
print_completion() {
    SHELL_NAME=$(basename "${SHELL:-sh}")
    printf "\n${BOLD}Shell:${RESET} %s\n" "$SHELL_NAME"
    case "$SHELL_NAME" in
        bash)
            printf "  Add to ~/.bashrc:  export PATH=\"\$HOME/.local/bin:\$PATH\"\n" ;;
        zsh)
            printf "  Add to ~/.zshrc:   export PATH=\"\$HOME/.local/bin:\$PATH\"\n" ;;
        fish)
            printf "  Run:               fish_add_path ~/.local/bin\n" ;;
    esac
}

# ── Main ───────────────────────────────────────────────────────────────────────
main() {
    printf "\n${BOLD}${CYAN}LLMManager Installer${RESET}\n"
    printf "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"

    DISTRO=$(detect_distro)
    PKG_MGR=$(detect_pkg_manager)
    info "Distro: ${DISTRO}  |  Package manager: ${PKG_MGR}"

    ensure_python
    ensure_pipx
    install_app
    check_system_deps
    print_completion

    printf "\n${BOLD}${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}\n"
    success "Installation complete!"
    printf "${BOLD}Run:${RESET} ${CYAN}llmmanager${RESET}\n\n"
    printf "If the command is not found, open a new terminal or run:\n"
    printf "  ${CYAN}export PATH=\"\$HOME/.local/bin:\$PATH\"${RESET}\n\n"
}

main "$@"
