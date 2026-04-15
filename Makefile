# LLMManager — development workflow
# Requires: Python 3.11+, pipx, and optionally uv

.DEFAULT_GOAL := help
PYTHON        := python3
PIPX          := pipx
APP           := llmmanager
SRC           := llmmanager
TESTS         := tests

# ── Colours ────────────────────────────────────────────────────────────────────
BOLD  := \033[1m
RESET := \033[0m
GREEN := \033[0;32m
CYAN  := \033[0;36m

.PHONY: help
help: ## Show this help message
	@printf "\n$(BOLD)LLMManager — make targets$(RESET)\n\n"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  $(CYAN)%-18s$(RESET) %s\n", $$1, $$2}'
	@printf "\n"

# ── Installation ───────────────────────────────────────────────────────────────

.PHONY: install
install: ## Install via pipx (isolated, adds llmmanager to PATH)
	$(PIPX) install . --force
	@printf "$(GREEN)Installed. Run: llmmanager$(RESET)\n"

.PHONY: install-uv
install-uv: ## Install via uv tool (faster alternative to pipx)
	uv tool install . --force
	@printf "$(GREEN)Installed via uv. Run: llmmanager$(RESET)\n"

.PHONY: dev
dev: ## Install in editable mode for development (into current venv)
	$(PYTHON) -m pip install -e ".[dev]"
	@printf "$(GREEN)Dev install complete. Run: $(APP)$(RESET)\n"

.PHONY: venv
venv: ## Create a local .venv for development
	$(PYTHON) -m venv .venv
	@printf "$(GREEN)Created .venv. Activate with: source .venv/bin/activate$(RESET)\n"

.PHONY: upgrade
upgrade: ## Upgrade an existing pipx installation
	$(PIPX) upgrade $(APP)

.PHONY: uninstall
uninstall: ## Remove the pipx installation
	$(PIPX) uninstall $(APP)

# ── Running ────────────────────────────────────────────────────────────────────

.PHONY: run
run: ## Run the TUI app directly (requires dev install or active venv)
	$(PYTHON) -m $(APP)

.PHONY: run-dev
run-dev: dev ## Install dev deps then run
	$(PYTHON) -m $(APP)

# ── Code quality ───────────────────────────────────────────────────────────────

.PHONY: lint
lint: ## Run ruff linter
	ruff check $(SRC) $(TESTS)

.PHONY: lint-fix
lint-fix: ## Run ruff linter with auto-fix
	ruff check --fix $(SRC) $(TESTS)

.PHONY: format
format: ## Format code with ruff
	ruff format $(SRC) $(TESTS)

.PHONY: typecheck
typecheck: ## Run mypy type checker
	mypy $(SRC)

.PHONY: check
check: lint typecheck ## Run all checks (lint + typecheck)

# ── Tests ──────────────────────────────────────────────────────────────────────

.PHONY: test
test: ## Run test suite
	pytest $(TESTS) -v

.PHONY: test-unit
test-unit: ## Run unit tests only
	pytest $(TESTS)/unit -v

.PHONY: test-integration
test-integration: ## Run integration tests only (requires running servers)
	pytest $(TESTS)/integration -v

.PHONY: test-cov
test-cov: ## Run tests with coverage report
	pytest $(TESTS) --cov=$(SRC) --cov-report=term-missing --cov-report=html

# ── Build & publish ────────────────────────────────────────────────────────────

.PHONY: build
build: ## Build wheel and sdist
	$(PYTHON) -m pip install --quiet hatch
	hatch build

.PHONY: publish-test
publish-test: build ## Publish to TestPyPI
	$(PYTHON) -m pip install --quiet twine
	twine upload --repository testpypi dist/*

.PHONY: publish
publish: build ## Publish to PyPI (use CI instead for releases)
	$(PYTHON) -m pip install --quiet twine
	twine upload dist/*

# ── Utilities ──────────────────────────────────────────────────────────────────

.PHONY: clean
clean: ## Remove build artifacts, caches, and coverage reports
	rm -rf dist/ build/ *.egg-info .eggs/
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	rm -rf htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

.PHONY: deps-check
deps-check: ## Check for outdated dependencies
	$(PYTHON) -m pip list --outdated

.PHONY: system-deps
system-deps: ## Check optional system dependencies (GPU tools, clipboard)
	@printf "$(BOLD)Checking system dependencies...$(RESET)\n"
	@command -v nvidia-smi  >/dev/null 2>&1 && printf "  $(GREEN)✓$(RESET) nvidia-smi\n"  || printf "  ✗ nvidia-smi (optional — NVIDIA GPU monitoring)\n"
	@command -v rocm-smi    >/dev/null 2>&1 && printf "  $(GREEN)✓$(RESET) rocm-smi\n"    || printf "  ✗ rocm-smi (optional — AMD GPU monitoring)\n"
	@command -v xpu-smi     >/dev/null 2>&1 && printf "  $(GREEN)✓$(RESET) xpu-smi\n"     || printf "  ✗ xpu-smi (optional — Intel GPU monitoring)\n"
	@command -v xclip       >/dev/null 2>&1 && printf "  $(GREEN)✓$(RESET) xclip\n"       || printf "  ✗ xclip (optional — clipboard support)\n"
	@command -v ollama      >/dev/null 2>&1 && printf "  $(GREEN)✓$(RESET) ollama\n"      || printf "  ✗ ollama (install via llmmanager setup wizard)\n"
	@printf "\n"

.PHONY: version
version: ## Show current version
	@$(PYTHON) -c "import llmmanager; print(llmmanager.__version__)"
