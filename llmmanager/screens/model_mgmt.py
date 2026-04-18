"""Model Management screen — browse, download, load, delete models."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widget import Widget
from textual.widgets import (
    Button,
    DataTable,
    Input,
    Label,
    Static,
    TabbedContent,
    TabPane,
)

from llmmanager.models.llm_model import DownloadProgress, ModelSource
from llmmanager.widgets.confirm_dialog import ConfirmDialog
from llmmanager.widgets.version_picker import VersionPickerDialog

if TYPE_CHECKING:
    from llmmanager.app import LLMManagerApp
    from llmmanager.servers.base import AbstractServer


class ModelManagementScreen(Widget):
    """Screen 3 — model library, downloads, and management."""

    DEFAULT_CSS = """
    ModelManagementScreen { width: 1fr; height: 1fr; }

    .model-action-bar {
        height: auto;
        padding: 1 0 0 0;
    }
    .model-action-bar Button { width: 1fr; }

    #disk-usage-bar {
        height: 1;
        padding: 0 1;
        background: $surface-darken-1;
        margin-bottom: 1;
    }

    #download-bar {
        height: 1;
        padding: 0 1;
        color: $accent;
        background: $surface-darken-1;
    }

    /* ---- Credentials tab ---- */
    #creds-scroll {
        width: 1fr;
        height: 1fr;
        padding: 1 2;
    }

    .creds-service-box {
        height: auto;
        border: solid $primary-darken-2;
        padding: 1;
        margin-bottom: 1;
    }

    .creds-service-title {
        text-style: bold;
        color: $primary;
        margin-bottom: 1;
    }

    .creds-status {
        margin-bottom: 1;
    }

    .creds-input-row {
        height: auto;
    }

    .creds-input-row Input {
        width: 1fr;
    }

    .creds-save-btn {
        width: 12;
        margin-left: 1;
    }

    .hf-auth-banner {
        height: auto;
        padding: 0 1;
        color: $text-muted;
        margin-bottom: 1;
    }
    """

    BINDINGS = [
        ("l",      "load_model",    "Load"),
        ("u",      "unload_model",  "Unload"),
        ("d",      "download_model","Download"),
        ("delete", "delete_model",  "Delete"),
        ("/",      "focus_search",  "Search"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._ollama_lib_loaded = False
        self._hf_loaded = False

    def compose(self) -> ComposeResult:
        with TabbedContent(id="model-tabs"):
            # ---- Installed ------------------------------------------------
            with TabPane("Installed", id="tab-installed"):
                yield Static("", id="disk-usage-bar")
                yield Input(placeholder="Filter models...", id="installed-search")
                yield DataTable(id="installed-table", cursor_type="row")
                with Horizontal(classes="model-action-bar"):
                    yield Button("Load",    id="btn-load",            variant="success")
                    yield Button("Unload",  id="btn-unload",          variant="warning")
                    yield Button("Delete",  id="btn-delete",          variant="error")
                    yield Button("Refresh", id="btn-installed-refresh", variant="default")

            # ---- Ollama Library -------------------------------------------
            with TabPane("Ollama Library", id="tab-ollama-lib"):
                yield Static(
                    "[bold green]LOCAL[/]  Models are downloaded and run on your hardware.",
                    classes="hf-auth-banner",
                )
                yield Input(placeholder="Search Ollama library...", id="ollama-search")
                yield DataTable(id="ollama-table", cursor_type="row")
                with Horizontal(classes="model-action-bar"):
                    yield Button("Download Selected", id="btn-ollama-download", variant="primary")
                    yield Button("Refresh",           id="btn-ollama-refresh",  variant="default")

            # ---- HuggingFace ---------------------------------------------
            with TabPane("HuggingFace", id="tab-hf"):
                yield Static("", id="hf-auth-banner", classes="hf-auth-banner")
                yield Input(placeholder="Search HuggingFace GGUF models...", id="hf-search")
                yield DataTable(id="hf-table", cursor_type="row")
                with Horizontal(classes="model-action-bar"):
                    yield Button("Download Selected", id="btn-hf-download", variant="primary")
                    yield Button("Refresh",           id="btn-hf-refresh",  variant="default")

            # ---- Local Import --------------------------------------------
            with TabPane("Local Import", id="tab-local"):
                yield Label("Enter a local file path (GGUF or safetensors):", classes="form-label")
                yield Input(placeholder="/path/to/model.gguf", id="local-path-input")
                yield Button("Import", id="btn-local-import", variant="primary")
                yield Static("[dim]Local import is not yet supported.[/]")

            # ---- Credentials ---------------------------------------------
            with TabPane("Credentials", id="tab-creds"):
                yield Static("", id="creds-save-feedback")
                with Vertical(id="creds-scroll"):
                    # HuggingFace
                    with Vertical(classes="creds-service-box"):
                        yield Label("HuggingFace", classes="creds-service-title")
                        yield Static("", id="creds-hf-status", classes="creds-status")
                        yield Label("API Token  (needed for gated/private models)")
                        with Horizontal(classes="creds-input-row"):
                            yield Input(password=True, placeholder="hf_...", id="creds-hf-input")
                            yield Button("Save", id="btn-save-hf", variant="primary", classes="creds-save-btn")

                    # OpenAI
                    with Vertical(classes="creds-service-box"):
                        yield Label("OpenAI  [bold blue]CLOUD[/]", classes="creds-service-title")
                        yield Static("", id="creds-openai-status", classes="creds-status")
                        yield Label("API Key")
                        with Horizontal(classes="creds-input-row"):
                            yield Input(password=True, placeholder="sk-...", id="creds-openai-input")
                            yield Button("Save", id="btn-save-openai", variant="primary", classes="creds-save-btn")

                    # Anthropic
                    with Vertical(classes="creds-service-box"):
                        yield Label("Anthropic  [bold blue]CLOUD[/]", classes="creds-service-title")
                        yield Static("", id="creds-anthropic-status", classes="creds-status")
                        yield Label("API Key")
                        with Horizontal(classes="creds-input-row"):
                            yield Input(password=True, placeholder="sk-ant-...", id="creds-anthropic-input")
                            yield Button("Save", id="btn-save-anthropic", variant="primary", classes="creds-save-btn")

                    # Groq
                    with Vertical(classes="creds-service-box"):
                        yield Label("Groq  [bold blue]CLOUD[/]", classes="creds-service-title")
                        yield Static("", id="creds-groq-status", classes="creds-status")
                        yield Label("API Key")
                        with Horizontal(classes="creds-input-row"):
                            yield Input(password=True, placeholder="gsk_...", id="creds-groq-input")
                            yield Button("Save", id="btn-save-groq", variant="primary", classes="creds-save-btn")

        yield Static("", id="download-bar")

    # ------------------------------------------------------------------
    # Mount
    # ------------------------------------------------------------------

    def on_mount(self) -> None:
        self.query_one("#installed-table", DataTable).add_columns(
            "Name", "Size", "Context", "Server / Provider", "Type"
        )
        self.query_one("#ollama-table", DataTable).add_columns(
            "Name", "Tags", "Description"
        )
        self.query_one("#hf-table", DataTable).add_columns(
            "Repo ID", "Tags", "Downloads"
        )
        self.run_worker(self._load_installed())
        self.run_worker(self._watch_downloads())
        self._refresh_creds_status()
        self._refresh_hf_banner()

    # ------------------------------------------------------------------
    # Tab activation — lazy-load on first visit
    # ------------------------------------------------------------------

    def on_tabbed_content_tab_activated(self, event: TabbedContent.TabActivated) -> None:
        if event.tab is None:
            return
        if event.tab.id == "tab-ollama-lib" and not self._ollama_lib_loaded:
            self.run_worker(self._load_ollama_library())
        elif event.tab.id == "tab-hf" and not self._hf_loaded:
            self.run_worker(self._load_hf_library())
        elif event.tab.id == "tab-creds":
            self._refresh_creds_status()

    # ------------------------------------------------------------------
    # Data loaders
    # ------------------------------------------------------------------

    async def _load_installed(self, query: str = "") -> None:
        import shutil
        from llmmanager.hub.cloud_models import get_cloud_models
        app: LLMManagerApp = self.app  # type: ignore[assignment]
        table = self.query_one("#installed-table", DataTable)
        table.clear()
        q = query.lower()
        cfg = app.config_manager.config
        total_model_gb = 0.0

        # Local models from running servers
        for server in app.registry.all_enabled():
            try:
                models = await server.list_loaded_models()
                for m in models:
                    if q and q not in m.display_name.lower():
                        continue
                    ctx = f"{m.context_length // 1000}k" if m.context_length else "?"
                    size_gb = m.size_gb or 0.0
                    total_model_gb += size_gb
                    table.add_row(
                        m.display_name,
                        f"{size_gb:.1f} GB" if m.size_gb else "?",
                        ctx,
                        server.display_name,
                        "[bold green]LOCAL[/]",
                        key=f"{server.name}:{m.model_id}",
                    )
            except Exception:
                pass

        # Cloud models (shown when API key is configured)
        cloud = get_cloud_models(
            openai_key=cfg.openai_api_key,
            anthropic_key=cfg.anthropic_api_key,
            groq_key=cfg.groq_api_key,
        )
        for m in cloud:
            if q and q not in m.display_name.lower():
                continue
            ctx = f"{m.context_length // 1000}k" if m.context_length else "?"
            provider = m.source.value.title()
            table.add_row(
                m.display_name,
                "API",
                ctx,
                provider,
                "[bold blue]CLOUD[/]",
                key=f"cloud:{m.source.value}:{m.model_id}",
            )

        # Update disk usage bar
        try:
            import pathlib
            du = shutil.disk_usage(pathlib.Path.home())
            free_gb = du.free / 1024**3
            total_gb = du.total / 1024**3
            used_pct = (du.used / du.total * 100) if du.total else 0
            bar = self.query_one("#disk-usage-bar", Static)
            bar.update(
                f"[dim]Models on disk:[/] [cyan]{total_model_gb:.1f} GB[/]   "
                f"[dim]Disk free:[/] [{'red' if free_gb < 5 else 'green'}]{free_gb:.1f} GB[/]"
                f"[dim] / {total_gb:.0f} GB  ({used_pct:.0f}% used)[/]"
            )
        except Exception:
            pass

    async def _load_ollama_library(self, query: str = "") -> None:
        from llmmanager.hub.ollama_library import search_models
        table = self.query_one("#ollama-table", DataTable)
        table.clear()
        try:
            models = await search_models(query=query, limit=100)
            for m in models:
                tags = ", ".join(m.tags[:5]) if m.tags else "—"
                desc = (m.description[:60] + "…") if len(m.description) > 60 else (m.description or "—")
                table.add_row(m.display_name, tags, desc, key=m.model_id)
            self._ollama_lib_loaded = True
        except Exception as exc:
            self.notify(f"Ollama library error: {exc}", severity="error")

    async def _load_hf_library(self, query: str = "") -> None:
        from llmmanager.hub.huggingface import search_models
        app: LLMManagerApp = self.app  # type: ignore[assignment]
        table = self.query_one("#hf-table", DataTable)
        table.clear()
        try:
            token = app.config_manager.config.hf_token
            models = await search_models(query=query, limit=30, hf_token=token)
            for m in models:
                tags = ", ".join(m.tags[:5]) if m.tags else "—"
                table.add_row(m.display_name, tags, m.description or "—", key=m.model_id)
            self._hf_loaded = True
        except Exception as exc:
            self.notify(f"HuggingFace error: {exc}", severity="error")

    # ------------------------------------------------------------------
    # Download progress watcher
    # ------------------------------------------------------------------

    async def _watch_downloads(self) -> None:
        app: LLMManagerApp = self.app  # type: ignore[assignment]
        q = app.download_manager.progress_queue
        bar = self.query_one("#download-bar", Static)
        while True:
            try:
                prog: DownloadProgress = await asyncio.wait_for(q.get(), timeout=2.0)
            except (asyncio.TimeoutError, TimeoutError):
                continue
            except asyncio.CancelledError:
                return
            if prog.status == "complete":
                bar.update(f"[green]Download complete:[/] {prog.model_id}")
                self.run_worker(self._load_installed())
            elif prog.status == "error":
                bar.update(f"[red]Error:[/] {prog.model_id} — {prog.error}")
            else:
                pct = f" {prog.progress_pct:.0f}%" if prog.total_bytes else ""
                speed = f"  {prog.speed_bps / 1_048_576:.1f} MB/s" if prog.speed_bps and prog.speed_bps > 1024 else ""
                bar.update(f"[yellow]Downloading[/] {prog.model_id}{pct}{speed}")

    # ------------------------------------------------------------------
    # Credentials helpers
    # ------------------------------------------------------------------

    def _refresh_creds_status(self) -> None:
        try:
            app: LLMManagerApp = self.app  # type: ignore[assignment]
            cfg = app.config_manager.config
            _set = "[bold green]✓ Set[/]"
            _unset = "[dim]✗ Not configured[/]"
            self.query_one("#creds-hf-status",       Static).update(_set if cfg.hf_token          else _unset)
            self.query_one("#creds-openai-status",   Static).update(_set if cfg.openai_api_key    else _unset)
            self.query_one("#creds-anthropic-status",Static).update(_set if cfg.anthropic_api_key else _unset)
            self.query_one("#creds-groq-status",     Static).update(_set if cfg.groq_api_key      else _unset)
        except Exception:
            pass

    def _refresh_hf_banner(self) -> None:
        try:
            app: LLMManagerApp = self.app  # type: ignore[assignment]
            cfg = app.config_manager.config
            if cfg.hf_token:
                msg = "[bold green]LOCAL[/]  Authenticated with HuggingFace — gated models accessible."
            else:
                msg = "[bold green]LOCAL[/]  Models run on your hardware.  [dim]No HF token — gated models unavailable. Set one in Credentials.[/]"
            self.query_one("#hf-auth-banner", Static).update(msg)
        except Exception:
            pass

    def _save_credential(self, field: str, value: str) -> None:
        app: LLMManagerApp = self.app  # type: ignore[assignment]
        app.config_manager.update(**{field: value or None})
        self._refresh_creds_status()
        self._refresh_hf_banner()
        # Reload installed to pick up new cloud models
        self.run_worker(self._load_installed())
        # Invalidate HF cache so next visit re-authenticates
        if field == "hf_token":
            self._hf_loaded = False

    # ------------------------------------------------------------------
    # Selection helper
    # ------------------------------------------------------------------

    def _get_installed_selection(self) -> tuple["AbstractServer", str] | None:
        """Return (server, model_id) for the selected installed row. None for cloud/no-selection rows."""
        app: LLMManagerApp = self.app  # type: ignore[assignment]
        table = self.query_one("#installed-table", DataTable)
        if table.cursor_row is None:
            return None
        try:
            row_key, _ = table.coordinate_to_cell_key(table.cursor_coordinate)
            key_str = str(row_key)
        except Exception:
            return None
        # Keys: "server_name:model_id" for local, "cloud:provider:model_id" for cloud
        if key_str.startswith("cloud:"):
            return None
        parts = key_str.split(":", 1)
        if len(parts) != 2:
            return None
        server_name, model_id = parts
        server = app.registry.get(server_name)
        return (server, model_id) if server else None

    # ------------------------------------------------------------------
    # Button handler
    # ------------------------------------------------------------------

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        match event.button.id:
            case "btn-load":
                self.run_worker(self._do_load_selected())
            case "btn-unload":
                self.run_worker(self._do_unload_selected())
            case "btn-delete":
                self.run_worker(self._do_delete_selected())
            case "btn-installed-refresh":
                self.run_worker(self._load_installed())
            case "btn-ollama-download":
                self.run_worker(self._enqueue_ollama_download())
            case "btn-ollama-refresh":
                self._ollama_lib_loaded = False
                self.run_worker(self._load_ollama_library(
                    query=self.query_one("#ollama-search", Input).value
                ))
            case "btn-hf-download":
                self._enqueue_hf_download()
            case "btn-hf-refresh":
                self._hf_loaded = False
                self.run_worker(self._load_hf_library(
                    query=self.query_one("#hf-search", Input).value
                ))
            case "btn-local-import":
                self.notify("Local import is not yet supported.", severity="warning")
            case "btn-save-hf":
                self._save_credential("hf_token", self.query_one("#creds-hf-input", Input).value.strip())
                self.notify("HuggingFace token saved.")
            case "btn-save-openai":
                self._save_credential("openai_api_key", self.query_one("#creds-openai-input", Input).value.strip())
                self.notify("OpenAI API key saved.")
            case "btn-save-anthropic":
                self._save_credential("anthropic_api_key", self.query_one("#creds-anthropic-input", Input).value.strip())
                self.notify("Anthropic API key saved.")
            case "btn-save-groq":
                self._save_credential("groq_api_key", self.query_one("#creds-groq-input", Input).value.strip())
                self.notify("Groq API key saved.")

    # ------------------------------------------------------------------
    # Local model actions
    # ------------------------------------------------------------------

    async def _do_load_selected(self) -> None:
        sel = self._get_installed_selection()
        if not sel:
            self.notify("Select a local model to load.", severity="warning")
            return
        server, model_id = sel
        self.notify(f"Loading {model_id}…")
        try:
            await server.load_model(model_id)
            self.notify(f"{model_id} loaded.")
            await self._load_installed()
        except Exception as exc:
            self.notify(str(exc), severity="error")

    async def _do_unload_selected(self) -> None:
        sel = self._get_installed_selection()
        if not sel:
            self.notify("Select a local model to unload.", severity="warning")
            return
        server, model_id = sel
        try:
            await server.unload_model(model_id)
            self.notify(f"{model_id} unloaded.")
            await self._load_installed()
        except Exception as exc:
            self.notify(str(exc), severity="error")

    async def _do_delete_selected(self) -> None:
        sel = self._get_installed_selection()
        if not sel:
            self.notify("Select a local model to delete.", severity="warning")
            return
        server, model_id = sel
        confirmed = await self.app.push_screen_wait(
            ConfirmDialog(f"Delete {model_id}?", "This cannot be undone.")
        )
        if confirmed:
            try:
                await server.delete_model(model_id)
                self.notify(f"Deleted {model_id}.")
                await self._load_installed()
            except Exception as exc:
                self.notify(str(exc), severity="error")

    async def _enqueue_ollama_download(self) -> None:
        app: LLMManagerApp = self.app  # type: ignore[assignment]
        table = self.query_one("#ollama-table", DataTable)
        if table.cursor_row is None:
            self.notify("Select a model first.", severity="warning")
            return
        row = table.get_row_at(table.cursor_row)
        model_name = str(row[0])
        # Parse tags from column 1 (stored as "1b, 3b, 7b, ...")
        tags_raw = str(row[1]) if len(row) > 1 else ""
        tags = [t.strip() for t in tags_raw.split(",") if t.strip() and t.strip() != "—"]

        ollama = app.registry.get("ollama")
        if ollama is None:
            self.notify("Ollama server is not configured.", severity="error")
            return

        if len(tags) > 1:
            chosen_tag = await self.app.push_screen_wait(
                VersionPickerDialog(model_name, tags)
            )
            if chosen_tag is None:
                return  # cancelled
            full_name = f"{model_name}:{chosen_tag}"
        elif len(tags) == 1:
            full_name = f"{model_name}:{tags[0]}"
        else:
            full_name = model_name  # fallback — Ollama will use :latest

        app.download_manager.enqueue(ollama, full_name)
        self.notify(f"Queued: {full_name}")

    def _enqueue_hf_download(self) -> None:
        app: LLMManagerApp = self.app  # type: ignore[assignment]
        table = self.query_one("#hf-table", DataTable)
        if table.cursor_row is None:
            self.notify("Select a model first.", severity="warning")
            return
        repo_id = str(table.get_row_at(table.cursor_row)[0])
        ollama = app.registry.get("ollama")
        if ollama is None:
            self.notify("Ollama is not running.", severity="error")
            return
        app.download_manager.enqueue(ollama, f"hf.co/{repo_id}")
        self.notify(f"Queued: {repo_id}")

    # ------------------------------------------------------------------
    # Input filter
    # ------------------------------------------------------------------

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "installed-search":
            self.run_worker(self._load_installed(query=event.value))
        elif event.input.id == "ollama-search":
            self._ollama_lib_loaded = False
            self.run_worker(self._load_ollama_library(query=event.value))
        elif event.input.id == "hf-search":
            self._hf_loaded = False
            self.run_worker(self._load_hf_library(query=event.value))

    # ------------------------------------------------------------------
    # Keyboard actions
    # ------------------------------------------------------------------

    def action_load_model(self) -> None:
        self.run_worker(self._do_load_selected())

    def action_unload_model(self) -> None:
        self.run_worker(self._do_unload_selected())

    def action_delete_model(self) -> None:
        self.run_worker(self._do_delete_selected())

    def action_download_model(self) -> None:
        tabs = self.query_one("#model-tabs", TabbedContent)
        if tabs.active == "tab-ollama-lib":
            self.run_worker(self._enqueue_ollama_download())
        elif tabs.active == "tab-hf":
            self._enqueue_hf_download()

    def action_focus_search(self) -> None:
        tabs = self.query_one("#model-tabs", TabbedContent)
        mapping = {
            "tab-installed":  "#installed-search",
            "tab-ollama-lib": "#ollama-search",
            "tab-hf":         "#hf-search",
        }
        search_id = mapping.get(tabs.active)
        if search_id:
            self.query_one(search_id, Input).focus()
