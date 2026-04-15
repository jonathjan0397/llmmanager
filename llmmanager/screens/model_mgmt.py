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

from llmmanager.models.llm_model import DownloadProgress
from llmmanager.widgets.confirm_dialog import ConfirmDialog

if TYPE_CHECKING:
    from llmmanager.app import LLMManagerApp
    from llmmanager.servers.base import AbstractServer


class ModelManagementScreen(Widget):
    """Screen 3 — model library, downloads, and management."""

    DEFAULT_CSS = """
    ModelManagementScreen { width: 1fr; height: 1fr; }

    #installed-actions {
        height: auto;
        padding: 1 0 0 0;
    }
    #installed-actions Button { width: 1fr; }

    #ollama-actions {
        height: auto;
        padding: 1 0 0 0;
    }
    #ollama-actions Button { width: 1fr; }

    #download-bar {
        height: 1;
        padding: 0 1;
        color: $accent;
        background: $surface-darken-1;
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

    def compose(self) -> ComposeResult:
        with TabbedContent(id="model-tabs"):
            with TabPane("Installed", id="tab-installed"):
                yield Input(placeholder="Filter installed models...", id="installed-search")
                yield DataTable(id="installed-table", cursor_type="row")
                with Horizontal(id="installed-actions"):
                    yield Button("Load",   id="btn-load",   variant="success")
                    yield Button("Unload", id="btn-unload", variant="warning")
                    yield Button("Delete", id="btn-delete", variant="error")

            with TabPane("Ollama Library", id="tab-ollama-lib"):
                yield Input(placeholder="Search Ollama library...", id="ollama-search")
                yield DataTable(id="ollama-table", cursor_type="row")
                with Horizontal(id="ollama-actions"):
                    yield Button("Download Selected", id="btn-ollama-download", variant="primary")
                    yield Button("Refresh",           id="btn-ollama-refresh",  variant="default")

            with TabPane("Local Import", id="tab-local"):
                yield Label("Enter a local file path (GGUF or safetensors):", classes="form-label")
                yield Input(placeholder="/path/to/model.gguf", id="local-path-input")
                yield Button("Import", id="btn-local-import", variant="primary")
                yield Static(
                    "[dim]Local import is not yet supported in this version.[/]",
                    id="local-import-note",
                )

        yield Static("", id="download-bar")

    # ------------------------------------------------------------------
    # Mount
    # ------------------------------------------------------------------

    def on_mount(self) -> None:
        t = self.query_one("#installed-table", DataTable)
        t.add_columns("Name", "Size", "Quant", "VRAM Est.", "Server", "Status")
        self.run_worker(self._load_installed())
        self.run_worker(self._watch_downloads())

    # ------------------------------------------------------------------
    # Tab activation — lazy-load Ollama Library on first visit
    # ------------------------------------------------------------------

    def on_tabbed_content_tab_activated(self, event: TabbedContent.TabActivated) -> None:
        if event.tab.id == "tab-ollama-lib" and not self._ollama_lib_loaded:
            t = self.query_one("#ollama-table", DataTable)
            if not t.columns:
                t.add_columns("Name", "Tags", "Description")
            self.run_worker(self._load_ollama_library())

    # ------------------------------------------------------------------
    # Data loaders
    # ------------------------------------------------------------------

    async def _load_installed(self, query: str = "") -> None:
        app: LLMManagerApp = self.app  # type: ignore[assignment]
        table = self.query_one("#installed-table", DataTable)
        table.clear()
        q = query.lower()
        for server in app.registry.all_enabled():
            try:
                models = await server.list_loaded_models()
                for m in models:
                    if q and q not in m.display_name.lower():
                        continue
                    table.add_row(
                        m.display_name,
                        f"{m.size_gb:.1f} GB" if m.size_gb else "?",
                        m.quantization or "?",
                        f"{m.vram_estimate_mb:.0f} MB" if m.vram_estimate_mb else "?",
                        server.display_name,
                        "[green]Loaded[/]" if m.is_loaded else "On-disk",
                        key=f"{server.name}:{m.model_id}",
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
                bar.update(f"Download complete: {prog.model_id}")
                self.run_worker(self._load_installed())
            elif prog.status == "error":
                bar.update(f"[red]Error downloading {prog.model_id}: {prog.error}[/]")
            else:
                pct = ""
                if prog.total_bytes:
                    pct = f" {prog.downloaded_bytes / prog.total_bytes * 100:.0f}%"
                speed = ""
                if prog.speed_bps and prog.speed_bps > 1024:
                    speed = f"  {prog.speed_bps / 1_048_576:.1f} MB/s"
                bar.update(f"Downloading {prog.model_id}{pct}{speed}")

    # ------------------------------------------------------------------
    # Selection helper
    # ------------------------------------------------------------------

    def _get_installed_selection(self) -> tuple["AbstractServer", str] | None:
        """Return (server, model_id) for the currently highlighted installed row."""
        app: LLMManagerApp = self.app  # type: ignore[assignment]
        table = self.query_one("#installed-table", DataTable)
        if table.cursor_row is None:
            return None
        row = table.get_row_at(table.cursor_row)
        model_name = str(row[0])
        server_display = str(row[4])
        server = next(
            (s for s in app.registry.all_enabled() if s.display_name == server_display),
            None,
        )
        return (server, model_name) if server else None

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
                await self._do_delete_selected()
            case "btn-ollama-download":
                self._enqueue_ollama_download()
            case "btn-ollama-refresh":
                self._ollama_lib_loaded = False
                self.run_worker(self._load_ollama_library(
                    query=self.query_one("#ollama-search", Input).value
                ))
            case "btn-local-import":
                self.notify("Local import is not yet supported.", severity="warning")

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    async def _do_load_selected(self) -> None:
        sel = self._get_installed_selection()
        if not sel:
            self.notify("Select a model first.", severity="warning")
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
            self.notify("Select a model first.", severity="warning")
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
            self.notify("Select a model first.", severity="warning")
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

    def _enqueue_ollama_download(self) -> None:
        app: LLMManagerApp = self.app  # type: ignore[assignment]
        table = self.query_one("#ollama-table", DataTable)
        if table.cursor_row is None:
            self.notify("Select a model first.", severity="warning")
            return
        row = table.get_row_at(table.cursor_row)
        model_name = str(row[0])
        ollama = app.registry.get("ollama")
        if ollama is None:
            self.notify("Ollama is not running.", severity="error")
            return
        app.download_manager.enqueue(ollama, model_name)
        self.notify(f"Queued download: {model_name}")

    # ------------------------------------------------------------------
    # Input filter
    # ------------------------------------------------------------------

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "installed-search":
            self.run_worker(self._load_installed(query=event.value))
        elif event.input.id == "ollama-search":
            self._ollama_lib_loaded = False
            self.run_worker(self._load_ollama_library(query=event.value))

    # ------------------------------------------------------------------
    # Keyboard action handlers
    # ------------------------------------------------------------------

    def action_load_model(self) -> None:
        self.run_worker(self._do_load_selected())

    def action_unload_model(self) -> None:
        self.run_worker(self._do_unload_selected())

    def action_delete_model(self) -> None:
        self.run_worker(self._do_delete_selected())

    def action_download_model(self) -> None:
        self._enqueue_ollama_download()

    def action_focus_search(self) -> None:
        tabs = self.query_one("#model-tabs", TabbedContent)
        if tabs.active == "tab-installed":
            self.query_one("#installed-search", Input).focus()
        elif tabs.active == "tab-ollama-lib":
            self.query_one("#ollama-search", Input).focus()
