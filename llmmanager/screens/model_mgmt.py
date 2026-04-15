"""Model Management screen — browse, download, load, delete models."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widget import Widget
from textual.widgets import (
    Button,
    DataTable,
    Input,
    Label,
    Tab,
    TabbedContent,
    TabPane,
)

from llmmanager.models.llm_model import LLMModel
from llmmanager.widgets.confirm_dialog import ConfirmDialog

if TYPE_CHECKING:
    from llmmanager.app import LLMManagerApp


class ModelManagementScreen(Widget):
    """Screen 3 — model library, downloads, and management."""

    DEFAULT_CSS = "ModelManagementScreen { width: 1fr; height: 1fr; }"

    BINDINGS = [
        ("d",      "download_model", "Download"),
        ("l",      "load_model",     "Load"),
        ("u",      "unload_model",   "Unload"),
        ("delete", "delete_model",   "Delete"),
        ("b",      "benchmark_model","Benchmark"),
        ("/",      "focus_search",   "Search"),
    ]

    def compose(self) -> ComposeResult:
        with TabbedContent(id="model-tabs"):
            with TabPane("Installed", id="tab-installed"):
                yield Input(placeholder="Search models...", id="installed-search")
                yield DataTable(id="installed-table", cursor_type="row")
                with Horizontal(id="installed-actions"):
                    yield Button("Load",      id="btn-load",      variant="success")
                    yield Button("Unload",    id="btn-unload",    variant="warning")
                    yield Button("Benchmark", id="btn-bench",     variant="primary")
                    yield Button("Delete",    id="btn-delete",    variant="error")

            with TabPane("Ollama Library", id="tab-ollama-lib"):
                yield Input(placeholder="Search Ollama library...", id="ollama-search")
                yield DataTable(id="ollama-table", cursor_type="row")
                yield Button("Download", id="btn-ollama-download", variant="primary")

            with TabPane("HuggingFace", id="tab-hf"):
                yield Input(placeholder="Search HuggingFace Hub...", id="hf-search")
                yield DataTable(id="hf-table", cursor_type="row")
                yield Button("Download", id="btn-hf-download", variant="primary")

            with TabPane("Local Import", id="tab-local"):
                yield Label("Enter a local file path (GGUF or safetensors):")
                yield Input(placeholder="/path/to/model.gguf", id="local-path-input")
                yield Button("Import", id="btn-local-import", variant="primary")

        yield Label("", id="model-status-bar", classes="status-bar")

    def on_mount(self) -> None:
        self._setup_installed_table()
        self.run_worker(self._load_installed())

    def _setup_installed_table(self) -> None:
        table = self.query_one("#installed-table", DataTable)
        table.add_columns("Name", "Size", "Quant", "VRAM Est.", "Server", "Status")

    def _setup_ollama_table(self) -> None:
        table = self.query_one("#ollama-table", DataTable)
        if not table.columns:
            table.add_columns("Name", "Tags", "Size", "Description")

    def _setup_hf_table(self) -> None:
        table = self.query_one("#hf-table", DataTable)
        if not table.columns:
            table.add_columns("Repo ID", "Size", "Quantization", "Downloads")

    async def _load_installed(self) -> None:
        app: LLMManagerApp = self.app  # type: ignore[assignment]
        table = self.query_one("#installed-table", DataTable)
        table.clear()
        for server in app.registry.all_enabled():
            try:
                models = await server.list_loaded_models()
                for m in models:
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

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        app: LLMManagerApp = self.app  # type: ignore[assignment]
        match event.button.id:
            case "btn-delete":
                await self._delete_selected()
            case "btn-load":
                await self._load_selected()
            case "btn-ollama-download":
                await self._download_from_ollama()
            case "btn-bench":
                await self._benchmark_selected()
            case "btn-local-import":
                await self._import_local()

    async def _delete_selected(self) -> None:
        app: LLMManagerApp = self.app  # type: ignore[assignment]
        table = self.query_one("#installed-table", DataTable)
        row_key = table.cursor_row
        if row_key is None:
            return
        model_key = table.get_row_at(row_key)[0]
        confirmed = await self.app.push_screen_wait(
            ConfirmDialog(f"Delete {model_key}?", "This cannot be undone.")
        )
        if confirmed:
            # Parse "server:model_id" from row key
            for server in app.registry.all_enabled():
                try:
                    await server.delete_model(str(model_key))
                    self.notify(f"Deleted {model_key}")
                    await self._load_installed()
                    return
                except Exception:
                    pass

    async def _load_selected(self) -> None:
        app: LLMManagerApp = self.app  # type: ignore[assignment]
        table = self.query_one("#installed-table", DataTable)
        if table.cursor_row is None:
            return
        row = table.get_row_at(table.cursor_row)
        model_name = str(row[0])
        server_name = str(row[4])
        server = next((s for s in app.registry.all() if s.display_name == server_name), None)
        if server:
            self.run_worker(self._do_load(server, model_name))

    async def _do_load(self, server, model_id: str) -> None:
        self.notify(f"Loading {model_id}...")
        try:
            await server.load_model(model_id)
            self.notify(f"{model_id} loaded.")
        except Exception as exc:
            self.notify(str(exc), severity="error")

    async def _download_from_ollama(self) -> None:
        app: LLMManagerApp = self.app  # type: ignore[assignment]
        table = self.query_one("#ollama-table", DataTable)
        if table.cursor_row is None:
            return
        row = table.get_row_at(table.cursor_row)
        model_name = str(row[0])
        ollama = app.registry.get("ollama")
        if ollama:
            app.download_manager.enqueue(ollama, model_name)
            self.notify(f"Queued download: {model_name}")

    async def _benchmark_selected(self) -> None:
        self.notify("Switch to the Benchmark screen to run benchmarks.")

    async def _import_local(self) -> None:
        path = self.query_one("#local-path-input", Input).value.strip()
        if not path:
            self.notify("Enter a file path.", severity="warning")
            return
        self.notify(f"Importing {path} — not yet implemented in this version.")

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "installed-search":
            self._filter_installed(event.value)

    def _filter_installed(self, query: str) -> None:
        # Simple filter — hide non-matching rows (Textual DataTable doesn't support
        # native filtering, so we reload with filter applied)
        self.run_worker(self._load_installed())
