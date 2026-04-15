"""API Panel screen — live endpoints, copy, and quick inference test."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import Button, Input, Label, Select, TextArea

from llmmanager.widgets.endpoint_badge import EndpointBadge
from llmmanager.widgets.log_view import LogView

if TYPE_CHECKING:
    from llmmanager.app import LLMManagerApp


class APIPanelScreen(Screen):
    """Screen 7 — active endpoints and quick inference test."""

    BINDINGS = [("r", "refresh_endpoints", "Refresh")]

    def compose(self) -> ComposeResult:
        yield Label("Active Endpoints", classes="section-heading")
        with VerticalScroll(id="endpoints-scroll"):
            yield Label("Loading...", id="endpoints-placeholder")

        yield Label("Quick Inference Test", classes="section-heading")
        with Horizontal(id="infer-controls"):
            yield Label("Server:")
            yield Select(
                options=[("Ollama", "ollama"), ("vLLM", "vllm"), ("LM Studio", "lmstudio")],
                value="ollama",
                id="infer-server-select",
            )
            yield Label("Model:")
            yield Input(placeholder="model ID", id="infer-model-input")

        yield Input(
            placeholder="Enter your prompt here...",
            id="infer-prompt-input",
        )
        with Horizontal(id="infer-actions"):
            yield Button("Send", id="btn-send-infer", variant="primary")
            yield Label("", id="infer-latency")

        yield LogView(max_lines=100, id="infer-response")

    def on_mount(self) -> None:
        self.run_worker(self._load_endpoints())

    async def _load_endpoints(self) -> None:
        app: LLMManagerApp = self.app  # type: ignore[assignment]
        scroll = self.query_one("#endpoints-scroll", VerticalScroll)
        scroll.remove_children()

        any_endpoints = False
        for server in app.registry.all_enabled():
            try:
                endpoints = await server.get_endpoints()
                if not endpoints:
                    continue
                any_endpoints = True
                scroll.mount(Label(
                    f"{server.display_name}",
                    classes="section-heading",
                ))
                for ep in endpoints:
                    scroll.mount(EndpointBadge(ep))
            except Exception:
                pass

        if not any_endpoints:
            scroll.mount(Label("No running servers detected."))

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-send-infer":
            await self._run_inference()

    async def _run_inference(self) -> None:
        app: LLMManagerApp = self.app  # type: ignore[assignment]
        server_type = str(self.query_one("#infer-server-select", Select).value)
        model_id = self.query_one("#infer-model-input", Input).value.strip()
        prompt = self.query_one("#infer-prompt-input", Input).value.strip()

        if not model_id or not prompt:
            self.notify("Enter both a model ID and a prompt.", severity="warning")
            return

        server = app.registry.get(server_type)
        if server is None:
            self.notify(f"Server '{server_type}' not configured.", severity="error")
            return

        response_log = self.query_one("#infer-response", LogView)
        response_log.clear_log()
        latency_label = self.query_one("#infer-latency", Label)
        latency_label.update("Running...")

        start = time.monotonic()
        try:
            async for token in server.quick_infer(model_id, prompt):
                response_log.append_line(token)
        except Exception as exc:
            response_log.append_line(f"ERROR: {exc}")

        elapsed_ms = (time.monotonic() - start) * 1000
        latency_label.update(f"Latency: {elapsed_ms:.0f} ms")

    def action_refresh_endpoints(self) -> None:
        self.run_worker(self._load_endpoints())
