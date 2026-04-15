"""Chat screen — interactive conversation with loaded models."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.widget import Widget
from textual.widgets import Button, Input, Label, Select, Static

if TYPE_CHECKING:
    from llmmanager.app import LLMManagerApp


class ChatScreen(Widget):
    """Screen 8 — stream chat responses directly from any loaded model."""

    DEFAULT_CSS = """
    ChatScreen { width: 1fr; height: 1fr; }

    #chat-toolbar {
        height: auto;
        padding: 0 1;
        background: $surface-darken-1;
        border-bottom: solid $primary-darken-2;
    }

    #chat-toolbar Label { margin: 1 1 0 0; }
    #chat-toolbar Select { width: 22; margin-right: 1; }
    #btn-chat-clear { width: 10; margin-left: 1; }

    #chat-history {
        width: 1fr;
        height: 1fr;
        padding: 1 2;
    }

    .msg-user {
        padding: 0 1 1 1;
        margin-bottom: 1;
        border-left: thick $accent;
        background: $surface-darken-1;
        width: 100%;
    }

    .msg-user-label {
        color: $accent;
        text-style: bold;
        margin-bottom: 0;
    }

    .msg-assistant {
        padding: 0 1 1 1;
        margin-bottom: 1;
        border-left: thick $success;
        width: 100%;
    }

    .msg-assistant-label {
        color: $success;
        text-style: bold;
        margin-bottom: 0;
    }

    .msg-error {
        padding: 0 1;
        margin-bottom: 1;
        color: $error;
        border-left: thick $error;
        width: 100%;
    }

    #chat-input-bar {
        height: auto;
        padding: 1 1 0 1;
        border-top: solid $primary-darken-2;
    }

    #chat-input { width: 1fr; }
    #btn-chat-send { width: 10; margin-left: 1; }

    #chat-status {
        height: 1;
        padding: 0 1;
        color: $text-muted;
        background: $surface-darken-1;
    }
    """

    BINDINGS = [
        ("ctrl+l", "clear_chat",    "Clear"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._conversation: list[dict] = []
        self._streaming = False
        self._system_prompt = "You are a helpful AI assistant."

    def compose(self) -> ComposeResult:
        with Horizontal(id="chat-toolbar"):
            yield Label("Server:")
            yield Select(
                options=[("Ollama", "ollama"), ("vLLM", "vllm")],
                value="ollama",
                id="chat-server-select",
            )
            yield Label("Model:")
            yield Select(
                options=[("—", "__none__")],
                value="__none__",
                id="chat-model-select",
            )
            yield Button("Clear", id="btn-chat-clear", variant="default")

        yield VerticalScroll(id="chat-history")
        yield Static("Ready", id="chat-status")

        with Horizontal(id="chat-input-bar"):
            yield Input(
                placeholder="Type a message and press Enter…",
                id="chat-input",
            )
            yield Button("Send", id="btn-chat-send", variant="primary")

    def on_mount(self) -> None:
        self.run_worker(self._populate_model_select())

    # ------------------------------------------------------------------
    # Model select population
    # ------------------------------------------------------------------

    async def _populate_model_select(self) -> None:
        app: LLMManagerApp = self.app  # type: ignore[assignment]
        server_type = str(self.query_one("#chat-server-select", Select).value)
        server = app.registry.get(server_type)
        select = self.query_one("#chat-model-select", Select)
        if server is None:
            return
        try:
            models = await server.list_loaded_models()
        except Exception:
            models = []

        if models:
            select.set_options([(m.display_name, m.model_id) for m in models])
        else:
            select.set_options([("No models loaded", "__none__")])

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "chat-server-select":
            self.run_worker(self._populate_model_select())

    # ------------------------------------------------------------------
    # Message sending
    # ------------------------------------------------------------------

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        match event.button.id:
            case "btn-chat-send":
                await self._send()
            case "btn-chat-clear":
                self.action_clear_chat()

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "chat-input":
            await self._send()

    async def _send(self) -> None:
        if self._streaming:
            return

        app: LLMManagerApp = self.app  # type: ignore[assignment]
        inp = self.query_one("#chat-input", Input)
        text = inp.value.strip()
        if not text:
            return

        model_id = str(self.query_one("#chat-model-select", Select).value)
        if model_id == "__none__" or not model_id:
            self.notify("Select a model first.", severity="warning")
            return

        server_type = str(self.query_one("#chat-server-select", Select).value)
        server = app.registry.get(server_type)
        if server is None:
            self.notify("Server not available.", severity="error")
            return

        inp.value = ""
        inp.disabled = True
        self.query_one("#btn-chat-send", Button).disabled = True
        self._streaming = True

        self._conversation.append({"role": "user", "content": text})

        history = self.query_one("#chat-history", VerticalScroll)
        status = self.query_one("#chat-status", Static)

        # User bubble
        user_label = Static("You", classes="msg-user-label")
        user_body  = Static(text)
        user_box   = Static(f"[bold cyan]You[/]\n{text}", classes="msg-user")
        await history.mount(user_box)

        # Assistant bubble (streamed into)
        asst_box = Static(f"[bold green]{model_id}[/]\n…", classes="msg-assistant")
        await history.mount(asst_box)
        history.scroll_end(animate=False)

        status.update(f"Streaming from {model_id}…")

        try:
            full = ""
            prompt = self._build_prompt()
            async for token in server.quick_infer(model_id, prompt):
                full += token
                asst_box.update(f"[bold green]{model_id}[/]\n{full}")
                history.scroll_end(animate=False)

            self._conversation.append({"role": "assistant", "content": full})
            status.update(f"Done  ·  {len(full.split())} words")

        except asyncio.CancelledError:
            asst_box.update(f"[bold green]{model_id}[/]\n[dim](cancelled)[/]")
            status.update("Cancelled")
        except Exception as exc:
            err = Static(f"[red]Error: {exc}[/]", classes="msg-error")
            await history.mount(err)
            status.update(f"Error: {exc}")
        finally:
            self._streaming = False
            inp.disabled = False
            self.query_one("#btn-chat-send", Button).disabled = False
            inp.focus()

    def _build_prompt(self) -> str:
        """Format full conversation history as a single prompt string."""
        lines = [f"System: {self._system_prompt}", ""]
        for msg in self._conversation:
            role = "User" if msg["role"] == "user" else "Assistant"
            lines.append(f"{role}: {msg['content']}")
        lines.append("Assistant:")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def action_clear_chat(self) -> None:
        self._conversation.clear()
        history = self.query_one("#chat-history", VerticalScroll)
        for child in list(history.children):
            child.remove()
        self.query_one("#chat-status", Static).update("Ready")
