"""Profiles screen — named configuration presets."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Button, DataTable, Input, Label

from llmmanager.config.schema import ProfileConfig
from llmmanager.widgets.confirm_dialog import ConfirmDialog

if TYPE_CHECKING:
    from llmmanager.app import LLMManagerApp


class ProfilesScreen(Screen):
    """Screen 6 — save, load, and switch named server configurations."""

    BINDINGS = [
        ("a", "add_profile",    "New Profile"),
        ("delete", "delete_profile", "Delete"),
        ("enter", "load_profile", "Load"),
    ]

    def compose(self) -> ComposeResult:
        with Horizontal():
            with Vertical(id="profile-list-panel"):
                yield Label("Profiles", classes="section-heading")
                yield DataTable(id="profiles-table", cursor_type="row")
                with Horizontal(id="profile-actions"):
                    yield Button("Save Current", id="btn-save-profile",  variant="primary")
                    yield Button("Load",         id="btn-load-profile",  variant="success")
                    yield Button("Delete",       id="btn-delete-profile",variant="error")

            with Vertical(id="profile-detail-panel"):
                yield Label("Profile Details", classes="section-heading")
                yield Label("Name:", classes="form-label")
                yield Input(placeholder="Profile name...", id="profile-name-input")
                yield Label("Description:", classes="form-label")
                yield Input(placeholder="Optional description...", id="profile-desc-input")
                yield Label("", id="profile-detail-body", classes="form-hint")

    def on_mount(self) -> None:
        table = self.query_one("#profiles-table", DataTable)
        table.add_columns("Name", "Description", "Servers", "Updated")
        self._refresh_table()

    def _refresh_table(self) -> None:
        app: LLMManagerApp = self.app  # type: ignore[assignment]
        table = self.query_one("#profiles-table", DataTable)
        table.clear()
        for name, profile in app.config_manager.config.profiles.items():
            active = " ★" if name == app.config_manager.config.active_profile else ""
            table.add_row(
                name + active,
                profile.description or "—",
                ", ".join(profile.servers.keys()) or "—",
                profile.updated_at[:10] if profile.updated_at else "—",
                key=name,
            )

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        match event.button.id:
            case "btn-save-profile":
                self._save_current_as_profile()
            case "btn-load-profile":
                self._load_selected_profile()
            case "btn-delete-profile":
                await self._delete_selected_profile()

    def _save_current_as_profile(self) -> None:
        app: LLMManagerApp = self.app  # type: ignore[assignment]
        name = self.query_one("#profile-name-input", Input).value.strip()
        desc = self.query_one("#profile-desc-input", Input).value.strip()
        if not name:
            self.notify("Enter a profile name.", severity="warning")
            return
        now = datetime.utcnow().isoformat()
        profile = ProfileConfig(
            name=name,
            description=desc,
            servers=dict(app.config_manager.config.servers),
            created_at=now,
            updated_at=now,
        )
        app.config_manager.config.profiles[name] = profile
        app.config_manager.save()
        self._refresh_table()
        self.notify(f"Profile '{name}' saved.")

    def _load_selected_profile(self) -> None:
        app: LLMManagerApp = self.app  # type: ignore[assignment]
        table = self.query_one("#profiles-table", DataTable)
        if table.cursor_row is None:
            return
        row = table.get_row_at(table.cursor_row)
        name = str(row[0]).rstrip(" ★")
        profile = app.config_manager.config.profiles.get(name)
        if profile is None:
            self.notify(f"Profile '{name}' not found.", severity="error")
            return
        app.config_manager.config.servers = dict(profile.servers)
        app.config_manager.config.active_profile = name
        app.config_manager.save()
        self._refresh_table()
        self.notify(f"Profile '{name}' loaded. Restart servers to apply.")

    async def _delete_selected_profile(self) -> None:
        app: LLMManagerApp = self.app  # type: ignore[assignment]
        table = self.query_one("#profiles-table", DataTable)
        if table.cursor_row is None:
            return
        row = table.get_row_at(table.cursor_row)
        name = str(row[0]).rstrip(" ★")
        confirmed = await self.app.push_screen_wait(
            ConfirmDialog(f"Delete profile '{name}'?", "This cannot be undone.")
        )
        if confirmed:
            app.config_manager.config.profiles.pop(name, None)
            app.config_manager.save()
            self._refresh_table()
            self.notify(f"Profile '{name}' deleted.")
