"""Root Textual App — wires together all screens, services, and background tasks."""

from __future__ import annotations

import asyncio
from pathlib import Path

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Footer, Header, Tab, TabbedContent, TabPane

from llmmanager.config.manager import ConfigManager
from llmmanager.constants import APP_DISPLAY_NAME, APP_VERSION
from llmmanager.gpu.detector import detect_gpu_provider
from llmmanager.notifications.manager import NotificationManager
from llmmanager.screens.api_panel import APIPanelScreen
from llmmanager.screens.benchmarks import BenchmarksScreen
from llmmanager.screens.dashboard import DashboardScreen
from llmmanager.screens.logs import LogsScreen
from llmmanager.screens.model_mgmt import ModelManagementScreen
from llmmanager.screens.profiles import ProfilesScreen
from llmmanager.screens.server_mgmt import ServerManagementScreen
from llmmanager.screens.setup_wizard import SetupWizardScreen
from llmmanager.servers.registry import ServerRegistry
from llmmanager.services.download_manager import DownloadManager
from llmmanager.services.log_tailer import LogTailerService
from llmmanager.services.poller import PollerService


class LLMManagerApp(App):
    """Central control panel for LLM server stacks."""

    CSS_PATH = Path(__file__).parent / "css" / "app.tcss"
    TITLE = f"{APP_DISPLAY_NAME} v{APP_VERSION}"

    BINDINGS = [
        Binding("1", "switch_tab('dashboard')",    "Dashboard",    show=True),
        Binding("2", "switch_tab('servers')",      "Servers",      show=True),
        Binding("3", "switch_tab('models')",       "Models",       show=True),
        Binding("4", "switch_tab('logs')",         "Logs",         show=True),
        Binding("5", "switch_tab('benchmarks')",   "Benchmarks",   show=True),
        Binding("6", "switch_tab('profiles')",     "Profiles",     show=True),
        Binding("7", "switch_tab('api')",          "API Panel",    show=True),
        Binding("f1",  "show_help",     "Help",     show=False),
        Binding("f5",  "force_refresh", "Refresh",  show=False),
        Binding("f10", "quit",          "Quit",     show=True),
        Binding("n",   "show_notifs",   "Notifs",   show=False),
        Binding("q",   "quit",          "Quit",     show=False),
    ]

    def __init__(self) -> None:
        super().__init__()
        # Core services — initialized in on_mount
        self.config_manager = ConfigManager()
        self.gpu_provider = detect_gpu_provider()
        self.registry = ServerRegistry(self.config_manager)
        self.poller: PollerService = None  # type: ignore[assignment]
        self.log_tailer = LogTailerService()
        self.download_manager = DownloadManager()
        self.notif_manager: NotificationManager = None  # type: ignore[assignment]
        self._notif_task: asyncio.Task | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        with TabbedContent(id="main-tabs", initial="dashboard"):
            with TabPane("Dashboard",  id="dashboard"):
                yield DashboardScreen()
            with TabPane("Servers",    id="servers"):
                yield ServerManagementScreen()
            with TabPane("Models",     id="models"):
                yield ModelManagementScreen()
            with TabPane("Logs",       id="logs"):
                yield LogsScreen()
            with TabPane("Benchmarks", id="benchmarks"):
                yield BenchmarksScreen()
            with TabPane("Profiles",   id="profiles"):
                yield ProfilesScreen()
            with TabPane("API Panel",  id="api"):
                yield APIPanelScreen()
        yield Footer()

    async def on_mount(self) -> None:
        # Load config
        cfg = self.config_manager.load()

        # Initialize server backends
        self.registry.initialize()

        # Initialize GPU provider
        await self.gpu_provider.initialize()

        # Wire up services
        self.poller = PollerService(
            gpu_provider=self.gpu_provider,
            server_registry=self.registry,
            interval_ms=cfg.poll_interval_ms,
        )
        self.notif_manager = NotificationManager(cfg)

        # Start background services
        await self.poller.start()
        await self.download_manager.start()

        # Start log tailing for all enabled servers
        for server in self.registry.all_enabled():
            self.log_tailer.start_server(server)

        # Start notification processing loop
        self._notif_task = asyncio.create_task(self._process_notifications())

        # Show setup wizard on first run if nothing installed
        await self._check_first_run()

    async def on_unmount(self) -> None:
        await self.poller.stop()
        await self.download_manager.stop()
        await self.log_tailer.stop_all()
        await self.gpu_provider.shutdown()
        if self._notif_task:
            self._notif_task.cancel()

    async def _process_notifications(self) -> None:
        """Consume poll snapshots and fire notification rules."""
        while True:
            try:
                try:
                    snapshot = self.poller.queue.get_nowait()
                    self.notif_manager.process_snapshot(snapshot)
                except asyncio.QueueEmpty:
                    pass

                # Surface new notifications as Textual toasts
                try:
                    notif = self.notif_manager.queue.get_nowait()
                    self.notify(
                        notif.body,
                        title=notif.title,
                        severity=notif.severity.value,  # type: ignore[arg-type]
                    )
                except asyncio.QueueEmpty:
                    pass

                await asyncio.sleep(0.5)
            except asyncio.CancelledError:
                return

    async def _check_first_run(self) -> None:
        """Show the setup wizard if no servers are installed."""
        any_installed = False
        for server in self.registry.all():
            if await server.is_installed():
                any_installed = True
                break
        if not any_installed:
            await self.push_screen(SetupWizardScreen())

    def action_switch_tab(self, tab_id: str) -> None:
        tabs = self.query_one("#main-tabs", TabbedContent)
        tabs.active = tab_id

    def action_force_refresh(self) -> None:
        self.run_worker(self.poller.force_poll())

    def action_show_notifs(self) -> None:
        count = self.notif_manager.unread_count if self.notif_manager else 0
        self.notify(f"{count} unread notifications.")

    def action_show_help(self) -> None:
        self.notify(
            "Keys: 1-7 switch screens | r restart | s start | S stop | "
            "i install | / search | d download | b benchmark | q quit",
            title="Keyboard Shortcuts",
        )
