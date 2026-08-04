"""Textual dashboard and launcher for horde-worker-reGen."""

from __future__ import annotations

import asyncio
import sys
from collections import deque
from typing import TYPE_CHECKING, Protocol, TypedDict

from loguru import logger
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Footer, Header, RichLog, Static

from horde_worker_regen.tui.data_provider import build_worker_snapshot

if TYPE_CHECKING:
    from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager


class LogLevel(Protocol):
    """Loguru level shape used by the TUI sink."""

    name: str


class LogRecord(TypedDict):
    """Subset of a Loguru record consumed by the dashboard."""

    level: LogLevel
    message: str


class LogMessage(Protocol):
    """Loguru message shape used by the TUI sink."""

    record: LogRecord


class HordeWorkerTUI(App[None]):
    """Live dashboard for a worker process manager."""

    TITLE = "AI Horde Worker reGen"
    SUB_TITLE = "Xavier dashboard"
    BINDINGS = [
        ("q", "quit", "Quit"),
        ("r", "refresh_dashboard", "Refresh"),
        ("d", "toggle_dark", "Theme"),
    ]
    CSS = """
    Screen {
        background: #0d1518;
        color: #e5eee9;
    }
    #worker-summary {
        height: 3;
        padding: 1 2;
        background: #163037;
        border-bottom: solid #e1a955;
    }
    #configuration {
        height: 3;
        padding: 1 2;
        color: #b7cbc2;
        background: #102429;
    }
    #main-columns {
        height: 15;
        margin: 1 1 0 1;
    }
    .panel {
        height: 1fr;
        padding: 1 2;
        margin: 0 1;
        border: round #4f8076;
        background: #112126;
    }
    #processes {
        height: 11;
        margin: 1 2 0 2;
        padding: 1 2;
        border: round #4f8076;
        background: #112126;
    }
    #resources {
        height: 5;
        margin: 1 2 0 2;
        padding: 1 2;
        border: round #e1a955;
        background: #18251f;
    }
    #activity-log {
        height: 1fr;
        margin: 1 2;
        border: round #4f8076;
        background: #091114;
    }
    """

    def __init__(self, process_manager: HordeWorkerProcessManager) -> None:
        """Create a dashboard for the supplied process manager."""
        super().__init__()
        self.process_manager = process_manager
        self._pending_logs: deque[tuple[str, str]] = deque(maxlen=1000)
        self._log_sink_id: int | None = None

    def compose(self) -> ComposeResult:
        """Compose the worker dashboard."""
        yield Header(show_clock=True)
        yield Static(id="worker-summary", markup=True)
        yield Static(id="configuration", markup=True)
        with Horizontal(id="main-columns"):
            with Vertical(classes="panel"):
                yield Static("[b]Job Queues[/b]\n", id="job-queues", markup=True)
            with Vertical(classes="panel"):
                yield Static("[b]Session[/b]\n", id="session-stats", markup=True)
        yield Static("[b]Processes[/b]\n", id="processes", markup=True)
        yield Static("[b]System Resources[/b]\n", id="resources", markup=True)
        yield RichLog(id="activity-log", markup=True, highlight=True, auto_scroll=True)
        yield Footer()

    def on_mount(self) -> None:
        """Start dashboard and log refresh timers."""
        self._log_sink_id = logger.add(self._capture_log, level="INFO", enqueue=False)
        self.refresh_dashboard()
        self.set_interval(1.0, self.refresh_dashboard)
        self.set_interval(0.2, self._flush_logs)

    def on_unmount(self) -> None:
        """Remove the TUI-specific log sink."""
        if self._log_sink_id is not None:
            logger.remove(self._log_sink_id)
            self._log_sink_id = None

    def _capture_log(self, message: LogMessage) -> None:
        record = message.record
        self._pending_logs.append((record["level"].name, record["message"]))

    def _flush_logs(self) -> None:
        log = self.query_one("#activity-log", RichLog)
        while self._pending_logs:
            level, message = self._pending_logs.popleft()
            color = {"ERROR": "red", "CRITICAL": "red", "WARNING": "yellow", "SUCCESS": "green"}.get(
                level,
                "cyan",
            )
            log.write(f"[{color}]{level:8}[/{color}] {message}")

    def refresh_dashboard(self) -> None:
        """Refresh the worker state panels."""
        snapshot = build_worker_snapshot(self.process_manager)
        self.query_one("#worker-summary", Static).update(snapshot.worker_summary)
        self.query_one("#configuration", Static).update(snapshot.configuration_summary)
        self.query_one("#job-queues", Static).update(f"[b]Job Queues[/b]\n{snapshot.queue_summary}")
        self.query_one("#session-stats", Static).update(f"[b]Session[/b]\n{snapshot.session_summary}")
        self.query_one("#processes", Static).update(f"[b]Processes[/b]\n{snapshot.process_summary}")
        self.query_one("#resources", Static).update(f"[b]System Resources[/b]\n{snapshot.resource_summary}")

    def action_refresh_dashboard(self) -> None:
        """Refresh immediately when requested."""
        self.refresh_dashboard()


async def _run_tui(process_manager: HordeWorkerProcessManager) -> None:
    app = HordeWorkerTUI(process_manager)
    worker_task = asyncio.create_task(process_manager.run(), name="horde-worker")
    app_task = asyncio.create_task(app.run_async(), name="horde-worker-tui")

    done, _pending = await asyncio.wait((worker_task, app_task), return_when=asyncio.FIRST_COMPLETED)
    if worker_task in done:
        app.exit()
    else:
        process_manager._shutdown()

    try:
        results = await asyncio.wait_for(
            asyncio.gather(worker_task, app_task, return_exceptions=True),
            timeout=30,
        )
    except TimeoutError:
        worker_task.cancel()
        app_task.cancel()
        results = await asyncio.gather(worker_task, app_task, return_exceptions=True)

    for result in results:
        if isinstance(result, BaseException) and not isinstance(result, asyncio.CancelledError):
            raise result


def run_tui(process_manager: HordeWorkerProcessManager) -> None:
    """Run the worker and dashboard in the main thread."""
    asyncio.run(_run_tui(process_manager))


def main() -> None:
    """Launch the normal worker entry point with TUI mode selected."""
    if "--tui" not in sys.argv:
        sys.argv.insert(1, "--tui")
    from horde_worker_regen.run_worker import init

    init()
