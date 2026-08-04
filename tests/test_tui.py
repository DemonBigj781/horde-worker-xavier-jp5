import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from horde_worker_regen.process_management import main_entry_point
from horde_worker_regen.system_resources import SystemResourceSnapshot


class FakeProcessMap(dict):
    def get_process_info_strings(self) -> list[str]:
        return ["Process 1 (WAITING_FOR_JOB) (No model loaded)"]


def make_manager() -> SimpleNamespace:
    bridge_data = SimpleNamespace(
        dreamer_worker_name="Test Xavier",
        image_models_to_load=["model-a", "model-b"],
        max_power=46,
        max_threads=1,
        queue_size=0,
        safety_on_gpu=True,
    )
    user_info = SimpleNamespace(username="test-user")
    resources = SystemResourceSnapshot(
        ram_total_bytes=32 * 1024**3,
        ram_available_bytes=4 * 1024**3,
        swap_total_bytes=64 * 1024**3,
        swap_free_bytes=63 * 1024**3,
        worker_process_bytes=27 * 1024**3,
        load_average=(1.0, 2.0, 3.0),
        gpu_frequency_hz=1_377_000_000,
        gpu_temperature_c=44.0,
        fan_pwm=255,
    )
    return SimpleNamespace(
        bridge_data=bridge_data,
        user_info=user_info,
        _process_map=FakeProcessMap(),
        jobs_pending_inference=[],
        jobs_in_progress=[],
        jobs_pending_safety_check=[],
        jobs_pending_submit=[],
        num_jobs_total=3,
        total_num_completed_jobs=2,
        _num_jobs_faulted=0,
        _num_process_recoveries=1,
        _num_job_slowdowns=0,
        kudos_generated_this_session=12.5,
        get_pending_megapixelsteps=lambda: 0,
        get_system_resource_snapshot=lambda: resources,
    )


def test_tui_dashboard_composes_worker_state() -> None:
    from horde_worker_regen.tui.app import HordeWorkerTUI

    async def exercise() -> None:
        app = HordeWorkerTUI(make_manager())
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            assert "Test Xavier" in str(app.query_one("#worker-summary").render())
            assert "WAITING_FOR_JOB" in str(app.query_one("#processes").render())
            assert "Submitted" in str(app.query_one("#session-stats").render())
            assert "1377 MHz" in str(app.query_one("#resources").render())

    asyncio.run(exercise())


def test_start_working_routes_tui_without_starting_console_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = Mock()
    monkeypatch.setattr(main_entry_point, "HordeWorkerProcessManager", Mock(return_value=manager))
    run_tui = Mock()
    monkeypatch.setitem(sys.modules, "horde_worker_regen.tui", SimpleNamespace(run_tui=run_tui))

    main_entry_point.start_working(
        ctx=Mock(),
        bridge_data=Mock(),
        horde_model_reference_manager=Mock(),
        use_tui=True,
    )

    run_tui.assert_called_once_with(manager)
    manager.start.assert_not_called()


def test_tui_failure_shuts_down_worker_before_propagating(monkeypatch: pytest.MonkeyPatch) -> None:
    from horde_worker_regen.tui import app as tui_app

    class FailingApp:
        async def run_async(self) -> None:
            raise RuntimeError("terminal failed")

        def exit(self) -> None:
            pass

    class RunningManager:
        def __init__(self) -> None:
            self.shutdown = asyncio.Event()

        async def run(self) -> None:
            await self.shutdown.wait()

        def _shutdown(self) -> None:
            self.shutdown.set()

    manager = RunningManager()
    monkeypatch.setattr(tui_app, "HordeWorkerTUI", lambda _manager: FailingApp())

    with pytest.raises(RuntimeError, match="terminal failed"):
        asyncio.run(tui_app._run_tui(manager))

    assert manager.shutdown.is_set()
