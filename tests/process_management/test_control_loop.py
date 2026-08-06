"""Tests for the extracted control-loop tick.

The tick is the body of HordeWorkerProcessManager._process_control_loop; with a
no-op sleep injected it can be driven deterministically without wall-clock delays.
"""

from __future__ import annotations

import time
import uuid
from unittest.mock import Mock

from horde_worker_regen.process_management.horde_process import HordeProcessType
from horde_worker_regen.process_management.messages import HordeProcessState
from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager

from .conftest import (
    make_mock_job,
    make_mock_process_info,
    make_testable_process_manager,
    track_popped_job_async,
)


async def _noop_sleep(_delay: float) -> None:
    return None


def _make_tickable_manager() -> HordeWorkerProcessManager:
    """Build a testable manager whose tick can run without sleeping or printing."""
    process_manager = make_testable_process_manager()
    process_manager._sleep = _noop_sleep
    # Pretend a status message was just printed so the tick does not exercise
    # the status reporter (covered by its own tests).
    process_manager._last_status_message_time = time.time()
    return process_manager


class TestControlLoopTick:
    """Tests for _control_loop_tick."""

    async def test_idle_tick_keeps_running(self) -> None:
        """With nothing to do, a tick should complete and ask to keep looping."""
        process_manager = _make_tickable_manager()

        assert await process_manager._control_loop_tick() is True

    async def test_tick_requests_shutdown_when_ready(self) -> None:
        """When shutting down with no jobs and no processes, the tick should ask to stop."""
        process_manager = _make_tickable_manager()
        process_manager._state.shutting_down = True

        assert await process_manager._control_loop_tick() is False

    async def test_tick_dispatches_pending_safety_check(self) -> None:
        """A tick should send a job pending safety check to an available safety process."""
        process_manager = _make_tickable_manager()
        safety_proc = make_mock_process_info(
            10,
            model_name=None,
            state=HordeProcessState.WAITING_FOR_JOB,
            process_type=HordeProcessType.SAFETY,
        )
        process_manager._process_map.update({10: safety_proc})

        job = Mock()
        job.id_ = uuid.uuid4()
        job.model = "stable_diffusion"
        job.payload = Mock()
        job.payload.prompt = "test prompt"
        job.payload.use_nsfw_censor = False

        job_info = Mock()
        job_info.sdk_api_job_info = job
        job_info.job_image_results = [Mock()]
        job_info.images_base64 = ["base64data"]

        await process_manager._job_tracker.queue_for_safety(job_info)

        assert await process_manager._control_loop_tick() is True

        assert job_info in process_manager._job_tracker.jobs_being_safety_checked
        assert job_info not in process_manager._job_tracker.jobs_pending_safety_check

    async def test_tick_starts_inference_for_pending_job(self) -> None:
        """A tick should start inference for a pending job whose model is on a free process."""
        process_manager = _make_tickable_manager()
        inf_proc = make_mock_process_info(
            0,
            model_name="stable_diffusion",
            state=HordeProcessState.WAITING_FOR_JOB,
        )
        safety_proc = make_mock_process_info(
            10,
            model_name=None,
            state=HordeProcessState.WAITING_FOR_JOB,
            process_type=HordeProcessType.SAFETY,
        )
        process_manager._process_map.update({0: inf_proc, 10: safety_proc})

        job = await track_popped_job_async(process_manager._job_tracker, make_mock_job())

        assert await process_manager._control_loop_tick() is True

        assert job in process_manager._job_tracker.jobs_in_progress
        inf_proc.pipe_connection.send.assert_called()  # type: ignore[attr-defined]


class TestMemoryPressureRecovery:
    """Tests for Xavier unified-memory process reclamation."""

    @staticmethod
    def _make_manager(*, jobs_pending_submit: tuple[object, ...] = ()) -> HordeWorkerProcessManager:
        process_manager = _make_tickable_manager()
        process_manager._job_popper._ram_guard_active = True
        process_manager._memory_pressure_recovery_attempted = False
        process_manager._last_memory_pressure_recovery_time = 0.0

        process_manager._job_tracker = Mock()
        process_manager._job_tracker.jobs_pending_inference = ()
        process_manager._job_tracker.jobs_in_progress = ()
        process_manager._job_tracker.jobs_pending_safety_check = ()
        process_manager._job_tracker.jobs_being_safety_checked = ()
        process_manager._job_tracker.jobs_pending_submit = jobs_pending_submit

        process = make_mock_process_info(
            0,
            model_name="stable_diffusion",
            state=HordeProcessState.WAITING_FOR_JOB,
        )
        process_manager._process_map.clear()
        process_manager._process_map[0] = process
        process_manager._process_lifecycle._replace_inference_process = Mock()
        return process_manager

    def test_completed_upload_blocks_memory_pressure_recycle(self) -> None:
        """Never kill a process while its completed result still needs submission."""
        process_manager = self._make_manager(jobs_pending_submit=(object(),))

        assert process_manager._recover_idle_inference_process_for_memory_pressure() is False
        process_manager._process_lifecycle._replace_inference_process.assert_not_called()

    def test_idle_process_recycles_after_all_job_stages_drain(self) -> None:
        """Once submission drains, recycle one idle process without faulting a job."""
        process_manager = self._make_manager()
        process = process_manager._process_map[0]

        assert process_manager._recover_idle_inference_process_for_memory_pressure() is True
        process_manager._process_lifecycle._replace_inference_process.assert_called_once_with(
            process,
            fault_referenced_job=False,
        )

    def test_recovery_runs_only_once_per_pressure_episode(self) -> None:
        """A persistent low-memory reading must not cause continuous process churn."""
        process_manager = self._make_manager()

        assert process_manager._recover_idle_inference_process_for_memory_pressure() is True
        assert process_manager._recover_idle_inference_process_for_memory_pressure() is False
        process_manager._process_lifecycle._replace_inference_process.assert_called_once()
