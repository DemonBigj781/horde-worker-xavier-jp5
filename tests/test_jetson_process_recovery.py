import asyncio
import time
from unittest.mock import Mock, call

import pytest
from loguru import logger

from horde_worker_regen.process_management import process_manager, worker_entry_points
from horde_worker_regen.process_management.horde_process import HordeProcessType
from horde_worker_regen.process_management.messages import (
    HordeControlFlag,
    HordeControlMessage,
    HordeProcessState,
    HordeSafetyControlMessage,
)
from horde_worker_regen.process_management.process_manager import (
    HordeProcessInfo,
    HordeWorkerProcessManager,
    ProcessMap,
)
from horde_worker_regen.process_management.safety_process import HordeSafetyProcess


class BrokenConnection:
    def send(self, _message: HordeControlMessage) -> None:
        raise BrokenPipeError(32, "Broken pipe")


class UnexpectedFailureConnection:
    def send(self, _message: HordeControlMessage) -> None:
        raise RuntimeError("unexpected send failure")


def make_process_info(
    pipe_connection: object,
    *,
    alive: bool = False,
    state: HordeProcessState = HordeProcessState.PROCESS_ENDED,
    process_type: HordeProcessType = HordeProcessType.INFERENCE,
) -> HordeProcessInfo:
    process = Mock()
    process.is_alive.return_value = alive
    return HordeProcessInfo(
        mp_process=process,
        pipe_connection=pipe_connection,
        process_id=1,
        process_type=process_type,
        last_process_state=state,
        process_launch_identifier=1,
    )


def capture_send(pipe_connection: object) -> tuple[bool, list[dict]]:
    records = []
    sink = logger.add(lambda message: records.append(message.record), level="DEBUG")
    try:
        sent = make_process_info(pipe_connection).safe_send_message(
            HordeControlMessage(control_flag=HordeControlFlag.END_PROCESS),
        )
    finally:
        logger.remove(sink)
    return sent, records


def test_expected_broken_pipe_during_process_recovery_is_not_an_error() -> None:
    sent, records = capture_send(BrokenConnection())

    assert sent is False
    assert not any(record["level"].name == "ERROR" for record in records)
    assert any("control channel already closed" in record["message"] for record in records)


def test_unexpected_send_failure_remains_an_error() -> None:
    sent, records = capture_send(UnexpectedFailureConnection())

    assert sent is False
    assert any(record["level"].name == "ERROR" for record in records)


def test_running_process_reports_alive() -> None:
    process_info = make_process_info(
        Mock(),
        alive=True,
        state=HordeProcessState.WAITING_FOR_JOB,
    )

    assert process_info.is_process_alive() is True


def test_available_inference_count_excludes_safety_and_starting_processes() -> None:
    """Only inference processes ready to accept a job count as available."""
    safety = make_process_info(
        Mock(),
        alive=True,
        state=HordeProcessState.WAITING_FOR_JOB,
        process_type=HordeProcessType.SAFETY,
    )
    inference = make_process_info(
        Mock(),
        alive=True,
        state=HordeProcessState.PROCESS_STARTING,
    )
    inference.process_id = 2
    process_map = ProcessMap({0: safety, 2: inference})

    assert process_map.num_available_inference_processes() == 0

    inference.last_process_state = HordeProcessState.WAITING_FOR_JOB

    assert process_map.num_available_inference_processes() == 1


def test_model_switch_cycles_waiting_inference_process_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """A waiting process restarts before loading a different model on unified memory."""
    monkeypatch.setattr(process_manager, "HordePreloadInferenceModelMessage", lambda **kwargs: kwargs)

    process = Mock()
    process.process_id = 1
    process.loaded_horde_model_name = "previous-model"
    process.last_process_state = HordeProcessState.WAITING_FOR_JOB

    process_map = Mock()
    process_map.values.return_value = [process]
    process_map.num_loaded_inference_processes.return_value = 1
    process_map.get_first_available_inference_process.return_value = process

    job = Mock()
    job.model = "next-model"
    job.payload.loras = None

    manager = object.__new__(HordeWorkerProcessManager)
    manager._process_map = process_map
    manager._horde_model_map = Mock(root={})
    manager.jobs_pending_inference = [job]
    manager.jobs_in_progress = []
    manager.bridge_data = Mock(cycle_process_on_model_change=True)
    manager._shutting_down = False
    manager._replace_inference_process = Mock()

    assert manager.preload_models() is True
    manager._replace_inference_process.assert_called_once_with(process, fault_referenced_job=False)
    process.safe_send_message.assert_not_called()


def test_intentional_process_cycle_does_not_fault_referenced_job() -> None:
    """Model-change cycling preserves a prior job that is completing submission."""
    referenced_job = Mock()
    process = make_process_info(
        Mock(),
        state=HordeProcessState.WAITING_FOR_JOB,
    )
    process.loaded_horde_model_name = "previous-model"
    process.last_job_referenced = referenced_job

    manager = object.__new__(HordeWorkerProcessManager)
    manager._process_map = Mock()
    manager._process_map.values.return_value = [process]
    manager.jobs_lookup = {referenced_job: Mock()}
    manager._horde_model_map = Mock()
    manager.handle_job_fault = Mock()
    manager._end_inference_process = Mock()
    manager.receive_and_handle_process_messages = Mock()
    manager._start_inference_process = Mock()
    calls = Mock()
    calls.attach_mock(manager._end_inference_process, "end")
    calls.attach_mock(manager.receive_and_handle_process_messages, "drain")
    calls.attach_mock(manager._start_inference_process, "start")
    manager._num_process_recoveries = 0

    manager._replace_inference_process(process, fault_referenced_job=False)

    manager.handle_job_fault.assert_not_called()
    assert calls.mock_calls == [call.end(process), call.drain(), call.start(process.process_id)]


def test_safety_dispatch_marks_process_busy_before_sending_next_job(monkeypatch: pytest.MonkeyPatch) -> None:
    """A slow safety check cannot leave the subprocess available for another IPC message."""
    monkeypatch.setattr(process_manager, "HordeSafetyControlMessage", lambda **kwargs: kwargs)

    safety_process = Mock()
    safety_process.last_process_state = HordeProcessState.WAITING_FOR_JOB
    safety_process.safe_send_message.return_value = True

    job = Mock()
    job.sdk_api_job_info.id_ = "job-id"
    job.sdk_api_job_info.model = "model"
    job.sdk_api_job_info.payload.prompt = "prompt"
    job.sdk_api_job_info.payload.use_nsfw_censor = False
    job.job_image_results = [Mock()]
    job.images_base64 = ["image"]

    manager = object.__new__(HordeWorkerProcessManager)
    manager.jobs_pending_safety_check = [job]
    manager.jobs_being_safety_checked = []
    manager._process_map = Mock()
    manager._process_map.get_first_available_safety_process.return_value = safety_process
    manager._process_map.get_safety_process.return_value = safety_process
    manager.stable_diffusion_reference = Mock(root={})
    manager.bridge_data = Mock(nsfw=True)

    manager.start_evaluate_safety()

    assert safety_process.last_process_state == HordeProcessState.EVALUATING_SAFETY
    assert safety_process.last_control_flag == HordeControlFlag.EVALUATE_SAFETY
    assert manager.jobs_pending_safety_check == []
    assert manager.jobs_being_safety_checked == [job]


def test_safety_child_reports_busy_before_starting_evaluation() -> None:
    """The child reports its busy state before loading or evaluating image data."""
    safety_process = object.__new__(HordeSafetyProcess)
    safety_process.process_id = 0
    safety_process.process_launch_identifier = 1
    safety_process.process_message_queue = Mock()
    safety_process.send_process_state_change_message = Mock()
    safety_process.send_memory_report_message = Mock()
    calls = Mock()
    calls.attach_mock(safety_process.send_process_state_change_message, "state")
    calls.attach_mock(safety_process.send_memory_report_message, "memory")

    safety_process._receive_and_handle_control_message(
        HordeSafetyControlMessage(
            control_flag=HordeControlFlag.EVALUATE_SAFETY,
            job_id="00000000-0000-0000-0000-000000000000",
            prompt="prompt",
            censor_nsfw=False,
            sfw_worker=False,
            images_base64=[],
            horde_model_info={},
        ),
    )

    assert calls.mock_calls[:2] == [
        call.state(
            HordeProcessState.EVALUATING_SAFETY,
            "Evaluating safety for job 00000000-0000-0000-0000-000000000000",
        ),
        call.memory(include_vram=False),
    ]


def test_r2_upload_timeout_is_independent_of_slow_worker_flag() -> None:
    """Xavier can keep a long upload deadline without advertising as an extra-slow worker."""
    manager = object.__new__(HordeWorkerProcessManager)
    manager.bridge_data = Mock(extra_slow_worker=False, r2_upload_timeout=60)

    assert manager.get_r2_upload_timeout_seconds() == 60

    manager.bridge_data.extra_slow_worker = True

    assert manager.get_r2_upload_timeout_seconds() == 60


def test_memory_guard_blocks_job_pop_below_configured_reserve() -> None:
    manager = object.__new__(HordeWorkerProcessManager)
    manager.bridge_data = Mock(minimum_available_ram_gib=8)
    manager._system_resource_reader = Mock()
    manager._system_resource_reader.available_ram_bytes.return_value = 4 * 1024**3
    manager._last_memory_guard_log_time = 0
    manager._memory_pressure_recovery_requested = False

    assert manager._memory_guard_allows_job_pop() is False
    assert manager._memory_pressure_recovery_requested is True


def test_memory_guard_allows_job_pop_above_configured_reserve() -> None:
    manager = object.__new__(HordeWorkerProcessManager)
    manager.bridge_data = Mock(minimum_available_ram_gib=8)
    manager._system_resource_reader = Mock()
    manager._system_resource_reader.available_ram_bytes.return_value = 9 * 1024**3
    manager._last_memory_guard_log_time = 0
    manager._memory_pressure_recovery_requested = True

    assert manager._memory_guard_allows_job_pop() is True
    assert manager._memory_pressure_recovery_requested is False


def test_inference_process_reserves_guarded_memory_inside_comfyui() -> None:
    args = worker_entry_points._build_inference_comfyui_args(vram_reserve_gib=8)

    reserve_index = args.index("--reserve-vram")
    assert args[reserve_index + 1] == "8"
    assert "1.4" not in args


def test_inference_process_preserves_default_comfyui_reserve() -> None:
    args = worker_entry_points._build_inference_comfyui_args()

    reserve_index = args.index("--reserve-vram")
    assert args[reserve_index + 1] == "1.4"


@pytest.mark.parametrize(
    ("mode_options", "expected_arg"),
    [
        ({"low_memory_mode": True}, "--novram"),
        ({"very_high_memory_mode": True}, "--gpu-only"),
    ],
)
def test_explicit_memory_modes_do_not_add_a_vram_reserve(mode_options: dict[str, bool], expected_arg: str) -> None:
    args = worker_entry_points._build_inference_comfyui_args(vram_reserve_gib=8, **mode_options)

    assert expected_arg in args
    assert "--reserve-vram" not in args


def test_heavy_model_reserve_uses_larger_guarded_value() -> None:
    args = worker_entry_points._build_inference_comfyui_args(
        high_memory_mode=True,
        vram_heavy_models=True,
        vram_reserve_gib=8,
    )

    reserve_index = args.index("--reserve-vram")
    assert args[reserve_index + 1] == "8"


def test_process_manager_forwards_memory_reserve_to_inference_child(monkeypatch: pytest.MonkeyPatch) -> None:
    process = Mock()
    monkeypatch.setattr(process_manager.multiprocessing, "Pipe", Mock(return_value=(Mock(), Mock())))
    process_factory = Mock(return_value=process)
    monkeypatch.setattr(process_manager.multiprocessing, "Process", process_factory)

    manager = object.__new__(HordeWorkerProcessManager)
    manager.bridge_data = Mock(
        image_models_to_load=[],
        very_high_memory_mode=False,
        high_memory_mode=False,
        minimum_available_ram_gib=8,
    )
    manager._process_message_queue = Mock()
    manager._inference_semaphore = Mock()
    manager._disk_lock = Mock()
    manager._aux_model_lock = Mock()
    manager._vae_decode_semaphore = Mock()
    manager.num_processes_launched = 0
    manager._amd_gpu = False
    manager._directml = None
    manager._process_map = {}

    manager._start_inference_process(1)

    assert process_factory.call_args.kwargs["kwargs"]["vram_reserve_gib"] == 8


def test_api_job_pop_returns_before_api_work_when_memory_guard_is_active() -> None:
    manager = object.__new__(HordeWorkerProcessManager)
    manager._shutting_down = False
    manager._memory_guard_allows_job_pop = Mock(return_value=False)

    asyncio.run(manager.api_job_pop())

    manager._memory_guard_allows_job_pop.assert_called_once_with()


def test_memory_pressure_recycles_idle_inference_process_without_faulting_job() -> None:
    process = Mock()
    process.process_type = HordeProcessType.INFERENCE
    process.last_process_state = HordeProcessState.WAITING_FOR_JOB
    process.is_process_busy.return_value = False

    manager = object.__new__(HordeWorkerProcessManager)
    manager._memory_pressure_recovery_requested = True
    manager._last_memory_pressure_recovery_time = 0
    manager._process_map = Mock()
    manager._process_map.values.return_value = [process]
    manager.jobs_pending_inference = []
    manager.jobs_in_progress = []
    manager.jobs_pending_safety_check = []
    manager.jobs_being_safety_checked = []
    manager.jobs_pending_submit = []
    manager._replace_inference_process = Mock()

    assert manager._recover_idle_inference_processes_for_memory_pressure() is True
    manager._replace_inference_process.assert_called_once_with(process, fault_referenced_job=False)
    assert manager._memory_pressure_recovery_requested is False


def test_memory_pressure_does_not_recycle_process_while_job_is_active() -> None:
    manager = object.__new__(HordeWorkerProcessManager)
    manager._memory_pressure_recovery_requested = True
    manager._last_memory_pressure_recovery_time = 0
    manager._process_map = Mock()
    manager.jobs_pending_inference = []
    manager.jobs_in_progress = [Mock()]
    manager.jobs_pending_safety_check = []
    manager.jobs_being_safety_checked = []
    manager.jobs_pending_submit = []
    manager._replace_inference_process = Mock()

    assert manager._recover_idle_inference_processes_for_memory_pressure() is False
    manager._replace_inference_process.assert_not_called()


def test_memory_pressure_can_recycle_while_completed_job_is_uploading() -> None:
    process = Mock()
    process.process_id = 1
    process.process_type = HordeProcessType.INFERENCE
    process.last_process_state = HordeProcessState.WAITING_FOR_JOB

    manager = object.__new__(HordeWorkerProcessManager)
    manager._memory_pressure_recovery_requested = True
    manager._last_memory_pressure_recovery_time = 0
    manager._process_map = Mock()
    manager._process_map.values.return_value = [process]
    manager.jobs_pending_inference = []
    manager.jobs_in_progress = []
    manager.jobs_pending_safety_check = []
    manager.jobs_being_safety_checked = []
    manager.jobs_pending_submit = [Mock()]
    manager._replace_inference_process = Mock()

    assert manager._recover_idle_inference_processes_for_memory_pressure() is True
    manager._replace_inference_process.assert_called_once_with(process, fault_referenced_job=False)


def test_memory_pressure_recovery_respects_cooldown() -> None:
    process = Mock()
    process.process_type = HordeProcessType.INFERENCE
    process.last_process_state = HordeProcessState.WAITING_FOR_JOB

    manager = object.__new__(HordeWorkerProcessManager)
    manager._memory_pressure_recovery_requested = True
    manager._last_memory_pressure_recovery_time = time.time()
    manager._process_map = Mock()
    manager._process_map.values.return_value = [process]
    manager.jobs_pending_inference = []
    manager.jobs_in_progress = []
    manager.jobs_pending_safety_check = []
    manager.jobs_being_safety_checked = []
    manager.jobs_pending_submit = []
    manager._replace_inference_process = Mock()

    assert manager._recover_idle_inference_processes_for_memory_pressure() is False
    manager._replace_inference_process.assert_not_called()
