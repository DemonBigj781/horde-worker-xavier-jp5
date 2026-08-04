"""Small, defensive view model for the worker TUI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import horde_worker_regen

if TYPE_CHECKING:
    from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager


@dataclass(frozen=True)
class WorkerSnapshot:
    """Values rendered by the dashboard at one point in time."""

    worker_summary: str
    process_summary: str
    queue_summary: str
    session_summary: str
    configuration_summary: str
    resource_summary: str


def build_worker_snapshot(process_manager: HordeWorkerProcessManager) -> WorkerSnapshot:
    """Build a TUI snapshot without mutating worker state."""
    bridge_data = process_manager.bridge_data
    user_info = getattr(process_manager, "user_info", None)
    username = getattr(user_info, "username", None) or "Connecting"
    model_count = len(getattr(bridge_data, "image_models_to_load", ()))

    process_map = getattr(process_manager, "_process_map", None)
    process_lines = process_map.get_process_info_strings() if process_map is not None else []
    process_summary = "\n".join(process_lines) if process_lines else "Processes are starting..."

    pending_inference = len(getattr(process_manager, "jobs_pending_inference", ()))
    in_progress = len(getattr(process_manager, "jobs_in_progress", ()))
    pending_safety = len(getattr(process_manager, "jobs_pending_safety_check", ()))
    pending_submit = len(getattr(process_manager, "jobs_pending_submit", ()))
    pending_megapixelsteps = process_manager.get_pending_megapixelsteps()
    resources = process_manager.get_system_resource_snapshot()

    return WorkerSnapshot(
        worker_summary=(
            f"[b]{bridge_data.dreamer_worker_name}[/b]  "
            f"v{horde_worker_regen.__version__}  User: {username}  Models: {model_count}"
        ),
        process_summary=process_summary,
        queue_summary=(
            f"Pending start: [b]{pending_inference}[/b] ({pending_megapixelsteps} eMPS)\n"
            f"In progress: [b]{in_progress}[/b]\n"
            f"Safety checks: [b]{pending_safety}[/b]\n"
            f"Awaiting submit: [b]{pending_submit}[/b]"
        ),
        session_summary=(
            f"Jobs popped: [b]{process_manager.num_jobs_total}[/b]\n"
            f"Submitted: [b]{process_manager.total_num_completed_jobs}[/b]\n"
            f"Faulted: [b]{getattr(process_manager, '_num_jobs_faulted', 0)}[/b]\n"
            f"Slow jobs: [b]{getattr(process_manager, '_num_job_slowdowns', 0)}[/b]\n"
            f"Recoveries: [b]{getattr(process_manager, '_num_process_recoveries', 0)}[/b]\n"
            f"Session kudos: [b]{getattr(process_manager, 'kudos_generated_this_session', 0):,.2f}[/b]"
        ),
        configuration_summary=(
            f"Max power: [b]{bridge_data.max_power}[/b]  Threads: [b]{bridge_data.max_threads}[/b]  "
            f"Queue: [b]{bridge_data.queue_size}[/b]  Safety GPU: [b]{bridge_data.safety_on_gpu}[/b]"
        ),
        resource_summary=resources.format_tui(),
    )
