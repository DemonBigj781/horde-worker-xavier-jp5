from multiprocessing.context import BaseContext

from horde_model_reference.model_reference_manager import ModelReferenceManager

from horde_worker_regen.bridge_data.data_model import reGenBridgeData
from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager


def start_working(
    ctx: BaseContext,
    bridge_data: reGenBridgeData,
    horde_model_reference_manager: ModelReferenceManager,
    *,
    amd_gpu: bool = False,
    directml: int | None = None,
    use_tui: bool = False,
) -> None:
    """Create and start process manager."""
    process_manager = HordeWorkerProcessManager(
        ctx=ctx,
        bridge_data=bridge_data,
        horde_model_reference_manager=horde_model_reference_manager,
        amd_gpu=amd_gpu,
        directml=directml,
    )

    if use_tui:
        try:
            from horde_worker_regen.tui import run_tui
        except ImportError as exc:
            raise RuntimeError("TUI mode requires Textual. Reinstall the worker dependencies.") from exc
        run_tui(process_manager)
        return

    process_manager.start()
