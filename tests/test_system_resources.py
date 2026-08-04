from pathlib import Path
from unittest.mock import Mock

from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager
from horde_worker_regen.system_resources import SystemResourceReader, SystemResourceSnapshot


def write_fixture(root: Path, relative_path: str, content: str) -> None:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def test_resource_reader_collects_linux_and_jetson_metrics(tmp_path: Path) -> None:
    write_fixture(
        tmp_path,
        "proc/meminfo",
        "MemTotal:       31733760 kB\nMemAvailable:    5242880 kB\n"
        "SwapTotal:      67108864 kB\nSwapFree:       66060288 kB\n",
    )
    write_fixture(tmp_path, "proc/loadavg", "1.25 2.50 3.75 2/100 123\n")
    write_fixture(tmp_path, "proc/100/status", "Name:\tworker\nVmRSS:\t204800 kB\n")
    write_fixture(tmp_path, "proc/100/task/100/children", "101 102\n")
    write_fixture(tmp_path, "proc/101/status", "Name:\tinference\nVmRSS:\t20971520 kB\n")
    write_fixture(tmp_path, "proc/101/task/101/children", "\n")
    write_fixture(tmp_path, "proc/102/status", "Name:\tsafety\nVmRSS:\t6291456 kB\n")
    write_fixture(tmp_path, "proc/102/task/102/children", "\n")
    write_fixture(tmp_path, "sys/class/devfreq/17000000.gv11b/cur_freq", "1377000000\n")
    write_fixture(tmp_path, "sys/class/thermal/thermal_zone0/type", "GPU-therm\n")
    write_fixture(tmp_path, "sys/class/thermal/thermal_zone0/temp", "43000\n")
    write_fixture(tmp_path, "sys/devices/pwm-fan/target_pwm", "255\n")

    snapshot = SystemResourceReader(root=tmp_path).snapshot(pid=100)

    assert snapshot.ram_used_bytes == (31_733_760 - 5_242_880) * 1024
    assert snapshot.swap_used_bytes == (67_108_864 - 66_060_288) * 1024
    assert snapshot.worker_process_bytes == (204_800 + 20_971_520 + 6_291_456) * 1024
    assert snapshot.load_average == (1.25, 2.5, 3.75)
    assert snapshot.gpu_frequency_hz == 1_377_000_000
    assert snapshot.gpu_temperature_c == 43.0
    assert snapshot.fan_pwm == 255
    assert "GPU 1377 MHz" in snapshot.format_console()
    assert "Worker memory 26.2 GiB" in snapshot.format_console()


def test_process_manager_formats_shared_resource_snapshot() -> None:
    snapshot = SystemResourceSnapshot(
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
    manager = object.__new__(HordeWorkerProcessManager)
    manager._system_resource_reader = Mock()
    manager._system_resource_reader.snapshot.return_value = snapshot

    assert manager.get_system_resource_status_line() == snapshot.format_console()
    manager._system_resource_reader.snapshot.assert_called_once()


def test_resource_reader_falls_back_to_proc_stat_when_children_file_is_unavailable(tmp_path: Path) -> None:
    write_fixture(tmp_path, "proc/100/status", "VmRSS:\t1024 kB\n")
    write_fixture(tmp_path, "proc/101/status", "VmRSS:\t2048 kB\n")
    write_fixture(tmp_path, "proc/100/stat", "100 (worker parent) S 1 0 0 0 0\n")
    write_fixture(tmp_path, "proc/101/stat", "101 (inference child) S 100 0 0 0 0\n")

    assert SystemResourceReader(root=tmp_path)._read_process_tree_memory(100) == 3072 * 1024


def test_resource_reader_reports_available_ram_without_full_process_scan(tmp_path: Path) -> None:
    write_fixture(
        tmp_path,
        "proc/meminfo",
        "MemTotal:       31733760 kB\nMemAvailable:    8388608 kB\n",
    )

    assert SystemResourceReader(root=tmp_path).available_ram_bytes() == 8 * 1024**3
