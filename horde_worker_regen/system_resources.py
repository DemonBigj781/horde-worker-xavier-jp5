"""Low-overhead Linux and Jetson resource sampling."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _gib(value: int) -> float:
    return value / 1024**3


@dataclass(frozen=True)
class SystemResourceSnapshot:
    """System and worker-process resource values collected together."""

    ram_total_bytes: int
    ram_available_bytes: int
    swap_total_bytes: int
    swap_free_bytes: int
    worker_process_bytes: int
    load_average: tuple[float, float, float]
    gpu_frequency_hz: int | None = None
    gpu_temperature_c: float | None = None
    fan_pwm: int | None = None

    @property
    def ram_used_bytes(self) -> int:
        return max(0, self.ram_total_bytes - self.ram_available_bytes)

    @property
    def swap_used_bytes(self) -> int:
        return max(0, self.swap_total_bytes - self.swap_free_bytes)

    def format_console(self) -> str:
        """Format a compact status line suitable for periodic logging."""
        ram_percent = self.ram_used_bytes / self.ram_total_bytes * 100 if self.ram_total_bytes else 0
        parts = [
            f"RAM {_gib(self.ram_used_bytes):.1f}/{_gib(self.ram_total_bytes):.1f} GiB ({ram_percent:.0f}%)",
            f"available {_gib(self.ram_available_bytes):.1f} GiB",
            f"Swap {_gib(self.swap_used_bytes):.1f}/{_gib(self.swap_total_bytes):.1f} GiB",
            f"Worker memory {_gib(self.worker_process_bytes):.1f} GiB",
            f"Load {self.load_average[0]:.2f}/{self.load_average[1]:.2f}/{self.load_average[2]:.2f}",
        ]
        if self.gpu_frequency_hz is not None:
            parts.append(f"GPU {self.gpu_frequency_hz / 1_000_000:.0f} MHz")
        if self.gpu_temperature_c is not None:
            parts.append(f"GPU {self.gpu_temperature_c:.0f} C")
        if self.fan_pwm is not None:
            parts.append(f"Fan {self.fan_pwm / 255 * 100:.0f}%")
        return " | ".join(parts)

    def format_tui(self) -> str:
        """Format the same sample across readable TUI lines."""
        fields = self.format_console().split(" | ")
        return "\n".join(("  ".join(fields[:2]), "  ".join(fields[2:5]), "  ".join(fields[5:])))


class SystemResourceReader:
    """Read resource values directly from procfs and sysfs."""

    def __init__(self, root: Path = Path("/")) -> None:
        """Create a reader rooted at the real filesystem or a test fixture."""
        self.root = root

    def _path(self, absolute_path: str) -> Path:
        return self.root / absolute_path.lstrip("/")

    @staticmethod
    def _read_int(path: Path) -> int | None:
        try:
            return int(path.read_text().strip())
        except (OSError, ValueError):
            return None

    def _read_meminfo(self) -> dict[str, int]:
        values: dict[str, int] = {}
        try:
            lines = self._path("/proc/meminfo").read_text().splitlines()
        except OSError:
            return values
        for line in lines:
            name, separator, raw_value = line.partition(":")
            if not separator:
                continue
            try:
                values[name] = int(raw_value.strip().split()[0]) * 1024
            except (IndexError, ValueError):
                continue
        return values

    def _read_load_average(self) -> tuple[float, float, float]:
        try:
            values = self._path("/proc/loadavg").read_text().split()[:3]
            return float(values[0]), float(values[1]), float(values[2])
        except (OSError, IndexError, ValueError):
            return 0.0, 0.0, 0.0

    def _read_process_memory(self, pid: int) -> int:
        try:
            lines = self._path(f"/proc/{pid}/smaps_rollup").read_text().splitlines()
        except OSError:
            lines = []
        for line in lines:
            if line.startswith("Pss:"):
                try:
                    return int(line.split()[1]) * 1024
                except (IndexError, ValueError):
                    break

        try:
            lines = self._path(f"/proc/{pid}/status").read_text().splitlines()
        except OSError:
            return 0
        for line in lines:
            if line.startswith("VmRSS:"):
                try:
                    return int(line.split()[1]) * 1024
                except (IndexError, ValueError):
                    return 0
        return 0

    def _read_children(self, pid: int) -> list[int]:
        try:
            content = self._path(f"/proc/{pid}/task/{pid}/children").read_text()
        except OSError:
            content = ""
        children = [int(value) for value in content.split() if value.isdigit()]
        if children:
            return children

        fallback_children: list[int] = []
        for stat_path in self._path("/proc").glob("[0-9]*/stat"):
            try:
                stat = stat_path.read_text()
                closing_parenthesis = stat.rfind(")")
                fields = stat[closing_parenthesis + 2 :].split()
                parent_pid = int(fields[1])
                child_pid = int(stat_path.parent.name)
            except (OSError, IndexError, ValueError):
                continue
            if parent_pid == pid:
                fallback_children.append(child_pid)
        return fallback_children

    def _read_process_tree_memory(self, pid: int) -> int:
        pending = [pid]
        visited: set[int] = set()
        total = 0
        while pending:
            current = pending.pop()
            if current in visited:
                continue
            visited.add(current)
            total += self._read_process_memory(current)
            pending.extend(self._read_children(current))
        return total

    def _read_gpu_frequency(self) -> int | None:
        candidates = [
            self._path("/sys/devices/17000000.gv11b/devfreq/17000000.gv11b/cur_freq"),
            self._path("/sys/class/devfreq/17000000.gv11b/cur_freq"),
            self._path("/sys/class/devfreq/17000000.gpu/cur_freq"),
        ]
        candidates.extend(sorted(self._path("/sys/class/devfreq").glob("*gpu*/cur_freq")))
        candidates.extend(sorted(self._path("/sys/class/devfreq").glob("*gv11b*/cur_freq")))
        for path in candidates:
            value = self._read_int(path)
            if value is not None:
                return value
        return None

    def _read_gpu_temperature(self) -> float | None:
        for type_path in sorted(self._path("/sys/class/thermal").glob("thermal_zone*/type")):
            try:
                thermal_type = type_path.read_text().strip().lower()
            except OSError:
                continue
            if "gpu" not in thermal_type:
                continue
            temperature = self._read_int(type_path.with_name("temp"))
            if temperature is not None:
                return temperature / 1000
        return None

    def _read_fan_pwm(self) -> int | None:
        candidates = [self._path("/sys/devices/pwm-fan/target_pwm")]
        candidates.extend(sorted(self._path("/sys/class/hwmon").glob("hwmon*/pwm1")))
        for path in candidates:
            value = self._read_int(path)
            if value is not None:
                return value
        return None

    def snapshot(self, pid: int | None = None) -> SystemResourceSnapshot:
        """Collect one resource snapshot."""
        meminfo = self._read_meminfo()
        return SystemResourceSnapshot(
            ram_total_bytes=meminfo.get("MemTotal", 0),
            ram_available_bytes=meminfo.get("MemAvailable", 0),
            swap_total_bytes=meminfo.get("SwapTotal", 0),
            swap_free_bytes=meminfo.get("SwapFree", 0),
            worker_process_bytes=self._read_process_tree_memory(pid or os.getpid()),
            load_average=self._read_load_average(),
            gpu_frequency_hz=self._read_gpu_frequency(),
            gpu_temperature_c=self._read_gpu_temperature(),
            fan_pwm=self._read_fan_pwm(),
        )
