"""Runtime-compatible Python APIs used by the Xavier Python 3.10 port."""

from __future__ import annotations

try:
    from enum import StrEnum
except ImportError:
    from strenum import LowercaseStrEnum as StrEnum

__all__ = ["StrEnum"]
