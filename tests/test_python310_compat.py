from enum import auto

from horde_worker_regen.python_compat import StrEnum


def test_strenum_auto_matches_upstream_python_behavior() -> None:
    """The Python 3.10 backport must preserve stdlib StrEnum's lowercase auto values."""

    class Example(StrEnum):
        SOME_VALUE = auto()
        EXPLICIT = "explicit"

    assert Example.SOME_VALUE.value == "some_value"
    assert str(Example.SOME_VALUE) == "some_value"
    assert Example.EXPLICIT.value == "explicit"
