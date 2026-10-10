"""Tests for report formatting helpers."""

import pytest

from disc2plex.report import format_duration, format_size


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [(0, "0:00"), (59.6, "1:00"), (184.5, "3:04"), (2580.123, "43:00"), (5025, "1:23:45")],
)
def test_format_duration(seconds: float, expected: str) -> None:
    assert format_duration(seconds) == expected


@pytest.mark.parametrize(
    ("size", "expected"),
    [(0, "0 B"), (1023, "1023 B"), (1536, "1.5 KiB"), (1503238553, "1.4 GiB"), (2**42, "4.0 TiB")],
)
def test_format_size(size: int, expected: str) -> None:
    assert format_size(size) == expected
