"""Unit tests for helper functions in the mosaic CLI module."""

import importlib.metadata

import pytest

from mosaic import mosaic_generator


def test_lerp_float_midpoint() -> None:
    """Validate float interpolation at the midpoint."""
    float_value = mosaic_generator.lerp_float(10.0, 20.0, 0.5)
    assert float_value == 15.0


def test_lerp_int_rounding_behavior() -> None:
    """Validate integer interpolation uses rounded values."""
    int_value = mosaic_generator.lerp_int(1, 4, 0.5)
    assert int_value == 2


def test_safe_parse_int_valid() -> None:
    """Validate integer parsing returns an int for valid input."""
    int_value = mosaic_generator.safe_parse_int("12", "grid_size")
    assert int_value == 12


def test_safe_parse_int_invalid_exits() -> None:
    """Validate integer parsing exits with code 1 for invalid values."""
    with pytest.raises(SystemExit) as obj_exc_info:
        mosaic_generator.safe_parse_int("not-an-int", "grid_size")
    assert obj_exc_info.value.code == 1


def test_safe_parse_float_valid() -> None:
    """Validate float parsing returns a float for valid input."""
    float_value = mosaic_generator.safe_parse_float("1.25", "blur_factor")
    assert float_value == 1.25


def test_safe_parse_float_invalid_exits() -> None:
    """Validate float parsing exits with code 1 for invalid values."""
    with pytest.raises(SystemExit) as obj_exc_info:
        mosaic_generator.safe_parse_float("not-a-float", "blur_factor")
    assert obj_exc_info.value.code == 1


def test_get_version_returns_unknown_when_package_not_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Validate version fallback behavior when package metadata is unavailable."""

    def _raise_package_not_found(str_name: str) -> str:
        """Raise package not found for version lookup."""
        raise importlib.metadata.PackageNotFoundError(str_name)

    monkeypatch.setattr(mosaic_generator.importlib.metadata, "version", _raise_package_not_found)
    str_version = mosaic_generator.get_version()
    assert str_version == "unknown"
