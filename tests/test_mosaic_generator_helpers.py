"""Unit tests for helper functions in the mosaic CLI module."""

import importlib.metadata

import pytest

from mosaic import mosaic_generator


def raise_package_not_found(str_name: str) -> str:
    """Raise package not found for version lookup."""
    raise importlib.metadata.PackageNotFoundError(str_name)


def test_lerp_float_midpoint() -> None:
    """Validate float interpolation at the midpoint."""
    float_value = mosaic_generator.lerp_float(10.0, 20.0, 0.5)
    assert float_value == 15.0


def test_lerp_int_rounding_behavior() -> None:
    """Validate integer interpolation uses rounded values."""
    int_value = mosaic_generator.lerp_int(1, 4, 0.5)
    assert int_value == 2


def test_get_version_returns_unknown_when_package_not_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Validate version fallback behavior when package metadata is unavailable."""
    monkeypatch.setattr(mosaic_generator.importlib.metadata, "version", raise_package_not_found)
    str_version = mosaic_generator.get_version()
    assert str_version == "unknown"
