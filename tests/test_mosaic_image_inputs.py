"""Validation tests for the ``MosaicImageInputs`` constructor model."""

from __future__ import annotations

import pytest

from mosaic.mosaic_image_inputs import MosaicImageInputs


def test_accepts_image_path_source() -> None:
    """Accept a non-empty filesystem path as the sole source."""
    obj_inputs = MosaicImageInputs(str_input_image_path="input.png")
    assert obj_inputs.str_input_image_path == "input.png"
    assert obj_inputs.list_bytes_frame_image_buffers is None


def test_accepts_frame_buffer_source() -> None:
    """Accept a non-empty list of image buffers as the sole source."""
    list_bytes_buffers = [b"frame-a", b"frame-b"]
    obj_inputs = MosaicImageInputs(list_bytes_frame_image_buffers=list_bytes_buffers)
    assert obj_inputs.str_input_image_path is None
    assert obj_inputs.list_bytes_frame_image_buffers == list_bytes_buffers


def test_rejects_missing_input_sources() -> None:
    """Reject construction when no source input is provided."""
    with pytest.raises(ValueError):
        MosaicImageInputs()


def test_rejects_both_input_sources() -> None:
    """Reject construction when both path and frame buffers are provided."""
    with pytest.raises(ValueError):
        MosaicImageInputs(
            str_input_image_path="input.png",
            list_bytes_frame_image_buffers=[b"frame-a"],
        )


def test_rejects_empty_frame_buffer_entry() -> None:
    """Reject frame buffer lists containing empty byte payloads."""
    with pytest.raises(ValueError):
        MosaicImageInputs(list_bytes_frame_image_buffers=[b"", b"frame-b"])


def test_rejects_empty_frame_buffer_list() -> None:
    """Reject frame-buffer source when list is present but empty."""
    with pytest.raises(ValueError):
        MosaicImageInputs(list_bytes_frame_image_buffers=[])
