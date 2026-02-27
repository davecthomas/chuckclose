"""Tests for video assembly from generated image buffers."""

from __future__ import annotations

import io
from pathlib import Path

from PIL import Image
import pytest

from mosaic import mosaic_generator
from mosaic.mosaic_generator import Mosaic


def build_png_bytes(tuple_size: tuple[int, int], tuple_color: tuple[int, int, int]) -> bytes:
    """Create an in-memory PNG byte payload."""
    image_frame = Image.new("RGB", tuple_size, tuple_color)
    obj_buffer = io.BytesIO()
    image_frame.save(obj_buffer, format="PNG")
    bytes_result = obj_buffer.getvalue()
    return bytes_result


class VideoFragmentRecorder:
    """Capture frame sizes passed into save_video_fragment for assertions."""

    def __init__(self) -> None:
        self.list_tuple_recorded_sizes: list[tuple[int, int]] = []

    def save_video_fragment(
        self,
        self_mosaic: Mosaic,
        image_buffer: Image.Image,
        str_path: str,
        int_num_frames: int = 1,
        int_fps: int = 30,
    ) -> None:
        self.list_tuple_recorded_sizes.append(image_buffer.size)


def test_generate_video_from_image_buffers_normalizes_frame_sizes(
    path_input_image: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Resize mismatched frames to the first frame size before video append."""
    obj_recorder = VideoFragmentRecorder()

    monkeypatch.setattr(mosaic_generator, "bool_has_video_support", False)
    monkeypatch.setattr(Mosaic, "save_video_fragment", obj_recorder.save_video_fragment)

    obj_mosaic = Mosaic(str(path_input_image))
    list_bytes_buffers = [
        build_png_bytes((10, 10), (10, 10, 10)),
        build_png_bytes((20, 20), (20, 20, 20)),
    ]

    str_output_path = str(tmp_path / "storyboard.mp4")
    obj_mosaic.generate_video_from_image_buffers(list_bytes_buffers, str_output_path, int_fps=30)

    assert obj_recorder.list_tuple_recorded_sizes == [(10, 10), (10, 10)]


def test_generate_video_from_image_buffers_rejects_empty_input(path_input_image: Path, tmp_path: Path) -> None:
    """Fail fast when no image buffers are provided."""
    obj_mosaic = Mosaic(str(path_input_image))
    str_output_path = str(tmp_path / "storyboard.mp4")

    with pytest.raises(ValueError):
        obj_mosaic.generate_video_from_image_buffers([], str_output_path, int_fps=30)
