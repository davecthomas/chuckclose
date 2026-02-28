"""CLI tests for storyboard-driven mosaic video mode."""

from __future__ import annotations

import io
import sys

from PIL import Image
import pytest

from mosaic import mosaic_generator
from mosaic.mosaic_generator import Mosaic
from mosaic.mosaic_settings import MosaicSettings


def build_png_bytes(tuple_size: tuple[int, int], tuple_color: tuple[int, int, int]) -> bytes:
    """Create deterministic in-memory PNG bytes for storyboard stubs."""
    image_frame = Image.new("RGB", tuple_size, tuple_color)
    obj_buffer = io.BytesIO()
    image_frame.save(obj_buffer, format="PNG")
    bytes_result = obj_buffer.getvalue()
    return bytes_result


class StubVideoStoryboard:
    """Storyboard stub that avoids external AI calls in CLI tests."""

    def __init__(
        self,
        str_storyboard_prompt: str,
        int_num_frames: int,
        int_frames_per_image: int = 3,
    ) -> None:
        self.str_storyboard_prompt = str_storyboard_prompt
        self.int_num_frames = int_num_frames
        self.int_frames_per_image = int_frames_per_image

    def generate_storyboard(self) -> list[bytes]:
        """Return deterministic storyboard image buffers for constructor input."""
        list_bytes_frames: list[bytes] = []
        for int_index in range(self.int_num_frames):
            int_channel = (int_index * 47) % 255
            bytes_frame = build_png_bytes((12, 12), (int_channel, 20, 200))
            list_bytes_frames.append(bytes_frame)
        return list_bytes_frames


class GenerateVideoRecorder:
    """Capture `Mosaic.generate_video` calls for assertion."""

    def __init__(self) -> None:
        self.bool_called = False
        self.int_duration: int | None = None
        self.int_fps: int | None = None
        self.str_output_path: str | None = None
        self.obj_start_settings: MosaicSettings | None = None
        self.obj_end_settings: MosaicSettings | None = None

    def generate_video(
        self,
        obj_mosaic: Mosaic,
        obj_start_settings: MosaicSettings,
        obj_end_settings: MosaicSettings,
        int_duration: int,
        str_output_path: str,
        int_fps: int = 30,
    ) -> None:
        """Record video call inputs without invoking OpenCV runtime dependencies."""
        self.bool_called = True
        self.obj_start_settings = obj_start_settings
        self.obj_end_settings = obj_end_settings
        self.int_duration = int_duration
        self.str_output_path = str_output_path
        self.int_fps = int_fps


def test_main_storyboard_prompt_mode_routes_to_generate_video(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure CLI storyboard mode accepts prompt input and invokes video rendering."""
    obj_recorder = GenerateVideoRecorder()

    import mosaic.video_storyboard as video_storyboard_module

    monkeypatch.setattr(video_storyboard_module, "VideoStoryboard", StubVideoStoryboard)
    monkeypatch.setattr(Mosaic, "generate_video", obj_recorder.generate_video)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mosaic",
            "--storyboard_prompt",
            "A subject turns left then right.",
            "--storyboard_num_frames",
            "4",
            "--storyboard_frames_per_image",
            "2",
            "--grid_size",
            "16",
            "--fps",
            "12",
        ],
    )

    mosaic_generator.main()

    assert obj_recorder.bool_called is True
    assert obj_recorder.int_duration == 4
    assert obj_recorder.int_fps == 12
    assert obj_recorder.obj_start_settings is not None
    assert obj_recorder.obj_end_settings is not None
