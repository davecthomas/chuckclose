"""Integration tests for real MP4 generation and decoding."""

from __future__ import annotations

import io
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import pytest

from mosaic import mosaic_generator
from mosaic.mosaic_generator import Mosaic
from mosaic.mosaic_image_inputs import MosaicImageInputs
from mosaic.mosaic_settings import MosaicSettings


pytestmark = pytest.mark.real_video


EXPECTED_PATH_INPUT_IMAGE_SIZE: tuple[int, int] = (20, 20)
FRAME_BUFFER_TEST_SIZE: tuple[int, int] = (24, 24)
VIDEO_FRAME_COUNT_THREE: int = 3
VIDEO_FRAME_COUNT_FOUR: int = 4
VIDEO_TEST_FPS_SIX: int = 6
VIDEO_TEST_FPS_NINE: int = 9
VIDEO_TEST_FPS_FOUR: int = 4
MEAN_COLOR_DIFFERENCE_THRESHOLD: float = 25.0


def build_png_bytes(
    tuple_size: tuple[int, int], tuple_color: tuple[int, int, int]
) -> bytes:
    """Create one in-memory PNG buffer for video-integration testing.

    Purpose:
    - Build deterministic RGB frame payloads used to verify real MP4 encoding
      and decoding behavior in the integration suite.

    Inputs:
    - `tuple_size`: Two-integer `(width, height)` frame size. Both dimensions
      must be positive because Pillow image creation requires positive extents.
    - `tuple_color`: Three-integer `(red, green, blue)` fill color tuple. Each
      value should be in the inclusive range `0..255`.

    Output:
    - Returns one non-empty PNG byte buffer representing the requested solid
      color frame.
    """
    image_frame = Image.new("RGB", tuple_size, tuple_color)
    obj_buffer = io.BytesIO()
    image_frame.save(obj_buffer, format="PNG")
    bytes_result: bytes = obj_buffer.getvalue()
    # Normal return with one encoded PNG frame buffer.
    return bytes_result


def read_video_frame_count_and_size(
    path_video_file: Path,
) -> tuple[int, tuple[int, int]]:
    """Read one generated MP4 and resolve basic structural metadata.

    Purpose:
    - Verify that real MP4 outputs written by the mosaic runtime are decodable.
    - Resolve exact frame count plus output dimensions for test assertions.

    Inputs:
    - `path_video_file`: Filesystem path to an existing local MP4 artifact.

    Output:
    - Returns a tuple `(int_frame_count, tuple_size)` where `tuple_size` is one
      `(width, height)` pair.
    - Raises `ValueError` if the capture backend cannot resolve a positive frame
      count or output size from the generated file.
    """
    capture = cv2.VideoCapture(str(path_video_file))
    try:
        if not capture.isOpened():
            raise ValueError("Unable to open generated video for decoding.")

        int_width: int = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        int_height: int = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        tuple_size: tuple[int, int] = (int_width, int_height)
        int_reported_frame_count: int = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        int_frame_count: int = int_reported_frame_count
        if int_frame_count < 1:
            int_frame_count = 0
            # Count decoded frames when the backend does not expose direct frame-count metadata reliably.
            while True:
                bool_has_frame: bool
                array_frame: np.ndarray
                bool_has_frame, array_frame = capture.read()
                if not bool_has_frame:
                    break
                int_frame_count += 1

        if int_frame_count < 1 or int_width < 1 or int_height < 1:
            raise ValueError("Generated video did not contain any decodable frames.")
    finally:
        capture.release()
    # Normal return with resolved frame count and output dimensions.
    return int_frame_count, tuple_size


def read_video_frame_mean_color(
    path_video_file: Path, int_frame_index: int
) -> tuple[float, float, float]:
    """Read one decoded frame and summarize its mean RGB color.

    Purpose:
    - Confirm that encoded frame content changes across a generated MP4 rather
      than only asserting file existence and frame count.

    Inputs:
    - `path_video_file`: Filesystem path to an existing local MP4 file.
    - `int_frame_index`: Zero-based decoded frame index to inspect. The caller
      must provide an index that exists within the video bounds.

    Output:
    - Returns a three-float tuple `(red_mean, green_mean, blue_mean)` computed
      from the decoded RGB frame.
    """
    capture = cv2.VideoCapture(str(path_video_file))
    try:
        if not capture.isOpened():
            raise ValueError("Unable to open generated video for frame inspection.")

        bool_positioned: bool = capture.set(cv2.CAP_PROP_POS_FRAMES, int_frame_index)
        if not bool_positioned:
            raise ValueError(
                f"Unable to seek to generated video frame index {int_frame_index}."
            )

        bool_has_frame: bool
        array_bgr_frame: np.ndarray
        bool_has_frame, array_bgr_frame = capture.read()
        if not bool_has_frame:
            raise ValueError(
                f"Unable to decode generated video frame index {int_frame_index}."
            )
        array_frame: np.ndarray = cv2.cvtColor(array_bgr_frame, cv2.COLOR_BGR2RGB)
        tuple_mean_color: tuple[float, float, float] = (
            float(array_frame[:, :, 0].mean()),
            float(array_frame[:, :, 1].mean()),
            float(array_frame[:, :, 2].mean()),
        )
    finally:
        capture.release()
    # Normal return with the decoded frame mean-color summary.
    return tuple_mean_color


def test_generate_video_writes_decodable_mp4(
    path_input_image: Path, tmp_path: Path
) -> None:
    """Generate a real MP4 through `Mosaic.generate_video` and decode it back.

    Purpose:
    - Prove that the primary temporal renderer writes a valid MP4 in the normal
      test suite without mocking the OpenCV writer path.

    Inputs:
    - `path_input_image`: Deterministic source-image fixture created by
      `tests.conftest`.
    - `tmp_path`: Per-test temporary directory provided by pytest.

    Output:
    - Returns `None` after asserting that the generated MP4 exists, is non-zero
      in size, and decodes to the expected frame count and dimensions.
    """
    obj_mosaic_inputs = MosaicImageInputs(str_input_image_path=str(path_input_image))
    obj_mosaic = Mosaic(obj_mosaic_inputs)
    obj_settings = MosaicSettings(int_grid_size=5, float_blur_factor=0.0)
    path_output_video = tmp_path / "generated_from_render.mp4"

    obj_mosaic.generate_video(
        obj_settings,
        obj_settings,
        int_duration=VIDEO_FRAME_COUNT_THREE,
        str_output_path=str(path_output_video),
        int_fps=VIDEO_TEST_FPS_SIX,
    )

    int_frame_count: int
    tuple_size: tuple[int, int]
    int_frame_count, tuple_size = read_video_frame_count_and_size(path_output_video)

    assert path_output_video.exists()
    assert path_output_video.suffix == ".mp4"
    assert path_output_video.stat().st_size > 0
    assert int_frame_count == VIDEO_FRAME_COUNT_THREE
    assert tuple_size == EXPECTED_PATH_INPUT_IMAGE_SIZE
    # Normal return after validating the real rendered MP4 artifact.
    return None


def test_generate_video_from_image_buffers_writes_decodable_mp4(
    path_input_image: Path, tmp_path: Path
) -> None:
    """Generate a real MP4 from raw frame buffers and verify decoded content.

    Purpose:
    - Prove that the buffer-to-video encoder path writes a valid MP4 in the
      normal test suite without mocking `save_video_fragment`.

    Inputs:
    - `path_input_image`: Deterministic source-image fixture used only to create
      the `Mosaic` instance required by the encoder API.
    - `tmp_path`: Per-test temporary directory provided by pytest.

    Output:
    - Returns `None` after asserting that the generated MP4 exists, has the
      expected frame count and dimensions, and preserves visible temporal
      differences between the first and last frames.
    """
    obj_mosaic_inputs = MosaicImageInputs(str_input_image_path=str(path_input_image))
    obj_mosaic = Mosaic(obj_mosaic_inputs)
    list_bytes_frame_buffers: list[bytes] = [
        build_png_bytes(FRAME_BUFFER_TEST_SIZE, (240, 20, 20)),
        build_png_bytes(FRAME_BUFFER_TEST_SIZE, (20, 240, 20)),
        build_png_bytes(FRAME_BUFFER_TEST_SIZE, (20, 20, 240)),
    ]
    path_output_video = tmp_path / "generated_from_buffers.mp4"

    obj_mosaic.generate_video_from_image_buffers(
        list_bytes_frame_buffers,
        str(path_output_video),
        int_fps=VIDEO_TEST_FPS_NINE,
    )

    int_frame_count: int
    tuple_size: tuple[int, int]
    int_frame_count, tuple_size = read_video_frame_count_and_size(path_output_video)
    tuple_first_frame_mean_color = read_video_frame_mean_color(path_output_video, 0)
    tuple_last_frame_mean_color = read_video_frame_mean_color(
        path_output_video,
        VIDEO_FRAME_COUNT_THREE - 1,
    )
    float_total_mean_color_difference: float = 0.0
    # Sum absolute channel differences to confirm the encoded first and last frames remain visually distinct.
    for float_first_channel, float_last_channel in zip(
        tuple_first_frame_mean_color,
        tuple_last_frame_mean_color,
        strict=True,
    ):
        float_total_mean_color_difference += abs(
            float_first_channel - float_last_channel
        )

    assert path_output_video.exists()
    assert path_output_video.suffix == ".mp4"
    assert path_output_video.stat().st_size > 0
    assert int_frame_count == VIDEO_FRAME_COUNT_THREE
    assert tuple_size == FRAME_BUFFER_TEST_SIZE
    assert float_total_mean_color_difference > MEAN_COLOR_DIFFERENCE_THRESHOLD
    # Normal return after validating the real buffer-encoded MP4 artifact.
    return None


def test_cli_video_mode_writes_decodable_mp4(
    path_input_image: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run the CLI video mode end to end and decode the generated MP4 output.

    Purpose:
    - Prove that the normal CLI `--video` path writes a valid MP4 in the
      default test suite without mocking the rendering pipeline.

    Inputs:
    - `path_input_image`: Deterministic source-image fixture created by pytest.
    - `tmp_path`: Per-test temporary directory where the CLI writes `output/`.
    - `monkeypatch`: Pytest helper used only to set the working directory and
      command-line arguments for the CLI invocation.

    Output:
    - Returns `None` after asserting that the CLI created one decodable MP4 with
      the expected frame count and dimensions.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mosaic",
            str(path_input_image),
            "--video",
            str(VIDEO_FRAME_COUNT_FOUR),
            "--fps",
            str(VIDEO_TEST_FPS_FOUR),
            "--grid_size",
            "5",
        ],
    )

    mosaic_generator.main()

    list_path_output_videos: list[Path] = list(
        (tmp_path / "output").glob("*_mosaic_standard_4f_*.mp4")
    )
    assert len(list_path_output_videos) == 1

    path_output_video: Path = list_path_output_videos[0]
    int_frame_count: int
    tuple_size: tuple[int, int]
    int_frame_count, tuple_size = read_video_frame_count_and_size(path_output_video)

    assert path_output_video.exists()
    assert path_output_video.stat().st_size > 0
    assert int_frame_count == VIDEO_FRAME_COUNT_FOUR
    assert tuple_size == EXPECTED_PATH_INPUT_IMAGE_SIZE
    # Normal return after validating the real CLI-generated MP4 artifact.
    return None
