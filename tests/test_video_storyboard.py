"""Tests for storyboard prompt planning plus image and video source generation."""

from __future__ import annotations

import io
import importlib
import json
from pathlib import Path
from typing import Any, TypeVar

from PIL import Image
import pytest

from ai_api_unified import AIBaseVideoProperties
from ai_api_unified import AIStructuredPrompt
from ai_api_unified import AIVideoArtifact
from ai_api_unified import AIVideoGenerationJob
from ai_api_unified import AIVideoGenerationResult
from ai_api_unified import AIVideoGenerationStatus
from mosaic import mosaic_generator
from mosaic.mosaic_generator import Mosaic
from mosaic.mosaic_image_inputs import MosaicImageInputs
from mosaic.video_storyboard import VideoStoryboard

TypeStructuredPrompt = TypeVar("TypeStructuredPrompt", bound=AIStructuredPrompt)


class StubStoryboardAiClient:
    """Simple AI client stub for deterministic storyboard tests."""

    def __init__(
        self,
        str_send_prompt_response: str,
        path_video_artifact: Path | None = None,
        bytes_video_artifact: bytes | None = None,
    ) -> None:
        self.str_send_prompt_response = str_send_prompt_response
        self.path_video_artifact = path_video_artifact
        self.bytes_video_artifact = bytes_video_artifact or b"video-artifact"
        self.list_str_image_prompts: list[str] = []
        self.list_str_video_prompts: list[str] = []
        self.list_int_last_frame_indices: list[int] | None = None
        self.obj_last_video_properties: AIBaseVideoProperties | None = None
        self.str_last_video_model_name: str | None = None
        self.int_image_calls = 0
        self.int_video_calls = 0

    def send_structured_prompt(
        self,
        obj_structured_prompt: AIStructuredPrompt,
        cls_response_model: type[TypeStructuredPrompt],
    ) -> TypeStructuredPrompt:
        """Return configured storyboard planning response as structured payload."""
        dict_payload = json.loads(self.str_send_prompt_response)
        obj_response: TypeStructuredPrompt = cls_response_model.model_validate(
            dict_payload
        )
        return obj_response

    def create_image(self, str_prompt: str) -> bytes:
        """Return deterministic image bytes per prompt order."""
        self.int_image_calls += 1
        self.list_str_image_prompts.append(str_prompt)
        bytes_result = f"image-{self.int_image_calls}".encode("utf-8")
        return bytes_result

    def create_video(
        self,
        str_prompt: str,
        obj_video_properties: AIBaseVideoProperties,
        str_video_engine: str | None = None,
        str_video_model_name: str | None = None,
    ) -> AIVideoGenerationResult:
        """Return one deterministic materialized video artifact result."""
        if self.path_video_artifact is None:
            raise AssertionError(
                "path_video_artifact must be configured for video tests."
            )

        self.int_video_calls += 1
        self.list_str_video_prompts.append(str_prompt)
        self.obj_last_video_properties = obj_video_properties
        self.str_last_video_model_name = str_video_model_name

        obj_job = AIVideoGenerationJob(
            job_id="job-1",
            provider_job_id="provider-job-1",
            status=AIVideoGenerationStatus.COMPLETED,
            provider_engine="google-gemini",
            provider_model_name="veo-test",
        )
        obj_artifact = AIVideoArtifact(
            file_path=self.path_video_artifact,
            duration_seconds=8,
        )
        obj_result = AIVideoGenerationResult(job=obj_job, artifacts=[obj_artifact])
        return obj_result

    def extract_video_frames(
        self,
        bytes_video: bytes,
        list_float_time_offsets: list[float] | None = None,
        list_int_frame_indices: list[int] | None = None,
    ) -> list[bytes]:
        """Return deterministic extracted frame bytes for the requested indices."""
        self.list_int_last_frame_indices = list(list_int_frame_indices or [])
        if bytes_video != self.bytes_video_artifact:
            raise AssertionError("Video extraction received unexpected video bytes.")

        list_bytes_frames: list[bytes] = []
        # Build deterministic extracted frame bytes using the exact requested frame indices.
        for int_frame_index in self.list_int_last_frame_indices:
            list_bytes_frames.append(f"frame-{int_frame_index}".encode("utf-8"))
        return list_bytes_frames


class FailingImageStoryboardAiClient(StubStoryboardAiClient):
    """AI client stub that raises an image-generation failure."""

    def create_image(self, str_prompt: str) -> bytes:
        """Raise an image-generation error."""
        raise RuntimeError("Image generation failure")


class PngStoryboardAiClient(StubStoryboardAiClient):
    """AI client stub that returns valid PNG bytes for each frame prompt."""

    def create_image(self, str_prompt: str) -> bytes:
        """Return synthetic PNG bytes with deterministic frame coloring."""
        self.int_image_calls += 1
        self.list_str_image_prompts.append(str_prompt)

        int_color_seed = (self.int_image_calls * 37) % 255
        tuple_color = (
            int_color_seed,
            (int_color_seed * 2) % 255,
            (int_color_seed * 3) % 255,
        )
        image_frame = Image.new("RGB", (24, 24), tuple_color)
        obj_buffer = io.BytesIO()
        image_frame.save(obj_buffer, format="PNG")
        bytes_result = obj_buffer.getvalue()
        return bytes_result


class VideoFragmentRecorder:
    """Capture frame sizes forwarded to `save_video_fragment` during assembly."""

    def __init__(self) -> None:
        self.list_tuple_sizes: list[tuple[int, int]] = []

    def save_video_fragment(
        self,
        image_buffer: Image.Image,
        str_path: str,
        int_num_frames: int = 1,
        int_fps: int = 30,
    ) -> None:
        """Record rendered frame sizes without invoking OpenCV video output."""
        self.list_tuple_sizes.append(image_buffer.size)


class StubImageioFfmpegModule:
    """Simple `imageio_ffmpeg` stand-in that returns one resolved backend path."""

    def __init__(self, path_ffmpeg_executable: Path) -> None:
        """Store the backend executable path returned by `get_ffmpeg_exe()`."""
        self.path_ffmpeg_executable = path_ffmpeg_executable

    def get_ffmpeg_exe(self) -> str:
        """Return one existing backend executable path for preflight validation."""
        str_result: str = str(self.path_ffmpeg_executable)
        # Normal return with an existing backend executable path.
        return str_result


def test_build_frame_prompts_normalizes_to_requested_count() -> None:
    """Build frame prompts and normalize count deterministically in image mode."""
    str_response = """
    {
      "frames": [
        {"frame_index": 0, "frame_prompt": "frame zero"},
        {"frame_index": 1, "frame_prompt": "frame one"}
      ]
    }
    """
    obj_stub_client = StubStoryboardAiClient(str_response)
    obj_storyboard = VideoStoryboard(
        str_storyboard_prompt="Subject glances left then right.",
        int_num_frames=4,
        int_frames_per_image=1,
        obj_ai_api=obj_stub_client,
    )

    list_str_prompts = obj_storyboard.build_frame_prompts()

    assert len(list_str_prompts) == 4
    assert list_str_prompts == ["frame zero", "frame zero", "frame one", "frame one"]


def test_build_frame_prompts_raises_on_invalid_payload() -> None:
    """Raise when structured storyboard planning payload is invalid."""
    obj_stub_client = StubStoryboardAiClient("not-json-response")
    obj_storyboard = VideoStoryboard(
        str_storyboard_prompt="A person blinks and looks ahead.",
        int_num_frames=3,
        int_frames_per_image=1,
        obj_ai_api=obj_stub_client,
    )

    with pytest.raises(json.JSONDecodeError):
        obj_storyboard.build_frame_prompts()


def test_generate_storyboard_image_mode_calls_create_image_in_prompt_order() -> None:
    """Generate storyboard images in image mode and persist buffers in sequence."""
    str_response = """
    {
      "frames": [
        {"frame_index": 0, "frame_prompt": "first prompt"},
        {"frame_index": 1, "frame_prompt": "second prompt"}
      ]
    }
    """
    obj_stub_client = StubStoryboardAiClient(str_response)
    obj_storyboard = VideoStoryboard(
        str_storyboard_prompt="Storyboard prompt",
        int_num_frames=2,
        int_frames_per_image=1,
        obj_ai_api=obj_stub_client,
    )

    list_bytes_images = obj_storyboard.generate_storyboard(str_storyboard_mode="image")

    assert list_bytes_images == [b"image-1", b"image-2"]
    assert obj_stub_client.list_str_image_prompts == ["first prompt", "second prompt"]
    assert obj_stub_client.int_video_calls == 0


def test_generate_storyboard_image_mode_raises_for_image_generation_error() -> None:
    """Raise image-generation exceptions from the image storyboard provider path."""
    str_response = """
    {
      "frames": [
        {"frame_index": 0, "frame_prompt": "only prompt"}
      ]
    }
    """
    obj_failing_client = FailingImageStoryboardAiClient(str_response)
    obj_storyboard = VideoStoryboard(
        str_storyboard_prompt="Storyboard prompt",
        int_num_frames=1,
        int_frames_per_image=1,
        obj_ai_api=obj_failing_client,
    )

    with pytest.raises(RuntimeError):
        obj_storyboard.generate_storyboard(str_storyboard_mode="image")


def test_generate_storyboard_defaults_to_image_mode_for_backward_compatibility() -> (
    None
):
    """Ensure omitted storyboard mode preserves the legacy image-generation path.

    Purpose:
    - Verify the public library API still defaults to image generation when the
      caller does not pass `str_storyboard_mode`.
    - Protect existing Python integrations from silently switching to the paid
      video-generation path.

    Inputs:
    - None. The test constructs one storyboard with a deterministic AI stub and
      calls `generate_storyboard()` without arguments.

    Output:
    - Returns `None` after asserting the image path runs and the video path does
      not run.
    """
    str_response = """
    {
      "frames": [
        {"frame_index": 0, "frame_prompt": "first prompt"},
        {"frame_index": 1, "frame_prompt": "second prompt"}
      ]
    }
    """
    obj_stub_client = StubStoryboardAiClient(str_response)
    obj_storyboard = VideoStoryboard(
        str_storyboard_prompt="Storyboard prompt",
        int_num_frames=2,
        int_frames_per_image=1,
        obj_ai_api=obj_stub_client,
    )

    list_bytes_images = obj_storyboard.generate_storyboard()

    assert list_bytes_images == [b"image-1", b"image-2"]
    assert obj_stub_client.int_video_calls == 0


def test_generate_storyboard_video_mode_extracts_exact_frame_indices(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Extract one clean source frame per final output frame in video mode."""
    path_video_artifact = tmp_path / "generated.mp4"
    bytes_video_artifact = b"video-artifact"
    path_video_artifact.write_bytes(bytes_video_artifact)

    obj_stub_client = StubStoryboardAiClient(
        '{"frames": []}',
        path_video_artifact=path_video_artifact,
        bytes_video_artifact=bytes_video_artifact,
    )
    obj_storyboard = VideoStoryboard(
        str_storyboard_prompt="A face turns slowly from left to right.",
        int_num_frames=4,
        int_frames_per_image=99,
        obj_ai_api=obj_stub_client,
    )
    obj_video_properties = AIBaseVideoProperties(
        duration_seconds=8,
        aspect_ratio="16:9",
        resolution="1080p",
    )

    monkeypatch.setattr(
        VideoStoryboard,
        "_read_available_video_frame_count_from_path",
        staticmethod(lambda path_video_file: 8),
    )

    list_bytes_frames = obj_storyboard.generate_storyboard(
        str_storyboard_mode="video",
        obj_video_properties=obj_video_properties,
        str_video_model_name="veo-test",
    )

    assert list_bytes_frames == [b"frame-0", b"frame-2", b"frame-5", b"frame-7"]
    assert obj_stub_client.list_int_last_frame_indices == [0, 2, 5, 7]
    assert obj_stub_client.int_video_calls == 1
    assert obj_stub_client.list_str_video_prompts == [
        "A face turns slowly from left to right."
    ]
    assert obj_stub_client.obj_last_video_properties == obj_video_properties
    assert obj_stub_client.str_last_video_model_name == "veo-test"


def test_generate_storyboard_video_mode_raises_when_request_exceeds_source_frames(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Fail fast when exact indexed extraction would require synthetic frames."""
    path_video_artifact = tmp_path / "generated.mp4"
    path_video_artifact.write_bytes(b"video-artifact")
    obj_stub_client = StubStoryboardAiClient(
        '{"frames": []}',
        path_video_artifact=path_video_artifact,
    )
    obj_storyboard = VideoStoryboard(
        str_storyboard_prompt="A person takes a step forward.",
        int_num_frames=5,
        int_frames_per_image=1,
        obj_ai_api=obj_stub_client,
    )

    monkeypatch.setattr(
        VideoStoryboard,
        "_read_available_video_frame_count_from_path",
        staticmethod(lambda path_video_file: 4),
    )

    with pytest.raises(ValueError, match="requested=5 available=4"):
        obj_storyboard.generate_storyboard(
            str_storyboard_mode="video",
            obj_video_properties=AIBaseVideoProperties(duration_seconds=8),
        )


def test_validate_video_runtime_dependencies_accepts_imageio_ffmpeg_backend(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Accept a usable `imageio-ffmpeg` backend without requiring PATH `ffmpeg`."""
    path_ffmpeg_executable = tmp_path / "ffmpeg"
    path_ffmpeg_executable.write_text("stub-binary", encoding="utf-8")
    path_ffmpeg_executable.chmod(0o755)

    def import_module_stub(str_module_name: str) -> Any:
        """Return deterministic modules for decoder preflight validation.

        Purpose:
        - Simulate the exact decoder imports used by storyboard video preflight
          without relying on the host environment.

        Inputs:
        - `str_module_name`: Requested module import name.

        Output:
        - Returns a lightweight module substitute for supported names.
        - Raises `ImportError` for any unexpected module import.
        """
        if str_module_name == "imageio":
            # Normal return with a generic object because the preflight only validates importability.
            return object()
        if str_module_name == "imageio_ffmpeg":
            obj_module = StubImageioFfmpegModule(path_ffmpeg_executable)
            # Normal return with a stub backend module that resolves one executable path.
            return obj_module
        if str_module_name == "PIL.Image":
            # Normal return with the real PIL image module because the preflight only validates importability.
            return Image
        raise ImportError(f"Unexpected module import requested: {str_module_name}")

    monkeypatch.setattr(importlib, "import_module", import_module_stub)

    VideoStoryboard.validate_video_runtime_dependencies()


def test_validate_video_runtime_dependencies_rejects_non_executable_backend(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Reject a resolved `imageio-ffmpeg` path that is not executable.

    Purpose:
    - Verify storyboard video preflight fails early when `imageio_ffmpeg`
      resolves to a file without execute permissions.
    - Prevent later runtime failures that would occur after a paid source-video
      generation request.

    Inputs:
    - `monkeypatch`: Pytest monkeypatch fixture used to replace optional module
      imports with deterministic stubs.
    - `tmp_path`: Temporary directory used to create one non-executable backend
      file.

    Output:
    - Returns `None` after asserting that preflight raises `RuntimeError` with
      executable guidance.
    """
    path_ffmpeg_executable = tmp_path / "ffmpeg"
    path_ffmpeg_executable.write_text("stub-binary", encoding="utf-8")
    path_ffmpeg_executable.chmod(0o644)

    def import_module_stub(str_module_name: str) -> Any:
        """Return deterministic modules for non-executable backend validation.

        Purpose:
        - Simulate the optional decoder imports used by storyboard video
          preflight while forcing a non-executable backend file.

        Inputs:
        - `str_module_name`: Requested module import name.

        Output:
        - Returns one lightweight module substitute for supported names.
        - Raises `ImportError` for any unsupported module import.
        """
        if str_module_name == "imageio":
            # Normal return with a generic object because the preflight only validates importability.
            return object()
        if str_module_name == "imageio_ffmpeg":
            obj_module = StubImageioFfmpegModule(path_ffmpeg_executable)
            # Normal return with a stub backend module that resolves one non-executable file.
            return obj_module
        if str_module_name == "PIL.Image":
            # Normal return with the real PIL image module because the preflight only validates importability.
            return Image
        raise ImportError(f"Unexpected module import requested: {str_module_name}")

    monkeypatch.setattr(importlib, "import_module", import_module_stub)

    with pytest.raises(RuntimeError, match="execute permissions"):
        VideoStoryboard.validate_video_runtime_dependencies()


def test_storyboard_end_to_end_image_pipeline_with_video_buffer_assembly(
    path_input_image: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Validate the image storyboard path plus downstream mosaic buffer assembly."""
    list_dict_frames = [
        {"frame_index": 0, "frame_prompt": "Frame 1: subject looks straight ahead."},
        {"frame_index": 1, "frame_prompt": "Frame 2: subject looks left."},
        {"frame_index": 2, "frame_prompt": "Frame 3: subject blinks."},
        {
            "frame_index": 3,
            "frame_prompt": "Frame 4: subject looks straight ahead again.",
        },
    ]
    str_response = json.dumps({"frames": list_dict_frames})

    obj_stub_client = PngStoryboardAiClient(str_response)
    obj_storyboard = VideoStoryboard(
        str_storyboard_prompt="Storyboard prompt",
        int_num_frames=4,
        int_frames_per_image=1,
        obj_ai_api=obj_stub_client,
    )
    list_bytes_storyboard_images = obj_storyboard.generate_storyboard(
        str_storyboard_mode="image"
    )

    assert len(list_bytes_storyboard_images) == 4
    assert len(obj_storyboard.list_str_frame_prompts) == 4
    assert all(len(bytes_item) > 0 for bytes_item in list_bytes_storyboard_images)

    obj_recorder = VideoFragmentRecorder()
    monkeypatch.setattr(mosaic_generator, "bool_has_video_support", False)
    monkeypatch.setattr(Mosaic, "save_video_fragment", obj_recorder.save_video_fragment)

    obj_mosaic_inputs = MosaicImageInputs(str_input_image_path=str(path_input_image))
    obj_mosaic = Mosaic(obj_mosaic_inputs)
    str_output_path = str(tmp_path / "storyboard_pipeline.mp4")
    obj_mosaic.generate_video_from_image_buffers(
        list_bytes_storyboard_images, str_output_path, int_fps=24
    )

    assert obj_recorder.list_tuple_sizes == [(24, 24), (24, 24), (24, 24), (24, 24)]


def test_frames_per_image_default_reduces_image_prompt_count() -> None:
    """Use fewer generated prompts when image-mode frames_per_image is greater than one."""
    str_response = """
    {
      "frames": [
        {"frame_index": 0, "frame_prompt": "frame zero"},
        {"frame_index": 1, "frame_prompt": "frame one"},
        {"frame_index": 2, "frame_prompt": "frame two"},
        {"frame_index": 3, "frame_prompt": "frame three"}
      ]
    }
    """
    obj_stub_client = StubStoryboardAiClient(str_response)
    obj_storyboard = VideoStoryboard(
        str_storyboard_prompt="Storyboard prompt",
        int_num_frames=10,
        obj_ai_api=obj_stub_client,
    )

    list_str_prompts = obj_storyboard.build_frame_prompts()

    assert obj_storyboard.int_frames_per_image == 3
    assert obj_storyboard.int_num_images == 4
    assert len(list_str_prompts) == 4
