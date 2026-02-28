"""Tests for storyboard prompt and image buffer generation."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import TypeVar

from PIL import Image
import pytest

from ai_api_unified import AIStructuredPrompt
from mosaic import mosaic_generator
from mosaic.mosaic_generator import Mosaic
from mosaic.mosaic_image_inputs import MosaicImageInputs
from mosaic.video_storyboard import VideoStoryboard

TypeStructuredPrompt = TypeVar("TypeStructuredPrompt", bound=AIStructuredPrompt)


class StubStoryboardAiClient:
    """Simple AI client stub for deterministic storyboard tests."""

    def __init__(self, str_send_prompt_response: str) -> None:
        self.str_send_prompt_response = str_send_prompt_response
        self.list_str_image_prompts: list[str] = []
        self.int_image_calls = 0

    def send_structured_prompt(
        self,
        obj_structured_prompt: AIStructuredPrompt,
        cls_response_model: type[TypeStructuredPrompt],
    ) -> TypeStructuredPrompt:
        """Return configured storyboard planning response as structured payload."""
        dict_payload = json.loads(self.str_send_prompt_response)
        obj_response: TypeStructuredPrompt = cls_response_model.model_validate(dict_payload)
        return obj_response

    def create_image(self, str_prompt: str) -> bytes:
        """Return deterministic image bytes per prompt order."""
        self.int_image_calls += 1
        self.list_str_image_prompts.append(str_prompt)
        bytes_result = f"image-{self.int_image_calls}".encode("utf-8")
        return bytes_result


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
        tuple_color = (int_color_seed, (int_color_seed * 2) % 255, (int_color_seed * 3) % 255)
        image_frame = Image.new("RGB", (24, 24), tuple_color)
        obj_buffer = io.BytesIO()
        image_frame.save(obj_buffer, format="PNG")
        bytes_result = obj_buffer.getvalue()
        return bytes_result


class VideoFragmentRecorder:
    """Capture frame sizes forwarded to save_video_fragment during assembly."""

    def __init__(self) -> None:
        self.list_tuple_sizes: list[tuple[int, int]] = []

    def save_video_fragment(
        self,
        self_mosaic: Mosaic,
        image_buffer: Image.Image,
        str_path: str,
        int_num_frames: int = 1,
        int_fps: int = 30,
    ) -> None:
        self.list_tuple_sizes.append(image_buffer.size)


def test_build_frame_prompts_normalizes_to_requested_count() -> None:
    """Build frame prompts and normalize count deterministically."""
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


def test_generate_storyboard_calls_create_image_in_prompt_order() -> None:
    """Generate storyboard images in order and persist buffers in sequence."""
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
    assert obj_stub_client.list_str_image_prompts == ["first prompt", "second prompt"]


def test_generate_storyboard_raises_for_image_generation_error() -> None:
    """Raise image-generation exceptions from provider client."""
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
        obj_storyboard.generate_storyboard()


def test_storyboard_end_to_end_pipeline_with_video_buffer_assembly(
    path_input_image: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Validate storyboard prompt planning, image generation, and video-buffer assembly end to end."""
    list_dict_frames = [
        {"frame_index": 0, "frame_prompt": "Frame 1: subject looks straight ahead."},
        {"frame_index": 1, "frame_prompt": "Frame 2: subject looks left."},
        {"frame_index": 2, "frame_prompt": "Frame 3: subject blinks."},
        {"frame_index": 3, "frame_prompt": "Frame 4: subject looks straight ahead again."},
    ]
    dict_payload = {"frames": list_dict_frames}
    str_response = json.dumps(dict_payload)

    obj_stub_client = PngStoryboardAiClient(str_response)
    obj_storyboard = VideoStoryboard(
        str_storyboard_prompt="Storyboard prompt",
        int_num_frames=4,
        int_frames_per_image=1,
        obj_ai_api=obj_stub_client,
    )
    list_bytes_storyboard_images = obj_storyboard.generate_storyboard()

    assert len(list_bytes_storyboard_images) == 4
    assert len(obj_storyboard.list_str_frame_prompts) == 4
    assert all(len(bytes_item) > 0 for bytes_item in list_bytes_storyboard_images)

    obj_recorder = VideoFragmentRecorder()
    monkeypatch.setattr(mosaic_generator, "bool_has_video_support", False)
    monkeypatch.setattr(Mosaic, "save_video_fragment", obj_recorder.save_video_fragment)

    obj_mosaic_inputs = MosaicImageInputs(str_input_image_path=str(path_input_image))
    obj_mosaic = Mosaic(obj_mosaic_inputs)
    str_output_path = str(tmp_path / "storyboard_pipeline.mp4")
    obj_mosaic.generate_video_from_image_buffers(list_bytes_storyboard_images, str_output_path, int_fps=24)

    assert obj_recorder.list_tuple_sizes == [(24, 24), (24, 24), (24, 24), (24, 24)]


def test_frames_per_image_default_reduces_image_prompt_count() -> None:
    """Use fewer generated prompts when frames_per_image > 1."""
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
