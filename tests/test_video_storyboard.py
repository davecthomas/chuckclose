"""Tests for storyboard prompt and image buffer generation."""

from __future__ import annotations

from mosaic.video_storyboard import VideoStoryboard


class StubStoryboardAiClient:
    """Simple AI client stub for deterministic storyboard tests."""

    def __init__(self, str_send_prompt_response: str) -> None:
        self.str_send_prompt_response = str_send_prompt_response
        self.list_str_image_prompts: list[str] = []
        self.int_image_calls = 0

    def send_prompt(self, str_prompt: str) -> str:
        """Return configured storyboard planning response."""
        return self.str_send_prompt_response

    def create_image(self, str_prompt: str) -> bytes:
        """Return deterministic image bytes per prompt order."""
        self.int_image_calls += 1
        self.list_str_image_prompts.append(str_prompt)
        bytes_result = f"image-{self.int_image_calls}".encode("utf-8")
        return bytes_result


class FlakyStoryboardAiClient(StubStoryboardAiClient):
    """AI client stub that fails once with a retryable error before succeeding."""

    def __init__(self, str_send_prompt_response: str) -> None:
        super().__init__(str_send_prompt_response)
        self.bool_failed_once = False

    def create_image(self, str_prompt: str) -> bytes:
        """Raise a retryable error once, then return bytes."""
        if not self.bool_failed_once:
            self.bool_failed_once = True
            raise RuntimeError("429 Too Many Requests")
        return super().create_image(str_prompt)


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
        obj_ai_api=obj_stub_client,
    )

    list_str_prompts = obj_storyboard.build_frame_prompts()

    assert len(list_str_prompts) == 4
    assert list_str_prompts == ["frame zero", "frame zero", "frame one", "frame one"]


def test_build_frame_prompts_falls_back_on_invalid_payload() -> None:
    """Use deterministic fallback prompts when planning response is invalid."""
    obj_stub_client = StubStoryboardAiClient("not-json-response")
    obj_storyboard = VideoStoryboard(
        str_storyboard_prompt="A person blinks and looks ahead.",
        int_num_frames=3,
        obj_ai_api=obj_stub_client,
    )

    list_str_prompts = obj_storyboard.build_frame_prompts()

    assert len(list_str_prompts) == 3
    assert "Storyboard frame 1/3." in list_str_prompts[0]
    assert "timeline progress 100%" in list_str_prompts[-1]


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
        obj_ai_api=obj_stub_client,
    )

    list_bytes_images = obj_storyboard.generate_storyboard()

    assert list_bytes_images == [b"image-1", b"image-2"]
    assert obj_stub_client.list_str_image_prompts == ["first prompt", "second prompt"]


def test_generate_storyboard_retries_retryable_image_error() -> None:
    """Retry image generation on transient errors and then succeed."""
    str_response = """
    {
      "frames": [
        {"frame_index": 0, "frame_prompt": "only prompt"}
      ]
    }
    """
    obj_flaky_client = FlakyStoryboardAiClient(str_response)
    obj_storyboard = VideoStoryboard(
        str_storyboard_prompt="Storyboard prompt",
        int_num_frames=1,
        obj_ai_api=obj_flaky_client,
    )
    obj_storyboard.float_retry_base_seconds = 0.0
    obj_storyboard.float_retry_max_seconds = 0.0

    list_bytes_images = obj_storyboard.generate_storyboard()

    assert list_bytes_images == [b"image-1"]
    assert obj_flaky_client.int_image_calls == 1
