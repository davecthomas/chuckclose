"""Storyboard-to-image-sequence orchestration for AI-generated video frames."""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Callable
from typing import Any
from typing import Protocol
from typing import TypeVar


logger_app = logging.getLogger(__name__)

TypeRetryValue = TypeVar("TypeRetryValue")


class AiImageStoryboardClient(Protocol):
    """Protocol for storyboard-capable AI clients used by VideoStoryboard."""

    def send_prompt(self, str_prompt: str) -> str:
        """Return a text response for a prompt."""

    def create_image(self, str_prompt: str) -> bytes:
        """Return a generated image as raw bytes."""


class VideoStoryboard:
    """
    Build frame prompts from a storyboard description and generate frame images.

    The class is intentionally API-provider agnostic at runtime via protocol typing.
    By default it initializes ``AiApi`` from this package when no client is supplied.
    """

    def __init__(
        self,
        str_storyboard_prompt: str,
        int_num_frames: int,
        obj_ai_api: AiImageStoryboardClient | None = None,
    ) -> None:
        """Initialize storyboard configuration, cache state, and AI client."""
        self._validate_constructor_inputs(str_storyboard_prompt, int_num_frames)

        self.str_storyboard_prompt = str_storyboard_prompt.strip()
        self.int_num_frames = int_num_frames
        self.int_max_retries = 3
        self.float_retry_base_seconds = 0.5
        self.float_retry_max_seconds = 4.0
        self.list_str_frame_prompts: list[str] = []
        self.list_bytes_frame_image_buffers: list[bytes] = []

        if obj_ai_api is None:
            from .ai_api import AiApi

            self.obj_ai_api: AiImageStoryboardClient = AiApi()
        else:
            self.obj_ai_api = obj_ai_api

    @staticmethod
    def _validate_constructor_inputs(str_storyboard_prompt: str, int_num_frames: int) -> None:
        """Validate required constructor inputs."""
        if not str_storyboard_prompt or not str_storyboard_prompt.strip():
            logger_app.error("Storyboard prompt is required and cannot be empty.")
            raise ValueError("Storyboard prompt is required and cannot be empty.")
        if int_num_frames < 1:
            logger_app.error("Frame count must be >= 1. Received: %d", int_num_frames)
            raise ValueError("Frame count must be >= 1.")

    @staticmethod
    def _normalize_whitespace(str_text: str) -> str:
        """Collapse repeated whitespace to single spaces."""
        str_normalized = re.sub(r"\s+", " ", str_text.strip())
        return str_normalized

    @staticmethod
    def _strip_markdown_fence(str_text: str) -> str:
        """Remove leading and trailing markdown code fences if present."""
        str_result = str_text.strip()
        if str_result.startswith("```"):
            str_result = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", str_result)
            str_result = re.sub(r"\s*```$", "", str_result)
        return str_result.strip()

    @staticmethod
    def _extract_json_container_text(str_response: str) -> str:
        """Extract a JSON object/array substring from a model response."""
        str_clean = VideoStoryboard._strip_markdown_fence(str_response)
        if str_clean.startswith("{") or str_clean.startswith("["):
            return str_clean

        int_object_start = str_clean.find("{")
        int_object_end = str_clean.rfind("}")
        int_array_start = str_clean.find("[")
        int_array_end = str_clean.rfind("]")

        bool_has_object = int_object_start != -1 and int_object_end > int_object_start
        bool_has_array = int_array_start != -1 and int_array_end > int_array_start

        if bool_has_object and bool_has_array:
            int_object_len = int_object_end - int_object_start
            int_array_len = int_array_end - int_array_start
            if int_array_len > int_object_len:
                str_result = str_clean[int_array_start : int_array_end + 1]
                return str_result
            str_result = str_clean[int_object_start : int_object_end + 1]
            return str_result

        if bool_has_object:
            str_result = str_clean[int_object_start : int_object_end + 1]
            return str_result

        if bool_has_array:
            str_result = str_clean[int_array_start : int_array_end + 1]
            return str_result

        return ""

    @staticmethod
    def _coerce_model_payload_to_prompts(obj_payload: Any) -> list[str]:
        """Coerce structured model payload into an ordered prompt list."""
        list_dict_frame_items: list[dict[str, Any]] = []

        if isinstance(obj_payload, dict):
            obj_frames = obj_payload.get("frames")
            if isinstance(obj_frames, list):
                for obj_frame in obj_frames:
                    if isinstance(obj_frame, dict):
                        list_dict_frame_items.append(obj_frame)
        elif isinstance(obj_payload, list):
            for obj_frame in obj_payload:
                if isinstance(obj_frame, dict):
                    list_dict_frame_items.append(obj_frame)

        list_tuple_index_prompt: list[tuple[int, str]] = []
        for int_list_index, dict_frame_item in enumerate(list_dict_frame_items):
            obj_frame_prompt = dict_frame_item.get("frame_prompt")
            if not isinstance(obj_frame_prompt, str):
                continue

            str_frame_prompt = VideoStoryboard._normalize_whitespace(obj_frame_prompt)
            if not str_frame_prompt:
                continue

            obj_frame_index = dict_frame_item.get("frame_index")
            int_frame_index = int_list_index
            if isinstance(obj_frame_index, int):
                int_frame_index = obj_frame_index

            list_tuple_index_prompt.append((int_frame_index, str_frame_prompt))

        list_tuple_index_prompt.sort(key=lambda tuple_item: tuple_item[0])
        list_str_prompts = [tuple_item[1] for tuple_item in list_tuple_index_prompt]
        return list_str_prompts

    @staticmethod
    def _normalize_prompts_to_frame_count(list_str_raw_prompts: list[str], int_num_frames: int) -> list[str]:
        """Resize prompt list deterministically to match exact frame count."""
        if int_num_frames == 1:
            str_single_prompt = list_str_raw_prompts[0]
            return [str_single_prompt]

        int_raw_count = len(list_str_raw_prompts)
        list_str_result: list[str] = []
        for int_frame_index in range(int_num_frames):
            float_position = int_frame_index * (int_raw_count - 1) / float(int_num_frames - 1)
            int_source_index = int(round(float_position))
            int_source_index = max(0, min(int_source_index, int_raw_count - 1))
            str_prompt = list_str_raw_prompts[int_source_index]
            list_str_result.append(str_prompt)
        return list_str_result

    def _build_fallback_frame_prompts(self) -> list[str]:
        """Build deterministic fallback prompts when model planning fails."""
        list_str_fallback_prompts: list[str] = []
        int_denom = max(1, self.int_num_frames - 1)
        for int_frame_index in range(self.int_num_frames):
            float_progress = int_frame_index / float(int_denom)
            int_progress_pct = int(round(float_progress * 100))
            str_frame_prompt = (
                "Storyboard frame "
                f"{int_frame_index + 1}/{self.int_num_frames}. "
                "Keep subject identity, camera framing, lens context, lighting, and style consistent. "
                "Render the next temporal moment of this storyboard with smooth motion continuity: "
                f"{self.str_storyboard_prompt} "
                f"(timeline progress {int_progress_pct}%)."
            )
            list_str_fallback_prompts.append(str_frame_prompt)
        return list_str_fallback_prompts

    def _build_storybeat_planning_prompt(self) -> str:
        """Build constrained instruction prompt for frame prompt planning."""
        str_prompt = f"""
You are generating frame-by-frame image prompts for a storyboard.
Return ONLY valid JSON and no extra prose.

Required JSON format:
{{
  "frames": [
    {{
      "frame_index": 0,
      "frame_prompt": "..."
    }}
  ]
}}

Rules:
- Return exactly {self.int_num_frames} frames.
- frame_index values must start at 0 and be sequential.
- Each frame_prompt must be self-contained and fully specified.
- Keep identity, camera framing, lens context, lighting, and style consistent across frames.
- Vary only temporal motion and expression.

Storyboard:
{self.str_storyboard_prompt}
"""
        str_result = str_prompt.strip()
        return str_result

    def _is_retryable_exception(self, exc_error: Exception) -> bool:
        """Classify whether an exception is likely transient and retryable."""
        str_exception_name = exc_error.__class__.__name__.lower()
        str_error_text = str(exc_error).lower()
        list_str_retry_markers = [
            "timeout",
            "timed out",
            "rate limit",
            "too many requests",
            "temporarily unavailable",
            "service unavailable",
            "connection reset",
            "connection aborted",
            "network",
            "429",
            "500",
            "502",
            "503",
            "504",
        ]

        bool_name_retryable = "timeout" in str_exception_name or "connection" in str_exception_name
        bool_text_retryable = any(str_marker in str_error_text for str_marker in list_str_retry_markers)
        bool_retryable = bool_name_retryable or bool_text_retryable
        return bool_retryable

    def _execute_with_retry(
        self,
        fn_operation: Callable[[str], TypeRetryValue],
        str_payload: str,
        str_operation_name: str,
    ) -> TypeRetryValue:
        """Execute an AI operation with bounded exponential backoff for transient failures."""
        for int_attempt_index in range(self.int_max_retries):
            try:
                obj_result = fn_operation(str_payload)
                return obj_result
            except Exception as exc_error:
                bool_retryable = self._is_retryable_exception(exc_error)
                bool_has_attempt_remaining = int_attempt_index < (self.int_max_retries - 1)
                if not bool_retryable or not bool_has_attempt_remaining:
                    logger_app.error(
                        "Storyboard operation failed. operation=%s attempt=%d retryable=%s context=%s",
                        str_operation_name,
                        int_attempt_index + 1,
                        bool_retryable,
                        exc_error,
                    )
                    raise

                float_sleep_seconds = min(
                    self.float_retry_base_seconds * (2 ** int_attempt_index),
                    self.float_retry_max_seconds,
                )
                logger_app.warning(
                    "Retrying storyboard operation. operation=%s attempt=%d sleep_seconds=%.2f context=%s",
                    str_operation_name,
                    int_attempt_index + 1,
                    float_sleep_seconds,
                    exc_error,
                )
                time.sleep(float_sleep_seconds)

        raise RuntimeError("Unreachable retry state encountered.")

    def build_frame_prompts(self) -> list[str]:
        """
        Build the normalized list of frame prompts for the storyboard.

        Third-party API usage reference:
        https://pypi.org/project/ai-api-unified/
        """
        if len(self.list_str_frame_prompts) == self.int_num_frames:
            list_str_cached_prompts = list(self.list_str_frame_prompts)
            return list_str_cached_prompts

        str_planning_prompt = self._build_storybeat_planning_prompt()
        str_model_response = self._execute_with_retry(
            self.obj_ai_api.send_prompt,
            str_planning_prompt,
            "frame_prompt_planning",
        )

        list_str_frame_prompts: list[str]
        try:
            str_json_container = self._extract_json_container_text(str_model_response)
            if not str_json_container:
                raise ValueError("No JSON container found in storyboard planning response.")

            obj_payload = json.loads(str_json_container)
            list_str_raw_prompts = self._coerce_model_payload_to_prompts(obj_payload)
            if not list_str_raw_prompts:
                raise ValueError("No usable frame prompts in storyboard planning response.")

            list_str_frame_prompts = self._normalize_prompts_to_frame_count(
                list_str_raw_prompts,
                self.int_num_frames,
            )
        except Exception as exc_error:
            logger_app.warning(
                "Storyboard prompt decomposition failed, using deterministic fallback. context=%s",
                exc_error,
            )
            list_str_frame_prompts = self._build_fallback_frame_prompts()

        self.list_str_frame_prompts = list_str_frame_prompts
        list_str_result = list(self.list_str_frame_prompts)
        return list_str_result

    def generate_storyboard(self) -> list[bytes]:
        """
        Generate image bytes for each storyboard frame prompt in temporal order.

        Third-party API usage reference:
        https://pypi.org/project/ai-api-unified/
        """
        self.list_bytes_frame_image_buffers = []
        list_str_frame_prompts = self.build_frame_prompts()
        int_total_frames = len(list_str_frame_prompts)

        for int_frame_index, str_frame_prompt in enumerate(list_str_frame_prompts):
            logger_app.info(
                "Generating storyboard frame %d/%d",
                int_frame_index + 1,
                int_total_frames,
            )
            bytes_frame_image = self._execute_with_retry(
                self.obj_ai_api.create_image,
                str_frame_prompt,
                f"frame_image_generation_{int_frame_index}",
            )
            self.list_bytes_frame_image_buffers.append(bytes_frame_image)

        list_bytes_result = list(self.list_bytes_frame_image_buffers)
        return list_bytes_result

    def clear_storyboard_cache(self) -> None:
        """Clear generated prompt and image buffer caches for this storyboard instance."""
        self.list_str_frame_prompts = []
        self.list_bytes_frame_image_buffers = []
