"""Storyboard-to-image-sequence orchestration for AI-generated video frames."""

from __future__ import annotations

import argparse
import logging
import math
import re
import sys
from typing import Protocol

from ai_api_unified import AIStructuredPrompt
from .storyboard_decomposer_structured_prompt import (
    StoryboardDecomposerStructuredResult,
)
from .storyboard_decomposer_structured_prompt import (
    StoryboardDecomposerStructuredPrompt,
)


logger_app = logging.getLogger(__name__)


class AiStoryboardClient(Protocol):
    """Protocol for storyboard-capable AI clients used by VideoStoryboard."""

    def send_structured_prompt(
        self,
        obj_structured_prompt: AIStructuredPrompt,
        cls_response_model: type[AIStructuredPrompt],
    ) -> AIStructuredPrompt:
        """Return structured prompt output validated to the requested schema."""

    def create_image(self, str_prompt: str) -> bytes:
        """Return a generated image as raw bytes."""


class VideoStoryboard:
    """
    Build image prompts from a storyboard description and generate frame images.

    By default this class initializes ``AiApi`` and uses strict schema prompting
    to decompose one storyboard prompt into image-level prompts.
    """

    def __init__(
        self,
        str_storyboard_prompt: str,
        int_num_frames: int,
        int_frames_per_image: int = 3,
        obj_ai_api: AiStoryboardClient | None = None,
    ) -> None:
        """Initialize storyboard configuration, cache state, and AI client."""
        self._validate_constructor_inputs(
            str_storyboard_prompt,
            int_num_frames,
            int_frames_per_image,
        )

        self.str_storyboard_prompt = str_storyboard_prompt.strip()
        self.int_num_frames = int_num_frames
        self.int_frames_per_image = int_frames_per_image
        self.int_num_images = int(
            math.ceil(float(self.int_num_frames) / float(self.int_frames_per_image))
        )
        self.list_str_frame_prompts: list[str] = []
        self.list_bytes_frame_image_buffers: list[bytes] = []

        if obj_ai_api is None:
            from .ai_api import AiApi

            self.obj_ai_api: AiStoryboardClient = AiApi()
        else:
            self.obj_ai_api = obj_ai_api

    @staticmethod
    def _validate_constructor_inputs(
        str_storyboard_prompt: str,
        int_num_frames: int,
        int_frames_per_image: int,
    ) -> None:
        """Validate required constructor inputs."""
        if not str_storyboard_prompt or not str_storyboard_prompt.strip():
            logger_app.error("Storyboard prompt is required and cannot be empty.")
            raise ValueError("Storyboard prompt is required and cannot be empty.")
        if int_num_frames < 1:
            logger_app.error("Frame count must be >= 1. Received: %d", int_num_frames)
            raise ValueError("Frame count must be >= 1.")
        if int_frames_per_image < 1:
            logger_app.error(
                "frames_per_image must be >= 1. Received: %d",
                int_frames_per_image,
            )
            raise ValueError("frames_per_image must be >= 1.")

    @staticmethod
    def _normalize_whitespace(str_text: str) -> str:
        """Collapse repeated whitespace to single spaces."""
        str_normalized = re.sub(r"\s+", " ", str_text.strip())
        return str_normalized

    @staticmethod
    def _normalize_prompts_to_frame_count(
        list_str_raw_prompts: list[str], int_num_frames: int
    ) -> list[str]:
        """Resize prompt list deterministically to match exact frame count."""
        int_raw_count = len(list_str_raw_prompts)
        if int_raw_count == 0:
            raise ValueError(
                "Cannot normalize storyboard prompts because no prompts were provided."
            )

        if int_num_frames == int_raw_count:
            return list(list_str_raw_prompts)

        if int_num_frames == 1:
            str_single_prompt = list_str_raw_prompts[0]
            return [str_single_prompt]

        list_str_result: list[str] = []
        for int_frame_index in range(int_num_frames):
            float_position = (
                int_frame_index * (int_raw_count - 1) / float(int_num_frames - 1)
            )
            int_source_index = int(round(float_position))
            int_source_index = max(0, min(int_source_index, int_raw_count - 1))
            str_prompt = list_str_raw_prompts[int_source_index]
            list_str_result.append(str_prompt)
        return list_str_result

    @staticmethod
    def _coerce_structured_plan_to_prompts(
        obj_storyboard_plan: StoryboardDecomposerStructuredResult,
    ) -> list[str]:
        """Coerce structured storyboard plan into ordered frame prompts."""
        list_obj_frames = obj_storyboard_plan.frames or []
        list_tuple_index_prompt: list[tuple[int, str]] = []
        for obj_frame in list_obj_frames:
            str_prompt = VideoStoryboard._normalize_whitespace(obj_frame.frame_prompt)
            if not str_prompt:
                continue
            list_tuple_index_prompt.append((obj_frame.frame_index, str_prompt))

        list_tuple_index_prompt.sort(key=lambda tuple_item: tuple_item[0])
        list_str_prompts = [tuple_item[1] for tuple_item in list_tuple_index_prompt]
        return list_str_prompts

    def build_frame_prompts(self) -> list[str]:
        """
        Build the normalized list of image prompts for the storyboard.

        Third-party API usage reference:
        https://pypi.org/project/ai-api-unified/
        """
        if len(self.list_str_frame_prompts) == self.int_num_images:
            list_str_cached_prompts = list(self.list_str_frame_prompts)
            return list_str_cached_prompts

        obj_structured_prompt = StoryboardDecomposerStructuredPrompt(
            message_input=self.str_storyboard_prompt,
            int_num_frames=self.int_num_images,
        )
        obj_response = self.obj_ai_api.send_structured_prompt(
            obj_structured_prompt=obj_structured_prompt,
            cls_response_model=StoryboardDecomposerStructuredResult,
        )
        if not isinstance(obj_response, StoryboardDecomposerStructuredResult):
            logger_app.error("Storyboard decomposition response type was not expected.")
            raise TypeError("Storyboard decomposition response type was not expected.")

        list_str_raw_prompts = self._coerce_structured_plan_to_prompts(obj_response)
        if not list_str_raw_prompts:
            logger_app.error(
                "Structured storyboard plan did not include usable frame prompts."
            )
            raise ValueError(
                "Structured storyboard plan did not include usable frame prompts."
            )

        list_str_frame_prompts = self._normalize_prompts_to_frame_count(
            list_str_raw_prompts,
            self.int_num_images,
        )

        self.list_str_frame_prompts = list_str_frame_prompts
        list_str_result = list(self.list_str_frame_prompts)
        logger_app.debug("Built frame prompts: %s", list_str_result)
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
            bytes_frame_image = self.obj_ai_api.create_image(str_frame_prompt)
            self.list_bytes_frame_image_buffers.append(bytes_frame_image)

        list_bytes_result = list(self.list_bytes_frame_image_buffers)
        return list_bytes_result

    def clear_storyboard_cache(self) -> None:
        """Clear generated prompt and image buffer caches for this storyboard instance."""
        self.list_str_frame_prompts = []
        self.list_bytes_frame_image_buffers = []


def main() -> None:
    """Debug entrypoint for storyboard prompt decomposition."""
    obj_parser = argparse.ArgumentParser(
        description="Video storyboard prompt decomposition debugger"
    )
    obj_parser.add_argument(
        "--storyboard_prompt",
        type=str,
        default=(
            "An extreme closeup of a fair-skinned woman's right eye. "
            "She looks straight ahead, then left, blinks, and looks ahead again."
        ),
        help="Storyboard prompt to decompose into frame-level image prompts.",
    )
    obj_parser.add_argument(
        "--num_frames",
        type=int,
        default=8,
        help="Total number of output video frames to render.",
    )
    obj_parser.add_argument(
        "--frames_per_image",
        type=int,
        default=3,
        help="Number of video frames to hold each generated image.",
    )
    obj_args = obj_parser.parse_args()

    try:
        obj_storyboard = VideoStoryboard(
            str_storyboard_prompt=obj_args.storyboard_prompt,
            int_num_frames=obj_args.num_frames,
            int_frames_per_image=obj_args.frames_per_image,
        )
        list_bytes_frame_images: list[bytes] = obj_storyboard.generate_storyboard()
    except Exception as exc_error:
        logger_app.error(
            "Failed to generate storyboard prompt list. Context: %s", exc_error
        )
        return


if __name__ == "__main__":
    main()
