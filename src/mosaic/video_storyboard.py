"""Storyboard orchestration for image-first and video-first source generation."""

from __future__ import annotations

import argparse
import importlib
import logging
import math
import re
from pathlib import Path
from typing import Any, Literal, Protocol

from ai_api_unified import AIBaseVideoProperties
from ai_api_unified import AIStructuredPrompt
from ai_api_unified import AIVideoGenerationResult
from ai_api_unified.videos.frame_helpers import FRAME_EXTRA_INSTALL_MESSAGE

from .storyboard_decomposer_structured_prompt import (
    StoryboardDecomposerStructuredPrompt,
)
from .storyboard_decomposer_structured_prompt import (
    StoryboardDecomposerStructuredResult,
)


logger_app = logging.getLogger(__name__)

TypeStoryboardMode = Literal["image", "video"]
DEFAULT_STORYBOARD_IMAGE_MODE: str = "image"
DEFAULT_STORYBOARD_VIDEO_MODE: str = "video"
DEFAULT_STORYBOARD_FRAMES_PER_IMAGE: int = 3


class AiStoryboardClient(Protocol):
    """Protocol for storyboard-capable AI clients used by `VideoStoryboard`."""

    def send_structured_prompt(
        self,
        obj_structured_prompt: AIStructuredPrompt,
        cls_response_model: type[AIStructuredPrompt],
    ) -> AIStructuredPrompt:
        """Return structured prompt output validated to the requested schema."""

    def create_image(self, str_prompt: str) -> bytes:
        """Return a generated image as raw bytes."""

    def create_video(
        self,
        str_prompt: str,
        obj_video_properties: AIBaseVideoProperties,
        str_video_engine: str | None = None,
        str_video_model_name: str | None = None,
    ) -> AIVideoGenerationResult:
        """Return a normalized completed video-generation result."""

    def extract_video_frames(
        self,
        bytes_video: bytes,
        list_float_time_offsets: list[float] | None = None,
        list_int_frame_indices: list[int] | None = None,
    ) -> list[bytes]:
        """Return extracted image buffers from one local video buffer."""


class VideoStoryboard:
    """Build ordered source-image buffers for storyboard-driven mosaic rendering.

    Purpose:
    - Support both the existing image-first storyboard path and the new
      video-first storyboard path.
    - Preserve a single output contract for the mosaic renderer:
      `list[bytes]` ordered in temporal sequence.

    Constructor Inputs:
    - `str_storyboard_prompt`: Non-empty storyboard prompt describing the motion.
    - `int_num_frames`: Final mosaic video frame count requested by the caller.
    - `int_frames_per_image`: Number of final output frames that reuse one source
      image in image mode. This value is ignored in video mode.
    - `obj_ai_api`: Optional AI client implementation matching
      `AiStoryboardClient`. Defaults to `AiApi`.

    Output/Behavior:
    - `build_frame_prompts()` returns normalized image prompts for the image path.
    - `generate_storyboard()` returns ordered source frame buffers for the chosen
      storyboard path.
    - In video mode, the class extracts exact integer-indexed frames from a
      materialized local MP4 artifact.
    """

    def __init__(
        self,
        str_storyboard_prompt: str,
        int_num_frames: int,
        int_frames_per_image: int = DEFAULT_STORYBOARD_FRAMES_PER_IMAGE,
        obj_ai_api: AiStoryboardClient | None = None,
    ) -> None:
        """Initialize storyboard configuration, caches, and the AI client.

        Purpose:
        - Validate constructor inputs.
        - Store the requested final output frame count and the image-path frame
          reuse settings.
        - Lazily bind the default `AiApi` wrapper when the caller does not supply
          a test stub or alternative implementation.

        Inputs:
        - `str_storyboard_prompt`: Non-empty storyboard prompt string.
        - `int_num_frames`: Positive integer final output frame count.
        - `int_frames_per_image`: Positive integer describing image-path source
          frame reuse.
        - `obj_ai_api`: Optional storyboard-capable AI client.

        Output:
        - Returns `None` after instance state has been initialized.
        """
        self._validate_constructor_inputs(
            str_storyboard_prompt,
            int_num_frames,
            int_frames_per_image,
        )

        self.str_storyboard_prompt: str = str_storyboard_prompt.strip()
        self.int_num_frames: int = int_num_frames
        self.int_frames_per_image: int = int_frames_per_image
        self.int_num_images: int = int(
            math.ceil(float(self.int_num_frames) / float(self.int_frames_per_image))
        )
        self.list_str_frame_prompts: list[str] = []
        self.list_bytes_frame_image_buffers: list[bytes] = []

        if obj_ai_api is None:
            from .ai_api import AiApi

            self.obj_ai_api: AiStoryboardClient = AiApi()
        else:
            self.obj_ai_api = obj_ai_api
        # Normal return after initializing the storyboard state.
        return None

    @staticmethod
    def validate_video_runtime_dependencies() -> None:
        """Validate local runtime prerequisites for storyboard video extraction.

        Purpose:
        - Fail fast before dispatching a paid video-generation request.
        - Ensure the environment can materialize and decode a provider MP4 into
          clean source image frames.

        Inputs:
        - None. This method inspects the optional Python video-decoding modules
          used by the local extraction path.

        Output:
        - Returns `None` when the local environment can decode storyboard videos.
        - Raises `RuntimeError` with actionable dependency guidance when a
          required local tool or Python package is missing.
        """
        try:
            # Validate the exact optional decoder modules used by the frame-extraction helper before runtime video dispatch.
            importlib.import_module("imageio")
            imageio_ffmpeg_module: Any = importlib.import_module("imageio_ffmpeg")
            importlib.import_module("PIL.Image")
        except ImportError as exc_error:
            raise RuntimeError(FRAME_EXTRA_INSTALL_MESSAGE) from exc_error

        try:
            str_ffmpeg_executable_path: str = imageio_ffmpeg_module.get_ffmpeg_exe()
        except Exception as exc_error:
            raise RuntimeError(
                "Storyboard video mode requires a usable `imageio-ffmpeg` backend. "
                "Run `./setup.sh` to bootstrap local prerequisites."
            ) from exc_error

        path_ffmpeg_executable: Path = Path(str_ffmpeg_executable_path)
        if not path_ffmpeg_executable.exists():
            raise RuntimeError(
                "Storyboard video mode requires a usable `imageio-ffmpeg` backend. "
                "Run `./setup.sh` to bootstrap local prerequisites."
            )
        # Normal return after validating video extraction prerequisites.
        return None

    @staticmethod
    def _validate_constructor_inputs(
        str_storyboard_prompt: str,
        int_num_frames: int,
        int_frames_per_image: int,
    ) -> None:
        """Validate required constructor inputs.

        Purpose:
        - Keep instance configuration explicit and deterministic before any
          provider calls are attempted.

        Inputs:
        - `str_storyboard_prompt`: Candidate storyboard prompt.
        - `int_num_frames`: Candidate final output frame count.
        - `int_frames_per_image`: Candidate image-path frame reuse count.

        Output:
        - Returns `None` when all values are valid.
        - Raises `ValueError` when any input violates the constructor contract.
        """
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
        # Normal return after validating constructor inputs.
        return None

    @staticmethod
    def _normalize_whitespace(str_text: str) -> str:
        """Collapse repeated whitespace into single spaces.

        Purpose:
        - Normalize storyboard prompts before caching or comparison.

        Inputs:
        - `str_text`: Raw prompt text that may contain repeated whitespace.

        Output:
        - Returns one normalized string with repeated whitespace collapsed.
        """
        str_normalized: str = re.sub(r"\s+", " ", str_text.strip())
        # Normal return with whitespace-normalized prompt text.
        return str_normalized

    @staticmethod
    def _normalize_prompts_to_frame_count(
        list_str_raw_prompts: list[str], int_num_frames: int
    ) -> list[str]:
        """Resize a raw prompt list deterministically to match one target count.

        Purpose:
        - Guarantee that the image storyboard path emits exactly one prompt per
          generated source image.

        Inputs:
        - `list_str_raw_prompts`: Ordered prompt list returned by the structured
          storyboard decomposition step.
        - `int_num_frames`: Exact target prompt count required by the image path.

        Output:
        - Returns a `list[str]` whose length matches `int_num_frames`.
        - Raises `ValueError` when no prompts were supplied.
        """
        int_raw_count: int = len(list_str_raw_prompts)
        if int_raw_count == 0:
            raise ValueError(
                "Cannot normalize storyboard prompts because no prompts were provided."
            )

        if int_num_frames == int_raw_count:
            list_str_result: list[str] = list(list_str_raw_prompts)
            # Normal return because the raw prompt count already matches the requested count.
            return list_str_result

        if int_num_frames == 1:
            str_single_prompt: str = list_str_raw_prompts[0]
            # Normal return with the first prompt as the only required frame prompt.
            return [str_single_prompt]

        list_str_result = []
        # Map each requested prompt slot onto the nearest prompt in the raw temporal plan.
        for int_frame_index in range(int_num_frames):
            float_position: float = (
                int_frame_index * (int_raw_count - 1) / float(int_num_frames - 1)
            )
            int_source_index: int = int(round(float_position))
            int_source_index = max(0, min(int_source_index, int_raw_count - 1))
            str_prompt: str = list_str_raw_prompts[int_source_index]
            list_str_result.append(str_prompt)
        # Normal return with a deterministically resized prompt list.
        return list_str_result

    @staticmethod
    def _coerce_structured_plan_to_prompts(
        obj_storyboard_plan: StoryboardDecomposerStructuredResult,
    ) -> list[str]:
        """Convert a structured storyboard plan into ordered prompt strings.

        Purpose:
        - Strip empty prompts.
        - Sort prompts by `frame_index`.
        - Normalize whitespace before caching.

        Inputs:
        - `obj_storyboard_plan`: Structured storyboard decomposition result.

        Output:
        - Returns an ordered `list[str]` ready for prompt-count normalization.
        """
        list_obj_frames = obj_storyboard_plan.frames or []
        list_tuple_index_prompt: list[tuple[int, str]] = []
        # Collect usable prompt strings together with their temporal frame indexes.
        for obj_frame in list_obj_frames:
            str_prompt: str = VideoStoryboard._normalize_whitespace(
                obj_frame.frame_prompt
            )
            if not str_prompt:
                continue
            list_tuple_index_prompt.append((obj_frame.frame_index, str_prompt))

        list_tuple_index_prompt.sort(key=lambda tuple_item: tuple_item[0])
        list_str_prompts: list[str] = [
            tuple_item[1] for tuple_item in list_tuple_index_prompt
        ]
        # Normal return with temporal prompts sorted by frame index.
        return list_str_prompts

    @staticmethod
    def _load_imageio_module() -> Any:
        """Load the optional `imageio` dependency used for local video inspection.

        Purpose:
        - Delay importing optional frame-decoding dependencies until the video
          storyboard path is actually used.

        Inputs:
        - None.

        Output:
        - Returns the imported `imageio` module object.
        - Raises `RuntimeError` with install guidance when the dependency is
          unavailable.
        """
        try:
            imageio_module: Any = importlib.import_module("imageio")
        except ImportError as exc_error:
            raise RuntimeError(FRAME_EXTRA_INSTALL_MESSAGE) from exc_error
        # Normal return with the lazily imported `imageio` module.
        return imageio_module

    @staticmethod
    def _read_available_video_frame_count_from_path(path_video_file: Path) -> int:
        """Inspect one local MP4 artifact and resolve the exact frame count.

        Purpose:
        - Compute the available frame count before selecting extraction indices.
        - Avoid relying on approximate storyboard duration assumptions.

        Inputs:
        - `path_video_file`: Local filesystem path to one materialized MP4
          artifact returned by `ai-api-unified`.

        Output:
        - Returns one positive integer frame count.
        - Raises `ValueError` when the file does not exist or frame metadata
          cannot be resolved to a positive count.
        """
        if not path_video_file.exists():
            raise ValueError(
                f"Generated video artifact does not exist: {path_video_file}"
            )

        imageio_module: Any = VideoStoryboard._load_imageio_module()
        reader: Any = imageio_module.get_reader(str(path_video_file), format="ffmpeg")
        try:
            try:
                int_frame_count: int = int(reader.count_frames())
            except Exception:
                dict_metadata: dict[str, Any] = reader.get_meta_data()
                object_nframes: Any = dict_metadata.get("nframes")
                if isinstance(object_nframes, (int, float)) and math.isfinite(
                    float(object_nframes)
                ):
                    int_frame_count = int(object_nframes)
                else:
                    object_fps: Any = dict_metadata.get("fps")
                    object_duration: Any = dict_metadata.get("duration")
                    if object_fps in (None, 0) or object_duration in (None, 0):
                        raise ValueError(
                            "Unable to resolve frame count from generated video metadata."
                        )
                    int_frame_count = int(
                        round(float(object_fps) * float(object_duration))
                    )

            if int_frame_count < 1:
                raise ValueError(
                    "Generated video metadata did not resolve to a positive frame count."
                )
        finally:
            reader.close()
        # Normal return with the resolved positive video frame count.
        return int_frame_count

    @staticmethod
    def _validate_storyboard_mode(
        str_storyboard_mode: str,
    ) -> TypeStoryboardMode:
        """Validate and normalize the requested storyboard generation mode.

        Purpose:
        - Keep the public `generate_storyboard()` mode contract explicit.

        Inputs:
        - `str_storyboard_mode`: Raw mode string supplied by the caller.

        Output:
        - Returns `"image"` or `"video"` as a normalized mode literal.
        - Raises `ValueError` for unsupported values.
        """
        str_storyboard_mode_normalized: str = str_storyboard_mode.strip().lower()
        if str_storyboard_mode_normalized not in {
            DEFAULT_STORYBOARD_IMAGE_MODE,
            DEFAULT_STORYBOARD_VIDEO_MODE,
        }:
            raise ValueError("storyboard_mode must be either 'image' or 'video'.")
        str_result: TypeStoryboardMode = str_storyboard_mode_normalized  # type: ignore[assignment]
        # Normal return with the validated storyboard mode literal.
        return str_result

    def _build_video_frame_indices(self, int_available_frame_count: int) -> list[int]:
        """Build exact extraction indices for the final storyboard video path.

        Purpose:
        - Produce one source frame index per final requested mosaic frame.
        - Preserve the full source timeline by including the first and last
          available frames.

        Inputs:
        - `int_available_frame_count`: Positive integer count of frames available
          in the generated source video.

        Output:
        - Returns a monotonic `list[int]` of exact frame indices.
        - Raises `ValueError` when the request exceeds available source frames.
        """
        if int_available_frame_count < 1:
            raise ValueError("Available video frame count must be >= 1.")
        if self.int_num_frames > int_available_frame_count:
            raise ValueError(
                "Requested storyboard_num_frames exceeds available generated video frames. "
                f"requested={self.int_num_frames} available={int_available_frame_count}"
            )
        if self.int_num_frames == 1:
            # Normal return with the first available frame for a single-frame request.
            return [0]
        if self.int_num_frames == int_available_frame_count:
            list_int_result: list[int] = list(range(int_available_frame_count))
            # Normal return because every source frame should be extracted.
            return list_int_result

        list_int_frame_indices: list[int] = []
        # Map each final output frame slot onto an exact source video frame index spanning the full source timeline.
        for int_output_frame_index in range(self.int_num_frames):
            float_position: float = (
                int_output_frame_index
                * (int_available_frame_count - 1)
                / float(self.int_num_frames - 1)
            )
            int_source_frame_index: int = int(round(float_position))
            int_source_frame_index = max(
                0,
                min(int_source_frame_index, int_available_frame_count - 1),
            )
            list_int_frame_indices.append(int_source_frame_index)
        # Normal return with exact integer source frame indices for extraction.
        return list_int_frame_indices

    def build_frame_prompts(self) -> list[str]:
        """Build the normalized image-prompt list for storyboard image mode.

        Purpose:
        - Decompose the storyboard prompt into a structured temporal prompt plan.
        - Resize the prompt list to match the exact generated-image count
          required by the image storyboard path.

        Inputs:
        - None. This method uses the instance constructor state.

        Output:
        - Returns a normalized ordered `list[str]` whose length equals
          `self.int_num_images`.
        - Raises `TypeError` or `ValueError` when the structured plan is invalid.
        """
        if len(self.list_str_frame_prompts) == self.int_num_images:
            list_str_cached_prompts: list[str] = list(self.list_str_frame_prompts)
            # Normal return with the cached prompt list for the image storyboard path.
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

        list_str_raw_prompts: list[str] = self._coerce_structured_plan_to_prompts(
            obj_response
        )
        if not list_str_raw_prompts:
            logger_app.error(
                "Structured storyboard plan did not include usable frame prompts."
            )
            raise ValueError(
                "Structured storyboard plan did not include usable frame prompts."
            )

        self.list_str_frame_prompts = self._normalize_prompts_to_frame_count(
            list_str_raw_prompts,
            self.int_num_images,
        )
        list_str_result: list[str] = list(self.list_str_frame_prompts)
        logger_app.debug("Built frame prompts: %s", list_str_result)
        # Normal return with the normalized image storyboard prompt list.
        return list_str_result

    def _generate_storyboard_from_images(self) -> list[bytes]:
        """Generate ordered source image buffers through the image storyboard path.

        Purpose:
        - Preserve the existing storyboard decomposition behavior for single-frame
          storyboards and the explicit image fallback path.

        Inputs:
        - None. This method uses the stored storyboard prompt and image-path
          prompt-count settings.

        Output:
        - Returns a `list[bytes]` of generated source images in temporal order.
        """
        self.list_bytes_frame_image_buffers = []
        list_str_frame_prompts: list[str] = self.build_frame_prompts()
        int_total_frames: int = len(list_str_frame_prompts)

        # Generate one source image buffer for each temporal image prompt in the storyboard plan.
        for int_frame_index, str_frame_prompt in enumerate(list_str_frame_prompts):
            logger_app.info(
                "Generating storyboard image frame %d/%d",
                int_frame_index + 1,
                int_total_frames,
            )
            bytes_frame_image: bytes = self.obj_ai_api.create_image(str_frame_prompt)
            self.list_bytes_frame_image_buffers.append(bytes_frame_image)

        list_bytes_result: list[bytes] = list(self.list_bytes_frame_image_buffers)
        # Normal return with temporal source image buffers from the image path.
        return list_bytes_result

    @staticmethod
    def _read_primary_video_artifact(
        obj_video_result: AIVideoGenerationResult,
    ) -> tuple[bytes, Path]:
        """Read the primary materialized MP4 artifact from one video-generation result.

        Purpose:
        - Validate that the provider returned at least one materialized local
          video artifact.
        - Return both the local artifact path and the raw bytes used by the
          extraction helper.

        Inputs:
        - `obj_video_result`: Completed normalized video-generation result.

        Output:
        - Returns a `(bytes_video_buffer, path_video_file)` tuple.
        - Raises `ValueError` when no materialized local artifact is available.
        """
        if not obj_video_result.artifacts:
            raise ValueError(
                "No video artifacts returned from storyboard video generation."
            )

        obj_primary_artifact = obj_video_result.artifacts[0]
        if obj_primary_artifact.file_path is None:
            raise ValueError(
                "Storyboard video extraction requires a materialized local video artifact."
            )

        bytes_video_buffer: bytes = obj_primary_artifact.read_bytes()
        path_video_file: Path = obj_primary_artifact.file_path
        # Normal return with the primary local video artifact bytes and path.
        return bytes_video_buffer, path_video_file

    def _generate_storyboard_from_video(
        self,
        obj_video_properties: AIBaseVideoProperties,
        str_video_engine: str | None = None,
        str_video_model_name: str | None = None,
    ) -> list[bytes]:
        """Generate ordered source frames through the video storyboard path.

        Purpose:
        - Generate one coherent AI source video from the storyboard prompt.
        - Inspect the materialized local MP4 to determine the exact available
          frame count.
        - Extract exact integer-indexed source frames for the mosaic renderer.

        Inputs:
        - `obj_video_properties`: Shared video-generation properties.
        - `str_video_engine`: Optional provider engine override.
        - `str_video_model_name`: Optional provider model override.

        Output:
        - Returns a `list[bytes]` containing exactly `self.int_num_frames`
          extracted PNG source frames in temporal order.
        """
        logger_app.info(
            "Generating storyboard source video for %d final mosaic frames",
            self.int_num_frames,
        )
        obj_video_result: AIVideoGenerationResult = self.obj_ai_api.create_video(
            self.str_storyboard_prompt,
            obj_video_properties=obj_video_properties,
            str_video_engine=str_video_engine,
            str_video_model_name=str_video_model_name,
        )
        bytes_video_buffer: bytes
        path_video_file: Path
        bytes_video_buffer, path_video_file = self._read_primary_video_artifact(
            obj_video_result
        )
        int_available_frame_count: int = (
            self._read_available_video_frame_count_from_path(path_video_file)
        )
        list_int_frame_indices: list[int] = self._build_video_frame_indices(
            int_available_frame_count
        )
        logger_app.info(
            "Extracting %d storyboard source frames from %d available source video frames",
            len(list_int_frame_indices),
            int_available_frame_count,
        )
        self.list_bytes_frame_image_buffers = self.obj_ai_api.extract_video_frames(
            bytes_video_buffer,
            list_int_frame_indices=list_int_frame_indices,
        )
        list_bytes_result: list[bytes] = list(self.list_bytes_frame_image_buffers)
        # Normal return with exact indexed source frames extracted from the local MP4 artifact.
        return list_bytes_result

    def generate_storyboard(
        self,
        str_storyboard_mode: str = DEFAULT_STORYBOARD_VIDEO_MODE,
        obj_video_properties: AIBaseVideoProperties | None = None,
        str_video_engine: str | None = None,
        str_video_model_name: str | None = None,
    ) -> list[bytes]:
        """Generate ordered storyboard source buffers for the requested mode.

        Purpose:
        - Route storyboard generation through the image path or video path.
        - Preserve the image path as the fallback for single-frame requests.

        Inputs:
        - `str_storyboard_mode`: Requested path selection, either `"image"` or
          `"video"`.
        - `obj_video_properties`: Optional video-generation properties. These are
          used only when the effective path is `"video"`.
        - `str_video_engine`: Optional provider engine override for the video
          path.
        - `str_video_model_name`: Optional provider model override for the video
          path.

        Output:
        - Returns a `list[bytes]` containing ordered source frame buffers.
        - In image mode, the list length equals `self.int_num_images`.
        - In video mode, the list length equals `self.int_num_frames`.
        """
        str_effective_storyboard_mode: TypeStoryboardMode = (
            self._validate_storyboard_mode(str_storyboard_mode)
        )
        if (
            str_effective_storyboard_mode == DEFAULT_STORYBOARD_VIDEO_MODE
            and self.int_num_frames == 1
        ):
            logger_app.info(
                "Storyboard video mode requested for a single frame. Falling back to image mode."
            )
            str_effective_storyboard_mode = DEFAULT_STORYBOARD_IMAGE_MODE

        if str_effective_storyboard_mode == DEFAULT_STORYBOARD_IMAGE_MODE:
            list_bytes_result = self._generate_storyboard_from_images()
            # Normal return with source image buffers from the image storyboard path.
            return list_bytes_result

        obj_effective_video_properties: AIBaseVideoProperties = (
            obj_video_properties
            if obj_video_properties is not None
            else AIBaseVideoProperties()
        )
        list_bytes_result = self._generate_storyboard_from_video(
            obj_video_properties=obj_effective_video_properties,
            str_video_engine=str_video_engine,
            str_video_model_name=str_video_model_name,
        )
        # Normal return with exact indexed source frame buffers from the video storyboard path.
        return list_bytes_result

    def clear_storyboard_cache(self) -> None:
        """Clear cached prompt and source frame buffers for this storyboard.

        Purpose:
        - Reset the instance so repeated calls do not reuse prior prompt or image
          buffers after configuration changes in tests or interactive use.

        Inputs:
        - None.

        Output:
        - Returns `None` after clearing both the prompt and image-buffer caches.
        """
        self.list_str_frame_prompts = []
        self.list_bytes_frame_image_buffers = []
        # Normal return after clearing cached storyboard state.
        return None


def main() -> None:
    """Debug entrypoint for storyboard prompt routing and extraction.

    Purpose:
    - Provide a lightweight manual runner for local debugging of the storyboard
      orchestration layer.

    Inputs:
    - Uses CLI arguments parsed from `sys.argv`.

    Output:
    - Returns `None` after printing logs and attempting one storyboard run.
    """
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
        help="Storyboard prompt to route into image or video generation.",
    )
    obj_parser.add_argument(
        "--num_frames",
        type=int,
        default=8,
        help="Total number of final output video frames to render.",
    )
    obj_parser.add_argument(
        "--frames_per_image",
        type=int,
        default=DEFAULT_STORYBOARD_FRAMES_PER_IMAGE,
        help="Number of output frames to hold each generated image in image mode.",
    )
    obj_parser.add_argument(
        "--storyboard_mode",
        type=str,
        choices=[DEFAULT_STORYBOARD_IMAGE_MODE, DEFAULT_STORYBOARD_VIDEO_MODE],
        default=DEFAULT_STORYBOARD_VIDEO_MODE,
        help="Storyboard source generation path to test.",
    )
    obj_args = obj_parser.parse_args()

    try:
        obj_storyboard = VideoStoryboard(
            str_storyboard_prompt=obj_args.storyboard_prompt,
            int_num_frames=obj_args.num_frames,
            int_frames_per_image=obj_args.frames_per_image,
        )
        obj_storyboard.generate_storyboard(
            str_storyboard_mode=obj_args.storyboard_mode,
        )
    except Exception as exc_error:
        logger_app.error("Failed to generate storyboard output. Context: %s", exc_error)
        # Early return after logging the storyboard debug failure.
        return
    # Normal return after the debug storyboard run completes.
    return


if __name__ == "__main__":
    main()
