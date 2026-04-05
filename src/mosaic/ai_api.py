import logging
import os

from ai_api_unified import AIFactory
from ai_api_unified import AIBaseCompletions
from ai_api_unified import AIBaseImageProperties
from ai_api_unified import AIBaseImages
from ai_api_unified import AIBaseVideoProperties
from ai_api_unified import AIBaseVideos
from ai_api_unified import AIVideoGenerationResult
from ai_api_unified import AIStructuredPrompt

from .exceptions import AiApiInitError
from .exceptions import AiApiRequestError


# Apply AGENTS.md preferred structured logging format
logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)
logger_app = logging.getLogger(__name__)


class AiApi:
    """
    Thin wrapper around ai-api-unified for local application calls.

    This wrapper delegates retry/backoff behavior to ai-api-unified provider clients.
    """

    def __init__(self) -> None:
        """
        Initialize provider clients and engine configuration.

        Environment variables:
        - `COMPLETIONS_ENGINE` defaults to `google-gemini`.
        - `GOOGLE_AUTH_METHOD=api_key`
        - `GOOGLE_GEMINI_API_KEY`
        """
        self._image_client: AIBaseImages | None = None
        self._video_client: AIBaseVideos | None = None
        self._tuple_video_client_config: tuple[str, str | None] | None = None
        self.str_completions_engine = os.getenv("COMPLETIONS_ENGINE", "google-gemini")
        self.str_video_engine = os.getenv("VIDEO_ENGINE", "google-gemini")
        try:
            self._model_client: AIBaseCompletions = AIFactory.get_ai_completions_client(
                completions_engine=self.str_completions_engine
            )
            logger_app.info(
                "Initialized completions client. model=%s engine=%s",
                self._model_client.model_name,
                self.str_completions_engine,
            )
        except Exception as exc_error:
            logger_app.error(
                "Failed to initialize ai-api-unified completions client: %s", exc_error
            )
            raise AiApiInitError(
                "Failed to initialize completions client."
            ) from exc_error

    def _ensure_image_client(self) -> AIBaseImages:
        """Lazily initialize and return the image provider client."""
        if self._image_client is None:
            try:
                # ai-api-unified API docs: https://pypi.org/project/ai-api-unified/
                self._image_client = AIFactory.get_ai_images_client()
                logger_app.info(
                    "Initialized image client. model=%s", self._image_client.model_name
                )
            except Exception as exc_error:
                logger_app.error(
                    "Failed to initialize ai-api-unified image client: %s", exc_error
                )
                raise AiApiInitError(
                    "Failed to initialize image client."
                ) from exc_error

        obj_image_client = self._image_client
        # Normal return with the lazily initialized image client.
        return obj_image_client

    def _ensure_video_client(
        self,
        str_video_engine: str | None = None,
        str_video_model_name: str | None = None,
    ) -> AIBaseVideos:
        """Lazily initialize and return the video provider client.

        Purpose:
        - Resolve a video-generation provider client through `ai-api-unified`.
        - Rebuild the cached client when the requested engine or model changes.

        Inputs:
        - `str_video_engine`: Optional provider engine override. Empty values fall
          back to `VIDEO_ENGINE` or `google-gemini`.
        - `str_video_model_name`: Optional provider model override. Empty values
          fall back to the provider default.

        Output:
        - Returns a configured `AIBaseVideos` implementation for the requested
          provider configuration.
        """
        str_resolved_video_engine: str = (
            str_video_engine.strip().lower()
            if str_video_engine is not None and str_video_engine.strip()
            else self.str_video_engine
        )
        str_resolved_video_model_name: str | None = (
            str_video_model_name.strip()
            if str_video_model_name is not None and str_video_model_name.strip()
            else None
        )
        tuple_requested_config: tuple[str, str | None] = (
            str_resolved_video_engine,
            str_resolved_video_model_name,
        )

        if (
            self._video_client is None
            or self._tuple_video_client_config != tuple_requested_config
        ):
            try:
                # Initialize or refresh the cached video client for the requested provider configuration.
                self._video_client = AIFactory.get_ai_video_client(
                    model_name=str_resolved_video_model_name,
                    video_engine=str_resolved_video_engine,
                )
                self._tuple_video_client_config = tuple_requested_config
                logger_app.info(
                    "Initialized video client. engine=%s model=%s",
                    str_resolved_video_engine,
                    self._video_client.model_name,
                )
            except Exception as exc_error:
                logger_app.error(
                    "Failed to initialize ai-api-unified video client: %s", exc_error
                )
                raise AiApiInitError(
                    "Failed to initialize video client."
                ) from exc_error

        obj_video_client: AIBaseVideos = self._video_client
        # Normal return with the lazily initialized video client.
        return obj_video_client

    def send_prompt(self, str_prompt: str) -> str:
        """Dispatch a completion prompt and return provider response text."""
        try:
            logger_app.info(
                "Dispatching completion prompt to %s...", self._model_client.model_name
            )
            # ai-api-unified API docs: https://pypi.org/project/ai-api-unified/
            str_response = self._model_client.send_prompt(str_prompt)
            # Normal return with provider-generated completion text.
            return str_response
        except Exception as exc_error:
            logger_app.error("Failed to generate completion: %s", exc_error)
            raise AiApiRequestError("Completion request failed.") from exc_error

    def send_structured_prompt(
        self,
        obj_structured_prompt: AIStructuredPrompt,
        cls_response_model: type[AIStructuredPrompt],
    ) -> AIStructuredPrompt:
        """Dispatch a strict-schema prompt from an AIStructuredPrompt instance."""
        try:
            logger_app.info(
                "Dispatching structured prompt to %s...", self._model_client.model_name
            )
            # ai-api-unified API docs: https://pypi.org/project/ai-api-unified/
            obj_response: AIStructuredPrompt = self._model_client.strict_schema_prompt(
                prompt=obj_structured_prompt.prompt,
                response_model=cls_response_model,
            )
            # Normal return with schema-validated structured prompt output.
            return obj_response
        except Exception as exc_error:
            logger_app.error("Failed to generate structured completion: %s", exc_error)
            raise AiApiRequestError("Structured prompt request failed.") from exc_error

    def create_images(self, str_prompt: str, int_num_images: int = 1) -> list[bytes]:
        """Generate one or more image buffers for a prompt."""
        if int_num_images < 1:
            logger_app.error("num_images must be >= 1. Received: %d", int_num_images)
            raise ValueError("num_images must be >= 1.")

        obj_image_client = self._ensure_image_client()

        try:
            logger_app.info(
                "Dispatching image prompt to %s... num_images=%d",
                obj_image_client.model_name,
                int_num_images,
            )
            obj_image_properties = AIBaseImageProperties(num_images=int_num_images)
            # ai-api-unified API docs: https://pypi.org/project/ai-api-unified/
            list_bytes_images = obj_image_client.generate_images(
                str_prompt,
                image_properties=obj_image_properties,
            )
            if not list_bytes_images:
                raise ValueError("No image bytes returned from provider.")
            # Normal return with one or more generated image buffers.
            return list_bytes_images
        except Exception as exc_error:
            logger_app.error("Failed to generate image(s): %s", exc_error)
            raise AiApiRequestError("Image generation request failed.") from exc_error

    def create_image(self, str_prompt: str) -> bytes:
        """Generate one image and return the first image byte buffer.

        Purpose:
        - Provide the simplest image-generation entrypoint used by the existing
          storyboard image path.

        Inputs:
        - `str_prompt`: Non-empty prompt text sent to the configured image model.

        Output:
        - Returns the first generated image as raw bytes.
        """
        list_bytes_images = self.create_images(str_prompt, int_num_images=1)
        bytes_first_image = list_bytes_images[0]
        # Normal return with the first generated image buffer.
        return bytes_first_image

    def create_video(
        self,
        str_prompt: str,
        obj_video_properties: AIBaseVideoProperties,
        str_video_engine: str | None = None,
        str_video_model_name: str | None = None,
    ) -> AIVideoGenerationResult:
        """Generate one AI video artifact and return the normalized result.

        Purpose:
        - Submit one video-generation request through `ai-api-unified`.
        - Wait for completion and return the normalized artifact/result payload.

        Inputs:
        - `str_prompt`: Non-empty storyboard prompt sent to the video provider.
        - `obj_video_properties`: `AIBaseVideoProperties` carrying duration,
          aspect ratio, resolution, output directory, and timeout controls.
        - `str_video_engine`: Optional provider engine override. Empty values use
          `VIDEO_ENGINE` or `google-gemini`.
        - `str_video_model_name`: Optional provider model override. Empty values
          use the provider default.

        Output:
        - Returns an `AIVideoGenerationResult` containing the normalized job
          metadata plus one or more `AIVideoArtifact` entries.
        """
        obj_video_client: AIBaseVideos = self._ensure_video_client(
            str_video_engine=str_video_engine,
            str_video_model_name=str_video_model_name,
        )

        try:
            logger_app.info(
                "Dispatching video prompt to engine=%s model=%s",
                (
                    self._tuple_video_client_config[0]
                    if self._tuple_video_client_config is not None
                    else self.str_video_engine
                ),
                obj_video_client.model_name,
            )
            obj_result: AIVideoGenerationResult = obj_video_client.generate_video(
                str_prompt,
                video_properties=obj_video_properties,
            )
            # Normal return with a completed normalized video-generation result.
            return obj_result
        except TimeoutError as exc_error:
            logger_app.error("Video generation timed out: %s", exc_error)
            raise AiApiRequestError(
                "Video generation request timed out."
            ) from exc_error
        except Exception as exc_error:
            logger_app.error("Failed to generate video: %s", exc_error)
            raise AiApiRequestError("Video generation request failed.") from exc_error

    def extract_video_frames(
        self,
        bytes_video: bytes,
        list_float_time_offsets: list[float] | None = None,
        list_int_frame_indices: list[int] | None = None,
    ) -> list[bytes]:
        """Extract image buffers from one local video buffer.

        Purpose:
        - Convert a materialized MP4 byte stream into a deterministic ordered list
          of PNG frame buffers for the mosaic renderer.

        Inputs:
        - `bytes_video`: Non-empty video bytes representing one local MP4 file.
        - `list_float_time_offsets`: Optional extraction offsets in seconds. This
          path is supported for generality but the storyboard video path prefers
          explicit integer frame indices.
        - `list_int_frame_indices`: Optional explicit integer frame indices. This
          is the preferred path for clean, deterministic storyboard extraction.

        Output:
        - Returns a `list[bytes]` where each item is one extracted PNG frame in
          the same order as the requested offsets or indices.
        """
        try:
            list_bytes_frames: list[bytes] = (
                AIBaseVideos.extract_image_frames_from_video_buffer(
                    bytes_video,
                    time_offsets_seconds=list_float_time_offsets,
                    frame_indices=list_int_frame_indices,
                    image_format="png",
                )
            )
            # Normal return with extracted PNG frame buffers in requested order.
            return list_bytes_frames
        except Exception as exc_error:
            logger_app.error("Failed to extract video frames: %s", exc_error)
            raise AiApiRequestError("Video frame extraction failed.") from exc_error


if __name__ == "__main__":
    from dotenv import load_dotenv

    # Test block per user request
    load_dotenv(".env")
    print("--- Testing AiApi Wrapper ---")

    try:
        obj_api_client = AiApi()

        str_test_prompt = (
            "In one short sentence, describe the artistic style of Chuck Close."
        )
        print(f"Prompt: '{str_test_prompt}'\n")

        str_result = obj_api_client.send_prompt(str_test_prompt)
        print(f"Response:\n{str_result}")

        print("\n--- Testing Image Generation ---")
        str_img_prompt = "Extreme close-up of a fair-skinned woman's right eye. Widescreen color photograph."
        print(f"Image Prompt: '{str_img_prompt}'\n")

        bytes_img = obj_api_client.create_image(str_img_prompt)
        str_filename_clean = "".join(str_img_prompt.split())

        str_output_dir = "output"
        os.makedirs(str_output_dir, exist_ok=True)

        str_filename = os.path.join(str_output_dir, str_filename_clean[:30] + ".png")
        with open(str_filename, "wb") as file_out:
            file_out.write(bytes_img)
        print(f"Saved image to: {str_filename}")

    except Exception as exc_e:
        print(f"\nTest Failed: {exc_e}")
        print("Did you configure your `env_template.txt` variables in your shell?")
