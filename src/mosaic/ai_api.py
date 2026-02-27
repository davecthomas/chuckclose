import logging
import os
import time
from collections.abc import Callable
from functools import partial
from typing import TypeVar

from ai_api_unified import AIFactory
from ai_api_unified import AIBaseCompletions
from ai_api_unified import AIBaseImageProperties
from ai_api_unified import AIBaseImages

# Apply AGENTS.md preferred structured logging format
logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)
logger_app = logging.getLogger(__name__)

TypeRetryResult = TypeVar("TypeRetryResult")


class AiApi:
    """
    A thin wrapper around the ai-api-unified library for localized application calls.
    Specifically configured for provider selection through environment variables.
    """
    
    def __init__(self) -> None:
        """
        Initialize retry policy and provider clients.

        Environment variables:
        - `COMPLETIONS_ENGINE` defaults to `google-gemini`.
        - `GOOGLE_AUTH_METHOD=api_key`
        - `GOOGLE_GEMINI_API_KEY`
        """
        self.int_max_retries = 3
        self.float_retry_base_seconds = 0.5
        self.float_retry_max_seconds = 4.0
        self._image_client: AIBaseImages | None = None

        self.str_completions_engine = os.getenv("COMPLETIONS_ENGINE", "google-gemini")
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
            logger_app.error("Failed to initialize ai-api-unified client: %s", exc_error)
            raise

    def _is_retryable_exception(self, exc_error: Exception) -> bool:
        """Return True when an exception is likely transient and retryable."""
        str_exception_name = exc_error.__class__.__name__.lower()
        str_exception_text = str(exc_error).lower()

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
        bool_text_retryable = any(str_marker in str_exception_text for str_marker in list_str_retry_markers)
        bool_retryable = bool_name_retryable or bool_text_retryable
        return bool_retryable

    def _execute_with_retry(
        self,
        fn_operation: Callable[[str], TypeRetryResult],
        str_payload: str,
        str_operation_name: str,
    ) -> TypeRetryResult:
        """Execute provider operation with bounded exponential backoff."""
        for int_attempt_index in range(self.int_max_retries):
            try:
                obj_result = fn_operation(str_payload)
                return obj_result
            except Exception as exc_error:
                bool_retryable = self._is_retryable_exception(exc_error)
                bool_has_attempt_remaining = int_attempt_index < (self.int_max_retries - 1)
                if not bool_retryable or not bool_has_attempt_remaining:
                    logger_app.error(
                        "AI operation failed. operation=%s attempt=%d retryable=%s context=%s",
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
                    "Retrying AI operation. operation=%s attempt=%d sleep_seconds=%.2f context=%s",
                    str_operation_name,
                    int_attempt_index + 1,
                    float_sleep_seconds,
                    exc_error,
                )
                time.sleep(float_sleep_seconds)

        raise RuntimeError("Unreachable retry state encountered.")

    def _send_prompt_once(self, str_prompt: str) -> str:
        """Single-attempt completion request to provider client."""
        # ai-api-unified API docs: https://pypi.org/project/ai-api-unified/
        str_response = self._model_client.send_prompt(str_prompt)
        return str_response

    def _ensure_image_client(self) -> AIBaseImages:
        """Lazily initialize and return the image provider client."""
        if self._image_client is None:
            try:
                # ai-api-unified API docs: https://pypi.org/project/ai-api-unified/
                self._image_client = AIFactory.get_ai_images_client()
                logger_app.info("Initialized image client. model=%s", self._image_client.model_name)
            except Exception as exc_error:
                logger_app.error("Failed to initialize ai-api-unified image client: %s", exc_error)
                raise

        obj_image_client = self._image_client
        return obj_image_client

    def send_prompt(self, str_prompt: str) -> str:
        """Dispatch a completion prompt with retry/backoff behavior."""
        try:
            logger_app.info("Dispatching completion prompt to %s...", self._model_client.model_name)
            str_response = self._execute_with_retry(
                self._send_prompt_once,
                str_prompt,
                "send_prompt",
            )
            return str_response
        except Exception as exc_error:
            logger_app.error("Failed to generate completion: %s", exc_error)
            raise

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
            fn_generate_images = partial(
                self._generate_images_once,
                obj_image_client=obj_image_client,
                int_num_images=int_num_images,
            )
            list_bytes_images = self._execute_with_retry(
                fn_generate_images,
                str_prompt,
                "create_images",
            )
            if not list_bytes_images:
                raise ValueError("No image bytes returned from provider.")
            return list_bytes_images
        except Exception as exc_error:
            logger_app.error("Failed to generate image(s): %s", exc_error)
            raise

    def create_image(self, str_prompt: str) -> bytes:
        """Generate one image and return the first image byte buffer."""
        list_bytes_images = self.create_images(str_prompt, int_num_images=1)
        bytes_first_image = list_bytes_images[0]
        return bytes_first_image

    @staticmethod
    def _generate_images_once(
        str_prompt: str,
        obj_image_client: AIBaseImages,
        int_num_images: int,
    ) -> list[bytes]:
        """Single-attempt image generation call to provider client."""
        obj_image_properties = AIBaseImageProperties(num_images=int_num_images)
        # ai-api-unified API docs: https://pypi.org/project/ai-api-unified/
        list_bytes_images = obj_image_client.generate_images(
            str_prompt,
            image_properties=obj_image_properties,
        )
        return list_bytes_images

if __name__ == "__main__":
    from dotenv import load_dotenv
    
    # Test block per user request
    load_dotenv(".env")
    print("--- Testing AiApi Wrapper ---")
    
    try:
        obj_api_client = AiApi()
        
        str_test_prompt = "In one short sentence, describe the artistic style of Chuck Close."
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
