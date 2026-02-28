import logging
import os

from ai_api_unified import AIFactory
from ai_api_unified import AIBaseCompletions
from ai_api_unified import AIBaseImageProperties
from ai_api_unified import AIBaseImages
from ai_api_unified import AIStructuredPrompt


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
            logger_app.error("Failed to initialize ai-api-unified completions client: %s", exc_error)
            raise

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
        """Dispatch a completion prompt and return provider response text."""
        try:
            logger_app.info("Dispatching completion prompt to %s...", self._model_client.model_name)
            # ai-api-unified API docs: https://pypi.org/project/ai-api-unified/
            str_response = self._model_client.send_prompt(str_prompt)
            return str_response
        except Exception as exc_error:
            logger_app.error("Failed to generate completion: %s", exc_error)
            raise

    def send_structured_prompt(
        self,
        obj_structured_prompt: AIStructuredPrompt,
        cls_response_model: type[AIStructuredPrompt],
    ) -> AIStructuredPrompt:
        """Dispatch a strict-schema prompt from an AIStructuredPrompt instance."""
        try:
            logger_app.info("Dispatching structured prompt to %s...", self._model_client.model_name)
            # ai-api-unified API docs: https://pypi.org/project/ai-api-unified/
            obj_response: AIStructuredPrompt = self._model_client.strict_schema_prompt(
                prompt=obj_structured_prompt.prompt,
                response_model=cls_response_model,
            )
            return obj_response
        except Exception as exc_error:
            logger_app.error("Failed to generate structured completion: %s", exc_error)
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
            obj_image_properties = AIBaseImageProperties(num_images=int_num_images)
            # ai-api-unified API docs: https://pypi.org/project/ai-api-unified/
            list_bytes_images = obj_image_client.generate_images(
                str_prompt,
                image_properties=obj_image_properties,
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
