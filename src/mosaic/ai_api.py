import logging
from ai_api_unified import AIFactory, AIBaseCompletions

# Apply AGENTS.md preferred structured logging format
logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)
logger_app = logging.getLogger(__name__)

class AiApi:
    """
    A thin wrapper around the ai-api-unified library for localized application calls.
    Specifically configured for Google Gemini Completions via API Keys.
    """
    
    def __init__(self) -> None:
        """
        Initializes the underlying completions client using the Google Gemini engine.
        Auth relies on env vars (`GOOGLE_AUTH_METHOD=api_key`, `GOOGLE_GEMINI_API_KEY`).
        """
        try:
            self._model_client: AIBaseCompletions = AIFactory.get_ai_completions_client(
                completions_engine="google-gemini"
            )
            logger_app.info("Fully initialized %s client.", self._model_client.model_name)
        except Exception as exc_error:
            logger_app.error("Failed to initialize ai-api-unified client: %s", exc_error)
            raise

    def send_prompt(self, str_prompt: str) -> str:
        """Dispatches a basic string completion prompt to the configured model."""
        try:
            logger_app.info("Dispatching prompt to %s...", self._model_client.model_name)
            str_response = self._model_client.send_prompt(str_prompt)
            return str_response
        except Exception as exc_error:
            logger_app.error("Failed to generate completion: %s", exc_error)
            raise

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
        
    except Exception as exc_e:
        print(f"\nTest Failed: {exc_e}")
        print("Did you configure your `env_template.txt` variables in your shell?")
