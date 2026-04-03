"""Structured prompt models for storyboard frame decomposition."""

from __future__ import annotations

import textwrap

from ai_api_unified import AIStructuredPrompt
from pydantic import BaseModel
from pydantic import Field
from pydantic import model_validator


class StoryboardFrame(BaseModel):
    """Typed frame entry for storyboard prompt decomposition."""

    frame_index: int = Field(ge=0)
    frame_prompt: str = Field(min_length=1)


class StoryboardDecomposerStructuredPrompt(AIStructuredPrompt):
    """
    Structured prompt model for storyboard decomposition.

    Input fields are used to construct ``prompt``.
    Output schema is narrowed to ``frames`` for strict-schema responses.
    """

    message_input: str | None = None
    int_num_frames: int | None = Field(default=None, ge=1)
    frames: list[StoryboardFrame] | None = None

    @model_validator(mode="before")
    def validate_input(cls, dict_values: dict) -> dict:
        """Validate constructor input mode while allowing output-parse mode."""
        if "frames" in dict_values:
            # Early return: this is an output-parse call, not a prompt construction call.
            return dict_values

        str_message_input = dict_values.get("message_input")
        int_num_frames = dict_values.get("int_num_frames")

        if not isinstance(str_message_input, str) or not str_message_input.strip():
            raise ValueError(
                "message_input is required for storyboard decomposition prompt construction."
            )
        if not isinstance(int_num_frames, int) or int_num_frames < 1:
            raise ValueError(
                "int_num_frames must be an integer >= 1 for storyboard decomposition."
            )
        return dict_values

    def get_prompt(self) -> str:
        """Build the storyboard decomposition instruction for the LLM.

        Reads ``self.message_input`` and ``self.int_num_frames`` directly so that
        the base-class ``_populate_prompt`` validator can call ``self.get_prompt()``
        with no arguments. Returns an empty string when either field is absent
        (output-parse mode).

        Output:
        - Fully formatted LLM instruction string, or empty string in parse mode.
        """
        if self.message_input is None or self.int_num_frames is None:
            # Early return: output-parse mode; prompt text is not needed.
            return ""

        str_prompt = textwrap.dedent(
            f"""
            Break this storyboard into {self.int_num_frames} image generation prompts in sequence.

            Rules:
            - Return exactly {self.int_num_frames} frame objects in "frames".
            - frame_index values must start at 0 and be sequential.
            - Each frame_prompt must be self-contained and fully specified.
            - Keep subject identity, camera framing, lens context, lighting, and style consistent across frames.
            - Only change temporal motion from frame to frame.

            Storyboard:
            {self.message_input}
            """
        ).strip()
        return str_prompt


class StoryboardDecomposerStructuredResult(AIStructuredPrompt):
    """Structured prompt result schema for storyboard frame decomposition."""

    frames: list[StoryboardFrame]

    def get_prompt(self) -> str:
        """Return empty prompt text for response-schema-only model usage.

        Output:
        - Empty string; this model is only used to validate structured LLM output.
        """
        str_prompt = ""
        # Normal return: result-schema model has no prompt of its own.
        return str_prompt
