"""Structured prompt models for storyboard frame decomposition."""

from __future__ import annotations

import textwrap
from typing import Any

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
    def validate_input(cls, dict_values: dict[str, Any]) -> dict[str, Any]:
        """Validate constructor input mode while allowing output-parse mode."""
        if "frames" in dict_values:
            return dict_values

        str_message_input = dict_values.get("message_input")
        int_num_frames = dict_values.get("int_num_frames")

        if not isinstance(str_message_input, str) or not str_message_input.strip():
            raise ValueError("message_input is required for storyboard decomposition prompt construction.")
        if not isinstance(int_num_frames, int) or int_num_frames < 1:
            raise ValueError("int_num_frames must be an integer >= 1 for storyboard decomposition.")
        return dict_values

    @model_validator(mode="after")
    def _populate_prompt(
        self: StoryboardDecomposerStructuredPrompt, __: Any
    ) -> StoryboardDecomposerStructuredPrompt:
        """Populate the inherited prompt field from input fields when present."""
        if self.message_input is not None and self.int_num_frames is not None:
            object.__setattr__(
                self,
                "prompt",
                StoryboardDecomposerStructuredPrompt.get_prompt(
                    message_input=self.message_input,
                    int_num_frames=self.int_num_frames,
                ),
            )
        return self

    @staticmethod
    def get_prompt(message_input: str | None = None, int_num_frames: int | None = None) -> str:
        """Build the storyboard decomposition instruction for the LLM."""
        if message_input is None or int_num_frames is None:
            return ""

        str_prompt = textwrap.dedent(
            f"""
            Break this storyboard into {int_num_frames} image generation prompts in sequence.

            Rules:
            - Return exactly {int_num_frames} frame objects in "frames".
            - frame_index values must start at 0 and be sequential.
            - Each frame_prompt must be self-contained and fully specified.
            - Keep subject identity, camera framing, lens context, lighting, and style consistent across frames.
            - Only change temporal motion from frame to frame.

            Storyboard:
            {message_input}
            """
        ).strip()
        return str_prompt


class StoryboardDecomposerStructuredResult(AIStructuredPrompt):
    """Structured prompt result schema for storyboard frame decomposition."""

    frames: list[StoryboardFrame]

    @staticmethod
    def get_prompt() -> str:
        """Return empty prompt text for response-schema-only model usage."""
        str_prompt = ""
        return str_prompt
