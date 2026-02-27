"""Structured prompt models for storyboard frame decomposition."""

from __future__ import annotations

import textwrap
from copy import deepcopy
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
    def populate_prompt(self) -> StoryboardDecomposerStructuredPrompt:
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
    def get_prompt(message_input: str, int_num_frames: int) -> str:
        """Build the storyboard decomposition instruction for the LLM."""
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

    @classmethod
    def model_json_schema(cls) -> dict[str, Any]:
        """Return output-only schema consumed by strict-schema prompting."""
        dict_schema: dict[str, Any] = deepcopy(super().model_json_schema())
        dict_schema["type"] = "object"
        dict_schema["properties"] = {
            "frames": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "frame_index": {"type": "integer", "minimum": 0},
                        "frame_prompt": {"type": "string", "minLength": 1},
                    },
                    "required": ["frame_index", "frame_prompt"],
                },
            }
        }
        dict_schema["required"] = ["frames"]
        dict_schema["additionalProperties"] = False
        return dict_schema
