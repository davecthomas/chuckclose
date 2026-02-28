"""Input model for selecting the mosaic source image data.

This module centralizes validation for mosaic image sources. The goal is to make
constructor inputs explicit and deterministic: callers must supply exactly one
source type.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MosaicImageInputs:
    """Validated source-image inputs for ``Mosaic``.

    Inputs:
    - ``str_input_image_path``: Filesystem path to a single source image.
    - ``list_bytes_frame_image_buffers``: Ordered in-memory image buffers.

    Output/Behavior:
    - Exactly one input source is allowed.
    - ``list_bytes_frame_image_buffers`` must be non-empty when provided.
    - The class stores a defensive copy of the buffer list.
    """

    str_input_image_path: str | None = None
    list_bytes_frame_image_buffers: list[bytes] | None = None

    def __post_init__(self) -> None:
        """Validate source exclusivity and normalize stored values."""
        bool_has_image_path_field: bool = self.str_input_image_path is not None
        bool_has_frame_buffers_field: bool = (
            self.list_bytes_frame_image_buffers is not None
        )

        if bool_has_image_path_field and bool_has_frame_buffers_field:
            raise ValueError(
                "Provide only one image input source: path or frame buffers."
            )

        if not bool_has_image_path_field and not bool_has_frame_buffers_field:
            raise ValueError(
                "One image input source is required: path or frame buffers."
            )

        if self.str_input_image_path is not None:
            str_normalized_path: str = self.str_input_image_path.strip()
            if not str_normalized_path:
                raise ValueError("Image path cannot be empty or whitespace.")
            self.str_input_image_path = str_normalized_path

        if self.list_bytes_frame_image_buffers is not None:
            list_bytes_copy: list[bytes] = list(self.list_bytes_frame_image_buffers)
            if len(list_bytes_copy) == 0:
                raise ValueError("Frame image buffer list cannot be empty.")
            for int_index, bytes_buffer in enumerate(list_bytes_copy):
                if not isinstance(bytes_buffer, bytes) or len(bytes_buffer) == 0:
                    raise ValueError(
                        f"Frame buffer at index {int_index} must be non-empty bytes."
                    )
            self.list_bytes_frame_image_buffers = list_bytes_copy
