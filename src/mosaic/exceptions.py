"""Custom exception hierarchy for the mosaic package.

These exceptions allow callers to distinguish between input validation failures,
render-time failures, export failures, and AI provider failures without catching
broad base-class exceptions.
"""

from __future__ import annotations


class MosaicInputError(ValueError):
    """Raised when caller-supplied input values are invalid or out of bounds.

    Inputs: inherits from ValueError so existing code that catches ValueError
    continues to work.

    Output/Behavior: represents a caller error, not a system error.
    """


class MosaicRenderError(RuntimeError):
    """Raised when image decoding or mosaic rendering fails at runtime.

    Inputs: inherits from RuntimeError.

    Output/Behavior: represents a system or data error during rendering that is
    not caused by invalid caller arguments.
    """


class MosaicExportError(RuntimeError):
    """Raised when writing an image or video file to disk fails.

    Inputs: inherits from RuntimeError.

    Output/Behavior: represents an I/O failure during export.
    """


class AiApiInitError(RuntimeError):
    """Raised when an AI provider client fails to initialize.

    Inputs: inherits from RuntimeError.

    Output/Behavior: non-retryable; indicates a configuration or credential
    problem that must be resolved before retrying.
    """


class AiApiRequestError(RuntimeError):
    """Raised when an AI provider request fails after all provider-level retries.

    Inputs: inherits from RuntimeError.

    Output/Behavior: may be retryable depending on the underlying cause; callers
    should inspect the chained exception for provider error details.
    """
