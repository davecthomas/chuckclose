"""Settings and interpolation helpers for mosaic rendering.

This module isolates per-frame rendering settings and interpolation helpers from
the main renderer implementation so `mosaic_generator.py` stays focused on the
runtime pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass


def lerp_float(float_a: float, float_b: float, float_progress: float) -> float:
    """Linearly interpolate between two float values."""
    float_result: float = float_a + (float_b - float_a) * float_progress
    return float_result


def lerp_int(int_a: int, int_b: int, float_progress: float) -> int:
    """Linearly interpolate between two integers using rounded output."""
    float_res: float = int_a + (int_b - int_a) * float_progress
    int_result: int = int(round(float_res))
    return int_result


@dataclass
class MosaicSettings:
    """Per-frame rendering settings for mosaic generation.

    Inputs:
    - ``int_grid_size``: Base grid size for standard mode.
    - ``float_blur_factor``: Alpha-edge softening multiplier.
    - ``int_spatial_interpolation_start``/``int_spatial_interpolation_end``:
      Optional spatial gradient bounds.
    - ``bool_supersample``: Render high-resolution primitives then downsample.
    - ``str_gradient_style``: Grid layout style selector.

    Output/Behavior:
    - Can be interpolated frame-to-frame for temporal animation.
    """

    int_grid_size: int = 30
    float_blur_factor: float = 0.0
    int_spatial_interpolation_start: int | None = None
    int_spatial_interpolation_end: int | None = None
    bool_supersample: bool = False
    str_gradient_style: str = "linear_x"

    def interpolate(
        self, obj_other: MosaicSettings, float_progress: float
    ) -> MosaicSettings:
        """Interpolate this settings object toward ``obj_other``.

        Inputs:
        - ``obj_other``: target settings at progress ``1.0``.
        - ``float_progress``: interpolation progress in ``[0.0, 1.0]``.

        Output:
        - New ``MosaicSettings`` containing interpolated values.
        """
        int_interp_grid_size: int = lerp_int(
            self.int_grid_size, obj_other.int_grid_size, float_progress
        )
        float_interp_blur_factor: float = lerp_float(
            self.float_blur_factor, obj_other.float_blur_factor, float_progress
        )

        int_interp_spatial_start: int | None = self.int_spatial_interpolation_start
        int_interp_spatial_end: int | None = self.int_spatial_interpolation_end

        # Interpolate spatial gradient bounds only when both endpoints exist.
        if (
            self.int_spatial_interpolation_start is not None
            and obj_other.int_spatial_interpolation_start is not None
        ):
            int_interp_spatial_start = lerp_int(
                self.int_spatial_interpolation_start,
                obj_other.int_spatial_interpolation_start,
                float_progress,
            )

        if (
            self.int_spatial_interpolation_end is not None
            and obj_other.int_spatial_interpolation_end is not None
        ):
            int_interp_spatial_end = lerp_int(
                self.int_spatial_interpolation_end,
                obj_other.int_spatial_interpolation_end,
                float_progress,
            )

        bool_interp_supersample: bool = (
            self.bool_supersample
            if float_progress < 0.5
            else obj_other.bool_supersample
        )
        str_interp_gradient_style: str = (
            self.str_gradient_style
            if float_progress < 0.5
            else obj_other.str_gradient_style
        )

        obj_new_settings: MosaicSettings = MosaicSettings(
            int_grid_size=int_interp_grid_size,
            float_blur_factor=float_interp_blur_factor,
            int_spatial_interpolation_start=int_interp_spatial_start,
            int_spatial_interpolation_end=int_interp_spatial_end,
            bool_supersample=bool_interp_supersample,
            str_gradient_style=str_interp_gradient_style,
        )
        return obj_new_settings
