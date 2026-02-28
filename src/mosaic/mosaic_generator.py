"""Core mosaic rendering and video assembly runtime.

This module owns the rendering pipeline for static images and videos. It also
contains the CLI entrypoint used by ``poetry run mosaic``.
"""

from __future__ import annotations

import importlib.metadata
import io
import logging
import math
import os
import secrets
import sys
from collections.abc import Generator

from PIL import Image, ImageDraw, ImageFilter

from .mosaic_image_inputs import MosaicImageInputs
from .mosaic_settings import MosaicSettings
from .mosaic_settings import lerp_float
from .mosaic_settings import lerp_int

# Re-export interpolation helpers from this module path for existing imports.

# Structured logging without timestamps for cleaner CLI output.
logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)
logger_app = logging.getLogger(__name__)


try:
    import cv2
    import numpy as np
    from tqdm import tqdm

    bool_has_video_support = True
except ImportError:
    bool_has_video_support = False


def get_version() -> str:
    """Retrieve package version from installed metadata."""
    try:
        str_version_result: str = importlib.metadata.version("mosaic")
        return str_version_result
    except importlib.metadata.PackageNotFoundError as exc_error:
        logger_app.warning(
            "Package 'mosaic' not found. Using 'unknown' version. Context: %s",
            exc_error,
        )
        str_unknown_version: str = "unknown"
        return str_unknown_version


class Mosaic:
    """Render Chuck Close style mosaics from one image source definition.

    Constructor Input:
    - ``obj_mosaic_image_inputs``: ``MosaicImageInputs`` describing exactly one
      input source (a file path or a list of image buffers).

    Output/Behavior:
    - Maintains normalized RGB source frames internally.
    - ``render_buffer`` renders one mosaic frame from a selected source frame.
    - ``generate_video`` supports temporal interpolation of settings and source
      frame progression when multiple source frames exist.
    """

    def __init__(self, obj_mosaic_image_inputs: MosaicImageInputs) -> None:
        """Initialize renderer state from validated image inputs."""
        self.obj_mosaic_image_inputs: MosaicImageInputs = obj_mosaic_image_inputs
        self.list_image_source_frames: list[Image.Image] = self._load_source_images(
            obj_mosaic_image_inputs
        )

        self.image_original: Image.Image = self.list_image_source_frames[0]
        self.int_width: int
        self.int_height: int
        self.int_width, self.int_height = self.image_original.size

        # Runtime writer is lazily initialized by save_video_fragment.
        self._video_writer: object | None = None

    @staticmethod
    def _decode_image_buffer(bytes_image: bytes, int_frame_index: int) -> Image.Image:
        """Decode one RGB image from bytes with indexed error context."""
        try:
            obj_bytes_io: io.BytesIO = io.BytesIO(bytes_image)
            image_result: Image.Image = Image.open(obj_bytes_io).convert("RGB")
            return image_result
        except Exception as exc_error:
            logger_app.error(
                "Failed to decode source image buffer at index %d. Context: %s",
                int_frame_index,
                exc_error,
            )
            raise RuntimeError(
                f"Failed to decode source image buffer at index {int_frame_index}."
            ) from exc_error

    @staticmethod
    def _normalize_source_frame_sizes(
        list_image_source_frames: list[Image.Image],
    ) -> list[Image.Image]:
        """Resize source frames to a shared size using the first frame as reference."""
        if len(list_image_source_frames) == 0:
            raise ValueError("At least one source image frame is required.")

        tuple_reference_size: tuple[int, int] = list_image_source_frames[0].size
        list_image_normalized_frames: list[Image.Image] = []
        for image_source_frame in list_image_source_frames:
            if image_source_frame.size != tuple_reference_size:
                image_source_frame = image_source_frame.resize(
                    tuple_reference_size, Image.Resampling.LANCZOS
                )
            list_image_normalized_frames.append(image_source_frame)

        return list_image_normalized_frames

    def _load_source_images(
        self, obj_mosaic_image_inputs: MosaicImageInputs
    ) -> list[Image.Image]:
        """Load and normalize source images from file or in-memory buffers."""
        if obj_mosaic_image_inputs.str_input_image_path is not None:
            str_input_image_path: str = obj_mosaic_image_inputs.str_input_image_path
            try:
                image_source: Image.Image = Image.open(str_input_image_path).convert("RGB")
            except Exception as exc_error:
                logger_app.error(
                    "Failed to open image file: %s. Context: %s",
                    str_input_image_path,
                    exc_error,
                )
                raise RuntimeError(f"Error opening image: {exc_error}") from exc_error

            list_image_result: list[Image.Image] = [image_source]
            return list_image_result

        if obj_mosaic_image_inputs.list_bytes_frame_image_buffers is not None:
            list_bytes_buffers: list[bytes] = (
                obj_mosaic_image_inputs.list_bytes_frame_image_buffers
            )
            list_image_source_frames: list[Image.Image] = []
            for int_frame_index, bytes_image in enumerate(list_bytes_buffers):
                image_source_frame: Image.Image = self._decode_image_buffer(
                    bytes_image,
                    int_frame_index,
                )
                list_image_source_frames.append(image_source_frame)

            list_image_normalized_frames: list[Image.Image] = (
                self._normalize_source_frame_sizes(list_image_source_frames)
            )
            return list_image_normalized_frames

        logger_app.error("Mosaic input source did not contain a valid image source.")
        raise RuntimeError("Mosaic input source did not contain a valid image source.")

    def _resolve_source_frame_index(
        self, int_output_frame_index: int, int_output_total_frames: int
    ) -> int:
        """Map output frame position to a deterministic source-frame index."""
        int_source_frame_count: int = len(self.list_image_source_frames)
        if int_source_frame_count <= 1 or int_output_total_frames <= 1:
            return 0

        float_position: float = (
            int_output_frame_index
            * (int_source_frame_count - 1)
            / float(int_output_total_frames - 1)
        )
        int_source_index: int = int(round(float_position))
        int_source_index = max(0, min(int_source_index, int_source_frame_count - 1))
        return int_source_index

    def _get_source_image_by_index(self, int_source_frame_index: int) -> Image.Image:
        """Return a source frame with index clamping for safety."""
        int_source_frame_count: int = len(self.list_image_source_frames)
        if int_source_frame_count == 0:
            raise RuntimeError("No source images are available for rendering.")

        int_source_index: int = max(0, min(int_source_frame_index, int_source_frame_count - 1))
        image_source: Image.Image = self.list_image_source_frames[int_source_index]
        return image_source

    @staticmethod
    def get_shape_size_factor_bounds(int_grid_size: int) -> tuple[float, float]:
        """Return adaptive shape size-factor bounds for a grid size band."""
        if int_grid_size >= 8:
            tuple_bounds: tuple[float, float] = (0.90, 0.90)
            return tuple_bounds
        if int_grid_size >= 4:
            tuple_bounds = (0.93, 0.95)
            return tuple_bounds

        tuple_bounds = (0.96, 0.98)
        return tuple_bounds

    @staticmethod
    def get_adaptive_shape_dimension(
        int_grid_size: int, int_reference_dimension: int
    ) -> int:
        """Compute shape size using adaptive fill ratios by grid-size band."""
        tuple_factor_bounds: tuple[float, float] = Mosaic.get_shape_size_factor_bounds(
            int_grid_size
        )
        float_min_factor: float
        float_max_factor: float
        float_min_factor, float_max_factor = tuple_factor_bounds
        if float_min_factor == float_max_factor:
            float_shape_factor: float = float_min_factor
        else:
            float_shape_factor = secrets.SystemRandom().uniform(
                float_min_factor, float_max_factor
            )

        float_dimension: float = float(int_reference_dimension) * float_shape_factor
        int_dimension: int = max(1, int(round(float_dimension)))
        return int_dimension

    @staticmethod
    def get_dominant_color(
        image_input: Image.Image, tuple_region_box: tuple[int, int, int, int]
    ) -> tuple[int, int, int]:
        """Extract dominant color for one sample box in the source image."""
        try:
            int_left: int
            int_top: int
            int_right: int
            int_bottom: int
            int_left, int_top, int_right, int_bottom = tuple_region_box
            int_left = max(0, int_left)
            int_top = max(0, int_top)
            int_right = min(image_input.width, int_right)
            int_bottom = min(image_input.height, int_bottom)

            if int_right <= int_left or int_bottom <= int_top:
                tuple_black_color: tuple[int, int, int] = (0, 0, 0)
                return tuple_black_color

            image_cell: Image.Image = image_input.crop(
                (int_left, int_top, int_right, int_bottom)
            )
            if image_cell.width == 0 or image_cell.height == 0:
                tuple_black_color = (0, 0, 0)
                return tuple_black_color

            image_dominant: Image.Image = image_cell.quantize(colors=1)
            list_palette: list[int] | None = image_dominant.getpalette()
            if list_palette:
                tuple_dom_color: tuple[int, int, int] = (
                    list_palette[0],
                    list_palette[1],
                    list_palette[2],
                )
                return tuple_dom_color

            tuple_black_color = (0, 0, 0)
            return tuple_black_color
        except Exception as exc_error:
            logger_app.warning(
                "Failed to extract dominant color; defaulting to gray. Context: %s",
                exc_error,
            )
            tuple_gray_color: tuple[int, int, int] = (128, 128, 128)
            return tuple_gray_color

    @staticmethod
    def render_shape(
        tuple_color: tuple[int, int, int],
        int_width: int,
        int_height: int,
        str_shape_type: str = "random",
        bool_supersample: bool = False,
        float_blur_radius: float = 0.0,
    ) -> Image.Image:
        """Render one shape tile to an RGBA image buffer."""
        int_width = max(1, int_width)
        int_height = max(1, int_height)

        if str_shape_type == "random":
            list_shape_options: list[str] = ["square", "circle"]
            str_shape_type = secrets.choice(list_shape_options)

        int_scale: int = 4 if bool_supersample else 1
        int_draw_width: int = max(1, int_width * int_scale)
        int_draw_height: int = max(1, int_height * int_scale)

        tuple_transparent: tuple[int, int, int, int] = (0, 0, 0, 0)
        image_shape_layer: Image.Image = Image.new(
            "RGBA", (int_draw_width, int_draw_height), tuple_transparent
        )
        image_draw: ImageDraw.ImageDraw = ImageDraw.Draw(image_shape_layer)

        tuple_draw_box: tuple[int, int, int, int] = (
            0,
            0,
            int_draw_width,
            int_draw_height,
        )
        if str_shape_type == "square":
            int_min_dim: int = min(int_width, int_height)
            if int_min_dim < 3:
                float_corner_ratio: float = 0.05
            else:
                float_corner_ratio = 0.2

            int_radius: int = int(
                round(int(min(int_draw_width, int_draw_height)) * float_corner_ratio)
            )
            image_draw.rounded_rectangle(
                tuple_draw_box, radius=int_radius, fill=tuple_color
            )
        else:
            image_draw.ellipse(tuple_draw_box, fill=tuple_color)

        if bool_supersample:
            tuple_final_size: tuple[int, int] = (int_width, int_height)
            image_shape_layer = image_shape_layer.resize(
                tuple_final_size, Image.Resampling.LANCZOS
            )

        if float_blur_radius > 0:
            tuple_channels: tuple[Image.Image, Image.Image, Image.Image, Image.Image] = (
                image_shape_layer.split()
            )
            channel_alpha: Image.Image = tuple_channels[3]
            channel_alpha_blurred: Image.Image = channel_alpha.filter(
                ImageFilter.GaussianBlur(radius=float_blur_radius)
            )
            tuple_merged_channels: tuple[Image.Image, Image.Image, Image.Image, Image.Image] = (
                tuple_channels[0],
                tuple_channels[1],
                tuple_channels[2],
                channel_alpha_blurred,
            )
            image_shape_layer = Image.merge("RGBA", tuple_merged_channels)

        return image_shape_layer

    @staticmethod
    def generate_standard_grid(
        int_width: int,
        int_height: int,
        int_grid_size: int,
        float_blur_factor: float,
    ) -> Generator[
        tuple[tuple[int, int, int, int], tuple[int, int], tuple[int, int], float],
        None,
        None,
    ]:
        """Yield sample/paste geometry for a standard uniform grid."""
        int_cols: int = int_width // int_grid_size
        int_rows: int = int_height // int_grid_size
        int_shape_width: int = Mosaic.get_adaptive_shape_dimension(
            int_grid_size, int_grid_size
        )
        if int_grid_size < 3:
            float_blur_radius: float = 0.0
        else:
            float_blur_radius = (
                max(1.0, float(int_grid_size) * float_blur_factor)
                if float_blur_factor > 0
                else 0.0
            )

        for row_idx in range(int_rows):
            for col_idx in range(int_cols):
                int_left: int = col_idx * int_grid_size
                int_top: int = row_idx * int_grid_size
                int_right: int = int_left + int_grid_size
                int_bottom: int = int_top + int_grid_size
                int_center_x: int = int_left + (int_grid_size // 2)
                int_center_y: int = int_top + (int_grid_size // 2)

                tuple_box: tuple[int, int, int, int] = (
                    int_left,
                    int_top,
                    int_right,
                    int_bottom,
                )
                tuple_center: tuple[int, int] = (int_center_x, int_center_y)
                tuple_dims: tuple[int, int] = (int_shape_width, int_shape_width)

                yield tuple_box, tuple_center, tuple_dims, float_blur_radius

    @staticmethod
    def generate_linear_gradient(
        int_width: int,
        int_height: int,
        int_start_size: int,
        int_end_size: int,
        float_blur_factor: float,
        str_axis: str = "x",
        str_gradient_style: str = "linear",
    ) -> Generator[
        tuple[tuple[int, int, int, int], tuple[int, int], tuple[int, int], float],
        None,
        None,
    ]:
        """Yield geometry for a linear or center-weighted gradient grid."""
        bool_is_x: bool = str_axis == "x"
        int_primary_dim: int = int_width if bool_is_x else int_height
        int_secondary_dim: int = int_height if bool_is_x else int_width
        float_center_pos: float = int_primary_dim / 2.0
        int_current_pos: int = 0

        while int_current_pos < int_primary_dim:
            if str_gradient_style in ("center_x", "center_y"):
                float_progress: float = abs(int_current_pos - float_center_pos) / (
                    int_primary_dim / 2.0
                )
            else:
                float_progress = int_current_pos / float(int_primary_dim)

            float_progress = min(max(float_progress, 0.0), 1.0)

            float_calc_size: float = int_start_size + (
                int_end_size - int_start_size
            ) * float_progress
            int_grid_size: int = max(1, int(round(float_calc_size)))
            if int_grid_size < 3:
                float_blur_radius = 0.0
            else:
                float_blur_radius = (
                    max(1.0, float(int_grid_size) * float_blur_factor)
                    if float_blur_factor > 0
                    else 0.0
                )

            for int_sec_pos in range(0, int_secondary_dim, int_grid_size):
                int_left: int = int_current_pos if bool_is_x else int_sec_pos
                int_top: int = int_sec_pos if bool_is_x else int_current_pos
                int_right: int = int_left + int_grid_size
                int_bottom: int = int_top + int_grid_size

                int_center_x_pos: int = int_left + (int_grid_size // 2)
                int_center_y_pos: int = int_top + (int_grid_size // 2)

                int_size_dim: int = Mosaic.get_adaptive_shape_dimension(
                    int_grid_size, int_grid_size
                )

                tuple_box: tuple[int, int, int, int] = (
                    int_left,
                    int_top,
                    int_right,
                    int_bottom,
                )
                tuple_center: tuple[int, int] = (int_center_x_pos, int_center_y_pos)
                tuple_dims: tuple[int, int] = (int_size_dim, int_size_dim)

                yield tuple_box, tuple_center, tuple_dims, float_blur_radius

            int_current_pos += int_grid_size

    @staticmethod
    def generate_radial_grid(
        int_width: int,
        int_height: int,
        int_start_size: int,
        int_end_size: int,
        float_blur_factor: float,
    ) -> Generator[
        tuple[tuple[int, int, int, int], tuple[int, int], tuple[int, int], float],
        None,
        None,
    ]:
        """Yield geometry for concentric radial grid layout."""
        float_cx: float = int_width / 2.0
        float_cy: float = int_height / 2.0
        float_max_radius: float = math.sqrt((int_width / 2.0) ** 2 + (int_height / 2.0) ** 2) * 1.05
        float_current_r: float = 0.0

        while float_current_r < float_max_radius:
            float_progress: float = float_current_r / float_max_radius
            float_calc_width: float = int_start_size + (
                int_end_size - int_start_size
            ) * float_progress
            float_radial_width: float = max(2.0, float_calc_width)
            int_grid_size: int = max(1, int(round(float_radial_width)))
            if int_grid_size < 3:
                float_blur_radius: float = 0.0
            else:
                float_blur_radius = (
                    max(1.0, float_radial_width * float_blur_factor)
                    if float_blur_factor > 0
                    else 0.0
                )

            if float_current_r < 1:
                int_num_spokes: int = 1
            else:
                float_circumference: float = 2.0 * math.pi * float_current_r
                int_num_spokes = max(4, int(float_circumference / float_radial_width))

            float_d_theta: float = 360.0 / int_num_spokes

            for index_spoke in range(int_num_spokes):
                float_theta_start: float = index_spoke * float_d_theta
                float_theta_end: float = (index_spoke + 1) * float_d_theta
                float_rc: float = float_current_r + (float_radial_width / 2.0)
                float_theta_c_rad: float = math.radians(
                    (float_theta_start + float_theta_end) / 2.0
                )

                float_cent_x: float = float_cx + float_rc * math.cos(float_theta_c_rad)
                float_cent_y: float = float_cy + float_rc * math.sin(float_theta_c_rad)

                int_sample_half_size: int = max(1, int(float_radial_width / 2.0))
                int_left: int = int(float_cent_x - int_sample_half_size)
                int_top: int = int(float_cent_y - int_sample_half_size)
                int_right: int = int(float_cent_x + int_sample_half_size)
                int_bottom: int = int(float_cent_y + int_sample_half_size)

                if (
                    int_right < 0
                    or int_bottom < 0
                    or int_left > int_width
                    or int_top > int_height
                ):
                    continue

                float_arc_len: float = float_rc * math.radians(float_d_theta)
                float_max_dim: float = min(float_radial_width, float_arc_len)

                int_size_dim: int = Mosaic.get_adaptive_shape_dimension(
                    int_grid_size, max(1, int(round(float_max_dim)))
                )

                tuple_box: tuple[int, int, int, int] = (
                    int_left,
                    int_top,
                    int_right,
                    int_bottom,
                )
                tuple_center: tuple[int, int] = (int(float_cent_x), int(float_cent_y))
                tuple_dims: tuple[int, int] = (int_size_dim, int_size_dim)

                yield tuple_box, tuple_center, tuple_dims, float_blur_radius

            float_current_r += float_radial_width

    def render_buffer(
        self, obj_settings: MosaicSettings, int_source_frame_index: int = 0
    ) -> Image.Image:
        """Render a mosaic image from settings and one selected source frame.

        Inputs:
        - ``obj_settings``: rendering settings for this output frame.
        - ``int_source_frame_index``: source-frame index for color sampling.

        Output:
        - RGB Pillow image representing one rendered mosaic frame.
        """
        image_source: Image.Image = self._get_source_image_by_index(int_source_frame_index)

        tuple_white_color: tuple[int, int, int, int] = (255, 255, 255, 255)
        tuple_size: tuple[int, int] = (self.int_width, self.int_height)
        image_output: Image.Image = Image.new("RGBA", tuple_size, tuple_white_color)

        obj_generator: Generator[
            tuple[tuple[int, int, int, int], tuple[int, int], tuple[int, int], float],
            None,
            None,
        ]
        if (
            obj_settings.int_spatial_interpolation_start is not None
            and obj_settings.int_spatial_interpolation_end is not None
        ):
            int_start_size: int = obj_settings.int_spatial_interpolation_start
            int_end_size: int = obj_settings.int_spatial_interpolation_end
            if obj_settings.str_gradient_style == "radial":
                obj_generator = self.generate_radial_grid(
                    self.int_width,
                    self.int_height,
                    int_start_size,
                    int_end_size,
                    obj_settings.float_blur_factor,
                )
            elif obj_settings.str_gradient_style == "center_y":
                obj_generator = self.generate_linear_gradient(
                    self.int_width,
                    self.int_height,
                    int_start_size,
                    int_end_size,
                    obj_settings.float_blur_factor,
                    str_axis="y",
                    str_gradient_style="center_y",
                )
            elif obj_settings.str_gradient_style == "center_x":
                obj_generator = self.generate_linear_gradient(
                    self.int_width,
                    self.int_height,
                    int_start_size,
                    int_end_size,
                    obj_settings.float_blur_factor,
                    str_axis="x",
                    str_gradient_style="center_x",
                )
            else:
                obj_generator = self.generate_linear_gradient(
                    self.int_width,
                    self.int_height,
                    int_start_size,
                    int_end_size,
                    obj_settings.float_blur_factor,
                    str_axis="x",
                    str_gradient_style="linear",
                )
        else:
            obj_generator = self.generate_standard_grid(
                self.int_width,
                self.int_height,
                obj_settings.int_grid_size,
                obj_settings.float_blur_factor,
            )

        for (
            tuple_sample_box,
            tuple_paste_center,
            tuple_shape_dims,
            float_blur_radius,
        ) in obj_generator:
            tuple_color: tuple[int, int, int] = self.get_dominant_color(
                image_source, tuple_sample_box
            )
            image_shape: Image.Image = self.render_shape(
                tuple_color,
                tuple_shape_dims[0],
                tuple_shape_dims[1],
                bool_supersample=obj_settings.bool_supersample,
                float_blur_radius=float_blur_radius,
            )

            int_paste_x: int = int(tuple_paste_center[0] - image_shape.width // 2)
            int_paste_y: int = int(tuple_paste_center[1] - image_shape.height // 2)
            tuple_paste_loc: tuple[int, int] = (int_paste_x, int_paste_y)

            image_output.paste(image_shape, tuple_paste_loc, mask=image_shape)

        image_rgb_converted: Image.Image = image_output.convert("RGB")
        return image_rgb_converted

    def save_static_image(
        self, image_buffer: Image.Image, str_path: str, str_format: str = "png"
    ) -> None:
        """Save an image buffer to disk in the requested format."""
        try:
            image_buffer.save(str_path, format=str_format)
            logger_app.info("Saved static image to %s", str_path)
        except Exception as exc_error:
            logger_app.error("Failed to save static image. Context: %s", exc_error)
            raise RuntimeError(f"Error saving image: {exc_error}") from exc_error

    def save_video_fragment(
        self,
        image_buffer: Image.Image,
        str_path: str,
        int_num_frames: int = 1,
        int_fps: int = 30,
    ) -> None:
        """Append one image buffer to a video stream.

        Third-party API reference:
        https://docs.opencv.org/4.x/dd/d9e/classcv_1_1VideoWriter.html
        """
        if not bool_has_video_support:
            logger_app.error(
                "Video fragment save called but opencv-python, numpy, or tqdm is missing."
            )
            raise RuntimeError(
                "opencv-python, numpy, and tqdm are required for video generation."
            )

        if self._video_writer is None:
            obj_fourcc: int = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore
            tuple_size: tuple[int, int] = (image_buffer.width, image_buffer.height)
            self._video_writer = cv2.VideoWriter(  # type: ignore
                str_path,
                obj_fourcc,
                int_fps,
                tuple_size,
            )

        array_open_cv = np.array(image_buffer)
        array_bgr = array_open_cv[:, :, ::-1].copy()

        for _ in range(int_num_frames):
            self._video_writer.write(array_bgr)

    def generate_video(
        self,
        obj_start_settings: MosaicSettings,
        obj_end_settings: MosaicSettings,
        int_duration: int,
        str_output_path: str,
        int_fps: int = 30,
    ) -> None:
        """Render and encode a temporal mosaic animation.

        Behavior:
        - Interpolates settings across ``int_duration`` frames.
        - When multiple source frames exist, source-frame sampling also advances
          across the timeline.
        """
        try:
            if int_duration <= 1:
                image_buffer: Image.Image = self.render_buffer(
                    obj_start_settings, int_source_frame_index=0
                )
                self.save_video_fragment(
                    image_buffer,
                    str_output_path,
                    int_num_frames=1,
                    int_fps=int_fps,
                )
            else:
                if bool_has_video_support:
                    str_filename_short: str = os.path.basename(str_output_path)
                    if len(str_filename_short) > 30:
                        str_filename_short = str_filename_short[:27] + "..."
                    obj_range = tqdm(
                        range(int_duration),
                        desc=f"Rendering {str_filename_short}",
                        unit="frame",
                    )
                else:
                    obj_range = range(int_duration)

                for int_step in obj_range:
                    float_progress: float = int_step / float(int_duration - 1)
                    obj_current_settings: MosaicSettings = obj_start_settings.interpolate(
                        obj_end_settings,
                        float_progress,
                    )
                    int_source_frame_index: int = self._resolve_source_frame_index(
                        int_step,
                        int_duration,
                    )
                    image_buffer = self.render_buffer(
                        obj_current_settings,
                        int_source_frame_index=int_source_frame_index,
                    )
                    self.save_video_fragment(
                        image_buffer,
                        str_output_path,
                        int_num_frames=1,
                        int_fps=int_fps,
                    )
        finally:
            if self._video_writer is not None:
                self._video_writer.release()
                self._video_writer = None
                logger_app.info("\nVideo generation complete. Saved to: %s", str_output_path)

    def generate_video_from_image_buffers(
        self,
        list_bytes_frame_image_buffers: list[bytes],
        str_output_path: str,
        int_fps: int = 30,
    ) -> None:
        """Encode an MP4 directly from ordered raw image buffers.

        Inputs:
        - ``list_bytes_frame_image_buffers``: ordered image payloads.
        - ``str_output_path``: destination video file path.
        - ``int_fps``: output frames-per-second.

        Output:
        - Writes an MP4 video where each buffer becomes one frame.
        """
        if not list_bytes_frame_image_buffers:
            logger_app.error(
                "Cannot generate storyboard video because no image buffers were supplied."
            )
            raise ValueError("Image buffer list cannot be empty.")

        tuple_reference_size: tuple[int, int] | None = None
        try:
            if bool_has_video_support:
                str_filename_short: str = os.path.basename(str_output_path)
                if len(str_filename_short) > 30:
                    str_filename_short = str_filename_short[:27] + "..."
                obj_iterable_buffers = tqdm(
                    list_bytes_frame_image_buffers,
                    desc=f"Encoding {str_filename_short}",
                    unit="frame",
                )
            else:
                obj_iterable_buffers = list_bytes_frame_image_buffers

            for int_frame_index, bytes_frame_image in enumerate(obj_iterable_buffers):
                try:
                    obj_bytes_io: io.BytesIO = io.BytesIO(bytes_frame_image)
                    image_frame: Image.Image = Image.open(obj_bytes_io).convert("RGB")
                except Exception as exc_error:
                    logger_app.error(
                        "Failed to decode storyboard image buffer at frame index %d. Context: %s",
                        int_frame_index,
                        exc_error,
                    )
                    raise RuntimeError(
                        f"Failed to decode storyboard image buffer at frame index {int_frame_index}."
                    ) from exc_error

                if tuple_reference_size is None:
                    tuple_reference_size = image_frame.size
                elif image_frame.size != tuple_reference_size:
                    image_frame = image_frame.resize(
                        tuple_reference_size,
                        Image.Resampling.LANCZOS,
                    )

                self.save_video_fragment(
                    image_frame,
                    str_output_path,
                    int_num_frames=1,
                    int_fps=int_fps,
                )
        finally:
            if self._video_writer is not None:
                self._video_writer.release()
                self._video_writer = None
                logger_app.info(
                    "Storyboard video generation complete. Saved to: %s",
                    str_output_path,
                )


def main() -> None:
    """CLI entrypoint for static image and video mosaic generation."""
    import argparse

    obj_parser = argparse.ArgumentParser(description="Mosaic Image and Video Generator")

    obj_parser.add_argument(
        "input_image",
        nargs="?",
        type=str,
        help=(
            "Path to source image. Required unless --storyboard_prompt is provided."
        ),
    )
    obj_parser.add_argument(
        "--version", "-v", action="store_true", help="Print version and exit"
    )
    obj_parser.add_argument(
        "--storyboard_prompt",
        type=str,
        help=(
            "Storyboard prompt to decompose into AI-generated frames before mosaic rendering."
        ),
    )
    obj_parser.add_argument(
        "--storyboard_num_frames",
        type=int,
        default=24,
        help="Total number of output video frames in storyboard mode.",
    )
    obj_parser.add_argument(
        "--storyboard_frames_per_image",
        type=int,
        default=3,
        help="Number of output video frames to hold each generated storyboard image.",
    )

    # Global settings
    obj_parser.add_argument(
        "--mode",
        type=str,
        default="standard",
        choices=[
            "standard",
            "gradient",
            "supersample",
            "centervert",
            "centerhoriz",
            "radial",
        ],
        help="Rendering mode for the mosaic.",
    )
    obj_parser.add_argument(
        "--grid_size",
        type=int,
        default=30,
        help="Standard grid size (used if mode=standard).",
    )
    obj_parser.add_argument(
        "--blur_factor",
        type=float,
        default=0.0,
        help="Blur factor to soften mosaic edges.",
    )

    # Spatial interpolation settings (for gradients/radials)
    obj_parser.add_argument(
        "--spatial_start_size",
        type=int,
        help="Starting cell size for spatial interpolation.",
    )
    obj_parser.add_argument(
        "--spatial_end_size",
        type=int,
        help="Ending cell size for spatial interpolation.",
    )

    # Temporal interpolation settings (for video mode)
    obj_parser.add_argument(
        "--video",
        type=int,
        nargs="?",
        const=60,
        help="Generate an mp4 video with specified frames (default 60).",
    )
    obj_parser.add_argument(
        "--grid_size_temporal_end",
        type=int,
        help="Grid size at the temporal end of the video generated.",
    )
    obj_parser.add_argument(
        "--spatial_start_size_temporal_end",
        type=int,
        help="Spatial start size at the temporal end of the video generated.",
    )
    obj_parser.add_argument(
        "--spatial_end_size_temporal_end",
        type=int,
        help="Spatial end size at the temporal end of the video generated.",
    )
    obj_parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames per second for generated video output.",
    )

    obj_args = obj_parser.parse_args()

    if obj_args.version:
        str_version_text: str = f"mosaic v{get_version()} (Python {sys.version.split()[0]})"
        logger_app.info(str_version_text)
        sys.exit(0)

    str_storyboard_prompt: str = (
        obj_args.storyboard_prompt.strip()
        if obj_args.storyboard_prompt is not None
        else ""
    )
    bool_storyboard_mode: bool = len(str_storyboard_prompt) > 0

    if obj_args.video is not None and obj_args.video < 1:
        logger_app.error("Video frame count must be >= 1.")
        sys.exit(1)
    if obj_args.storyboard_num_frames < 1:
        logger_app.error("Storyboard frame count must be >= 1.")
        sys.exit(1)
    if obj_args.storyboard_frames_per_image < 1:
        logger_app.error("Storyboard frames_per_image must be >= 1.")
        sys.exit(1)
    if obj_args.fps < 1:
        logger_app.error("FPS must be >= 1.")
        sys.exit(1)
    if not bool_storyboard_mode and obj_args.input_image is None:
        logger_app.error(
            "input_image is required unless --storyboard_prompt is provided."
        )
        sys.exit(1)

    str_output_dir: str = "output"
    if not os.path.exists(str_output_dir):
        os.makedirs(str_output_dir)

    try:
        if bool_storyboard_mode:
            from .video_storyboard import VideoStoryboard

            obj_video_storyboard = VideoStoryboard(
                str_storyboard_prompt=str_storyboard_prompt,
                int_num_frames=obj_args.storyboard_num_frames,
                int_frames_per_image=obj_args.storyboard_frames_per_image,
            )
            list_bytes_storyboard_frames: list[bytes] = (
                obj_video_storyboard.generate_storyboard()
            )
            obj_mosaic_inputs: MosaicImageInputs = MosaicImageInputs(
                list_bytes_frame_image_buffers=list_bytes_storyboard_frames
            )
        else:
            obj_mosaic_inputs = MosaicImageInputs(str_input_image_path=obj_args.input_image)

        obj_mosaic: Mosaic = Mosaic(obj_mosaic_inputs)
    except Exception as exc_error:
        logger_app.error("Initialization error. Context: %s", exc_error)
        sys.exit(1)

    str_filename: str
    str_ext: str = ".png"
    if obj_args.input_image is not None:
        str_filename, str_ext = os.path.splitext(os.path.basename(obj_args.input_image))
    else:
        str_filename = "storyboard"

    # Base settings (temporal start)
    obj_base_settings: MosaicSettings = MosaicSettings(
        int_grid_size=obj_args.grid_size,
        float_blur_factor=obj_args.blur_factor,
        bool_supersample=obj_args.mode
        in ["gradient", "supersample", "centervert", "centerhoriz", "radial"],
    )

    if obj_args.mode != "standard":
        if obj_args.spatial_start_size is None or obj_args.spatial_end_size is None:
            logger_app.error(
                "%s mode requires --spatial_start_size and --spatial_end_size.",
                obj_args.mode,
            )
            sys.exit(1)

        obj_base_settings.int_spatial_interpolation_start = obj_args.spatial_start_size
        obj_base_settings.int_spatial_interpolation_end = obj_args.spatial_end_size

        if obj_args.mode == "centervert":
            obj_base_settings.str_gradient_style = "center_x"
        elif obj_args.mode == "centerhoriz":
            obj_base_settings.str_gradient_style = "center_y"
        elif obj_args.mode == "radial":
            obj_base_settings.str_gradient_style = "radial"
        elif obj_args.mode == "gradient" or obj_args.mode == "supersample":
            obj_base_settings.str_gradient_style = "linear_x"

    bool_video_mode: bool = obj_args.video is not None or bool_storyboard_mode

    # Execution mode
    if bool_video_mode:
        int_duration: int = (
            obj_args.video
            if obj_args.video is not None
            else obj_args.storyboard_num_frames
        )

        int_temp_end_grid: int = (
            obj_args.grid_size_temporal_end
            if obj_args.grid_size_temporal_end is not None
            else obj_base_settings.int_grid_size
        )
        str_temporal_tag: str = f"tempG_{obj_args.grid_size}-{int_temp_end_grid}"

        int_temp_start: int | None = None
        int_temp_end: int | None = None
        if obj_args.mode != "standard":
            int_temp_start = (
                obj_args.spatial_start_size_temporal_end
                if obj_args.spatial_start_size_temporal_end is not None
                else obj_base_settings.int_spatial_interpolation_start
            )
            int_temp_end = (
                obj_args.spatial_end_size_temporal_end
                if obj_args.spatial_end_size_temporal_end is not None
                else obj_base_settings.int_spatial_interpolation_end
            )
            str_temporal_tag += (
                f"_tempS_{obj_args.spatial_start_size}-{obj_args.spatial_end_size}"
                f"_to_{int_temp_start}-{int_temp_end}"
            )

        str_output_file: str = os.path.join(
            str_output_dir,
            (
                f"{str_filename}_mosaic_{obj_args.mode}_{int_duration}f_"
                f"{str_temporal_tag}_B{obj_args.blur_factor}.mp4"
            ),
        )

        obj_end_settings: MosaicSettings = MosaicSettings(
            int_grid_size=int_temp_end_grid,
            float_blur_factor=obj_base_settings.float_blur_factor,
            bool_supersample=obj_base_settings.bool_supersample,
            str_gradient_style=obj_base_settings.str_gradient_style,
        )

        if obj_args.mode != "standard":
            obj_end_settings.int_spatial_interpolation_start = int_temp_start
            obj_end_settings.int_spatial_interpolation_end = int_temp_end

        logger_app.info(
            "Rendering video (%d frames)... starting static/spatial rendering to target settings...",
            int_duration,
        )
        try:
            obj_mosaic.generate_video(
                obj_base_settings,
                obj_end_settings,
                int_duration,
                str_output_file,
                int_fps=obj_args.fps,
            )
        except Exception as exc_error:
            logger_app.error("Video rendering failed. Context: %s", exc_error)
            sys.exit(1)

    else:
        str_output_file = os.path.join(
            str_output_dir,
            f"{str_filename}_mosaic_static{str_ext}",
        )
        logger_app.info("Rendering static image...")
        try:
            image_rendered_buffer: Image.Image = obj_mosaic.render_buffer(obj_base_settings)
            str_fmt: str = str_ext.replace(".", "") if str_ext else "png"
            obj_mosaic.save_static_image(
                image_rendered_buffer,
                str_output_file,
                str_format=str_fmt,
            )
        except Exception as exc_error:
            logger_app.error("Rendering failed. Context: %s", exc_error)
            sys.exit(1)


if __name__ == "__main__":
    main()
