import importlib.metadata
import logging
import math
import os
import secrets
import sys
from collections.abc import Generator
from dataclasses import dataclass

from PIL import Image, ImageDraw, ImageFilter

# Structured logging without timestamps
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
    """Retrieve version from pyproject.toml via Poetry."""
    try:
        str_version_result = importlib.metadata.version("mosaic")
        return str_version_result
    except importlib.metadata.PackageNotFoundError as exc_error:
        logger_app.warning("Package 'mosaic' not found. Using 'unknown' version. Context: %s", exc_error)
        str_unknown_version = "unknown"
        return str_unknown_version


def safe_parse_int(str_value: str, str_arg_name: str) -> int:
    """Safely parse a string value to integer, exiting on failure."""
    try:
        int_parsed_val = int(str_value)
        return int_parsed_val
    except ValueError as exc_error:
        logger_app.error("Invalid format for %s: '%s'. Expected int. Context: %s", str_arg_name, str_value, exc_error)
        sys.exit(1)


def safe_parse_float(str_value: str, str_arg_name: str) -> float:
    """Safely parse a string value to float, exiting on failure."""
    try:
        float_parsed_val = float(str_value)
        return float_parsed_val
    except ValueError as exc_error:
        logger_app.error("Invalid format for %s: '%s'. Expected float. Context: %s", str_arg_name, str_value, exc_error)
        sys.exit(1)


def lerp_float(float_a: float, float_b: float, float_progress: float) -> float:
    """Linearly interpolate two float values."""
    float_result = float_a + (float_b - float_a) * float_progress
    return float_result


def lerp_int(int_a: int, int_b: int, float_progress: float) -> int:
    """Linearly interpolate two int values."""
    float_res = int_a + (int_b - int_a) * float_progress
    int_result = int(round(float_res))
    return int_result


@dataclass
class MosaicSettings:
    """Configuration class for Mosaic generation."""
    int_grid_size: int = 30
    float_blur_factor: float = 0.0
    int_spatial_interpolation_start: int | None = None
    int_spatial_interpolation_end: int | None = None
    bool_supersample: bool = False
    str_gradient_style: str = "linear_x"

    def interpolate(self, obj_other: 'MosaicSettings', float_progress: float) -> 'MosaicSettings':
        """
        Temporally interpolates between this settings object (start of video sequence) 
        and another (end of video sequence) based on float_progress [0.0, 1.0].
        
        This cross-frame temporal interpolation tweens the spatial start and spatial end 
        settings independently.
        """
        int_interp_grid_size = lerp_int(self.int_grid_size, obj_other.int_grid_size, float_progress)
        float_interp_blur_factor = lerp_float(self.float_blur_factor, obj_other.float_blur_factor, float_progress)
        
        int_interp_spatial_start: int | None = self.int_spatial_interpolation_start
        int_interp_spatial_end: int | None = self.int_spatial_interpolation_end
        
        # Temporal interpolation of the spatial gradient dimensions
        if self.int_spatial_interpolation_start is not None and obj_other.int_spatial_interpolation_start is not None:
            int_interp_spatial_start = lerp_int(self.int_spatial_interpolation_start, obj_other.int_spatial_interpolation_start, float_progress)
            
        if self.int_spatial_interpolation_end is not None and obj_other.int_spatial_interpolation_end is not None:
            int_interp_spatial_end = lerp_int(self.int_spatial_interpolation_end, obj_other.int_spatial_interpolation_end, float_progress)
        
        bool_interp_supersample = self.bool_supersample if float_progress < 0.5 else obj_other.bool_supersample
        str_interp_gradient_style = self.str_gradient_style if float_progress < 0.5 else obj_other.str_gradient_style

        obj_new_settings = MosaicSettings(
            int_grid_size=int_interp_grid_size,
            float_blur_factor=float_interp_blur_factor,
            int_spatial_interpolation_start=int_interp_spatial_start,
            int_spatial_interpolation_end=int_interp_spatial_end,
            bool_supersample=bool_interp_supersample,
            str_gradient_style=str_interp_gradient_style
        )
        return obj_new_settings


class Mosaic:
    """Main class for generating Chuck Close style mosaics."""
    
    def __init__(self, str_input_image_path: str) -> None:
        self.str_input_path = str_input_image_path
        try:
            self.image_original = Image.open(str_input_image_path).convert("RGB")
        except Exception as exc_error:
            logger_app.error("Failed to open image file: %s. Context: %s", str_input_image_path, exc_error)
            raise RuntimeError(f"Error opening image: {exc_error}") from exc_error
            
        self.int_width, self.int_height = self.image_original.size
        self._video_writer = None
        
    @staticmethod
    def get_dominant_color(image_input: Image.Image, tuple_region_box: tuple[int, int, int, int]) -> tuple[int, int, int]:
        """Extracts the dominant color from a given box within the image."""
        try:
            int_left, int_top, int_right, int_bottom = tuple_region_box
            int_left = max(0, int_left)
            int_top = max(0, int_top)
            int_right = min(image_input.width, int_right)
            int_bottom = min(image_input.height, int_bottom)

            if int_right <= int_left or int_bottom <= int_top:
                tuple_black_color = (0, 0, 0)
                return tuple_black_color
                
            image_cell = image_input.crop((int_left, int_top, int_right, int_bottom))
            if image_cell.width == 0 or image_cell.height == 0:
                tuple_black_color = (0, 0, 0)
                return tuple_black_color

            image_dominant = image_cell.quantize(colors=1)
            list_palette = image_dominant.getpalette()
            if list_palette:
                tuple_dom_color = (list_palette[0], list_palette[1], list_palette[2])
                return tuple_dom_color
                
            tuple_black_color = (0, 0, 0)
            return tuple_black_color
        except Exception as exc_error:
            logger_app.warning("Failed to extract dominant color, defaulting to gray. Context: %s", exc_error)
            tuple_gray_color = (128, 128, 128)
            return tuple_gray_color

    @staticmethod
    def render_shape(tuple_color: tuple[int, int, int], 
                     int_width: int, 
                     int_height: int, 
                     str_shape_type: str = "random", 
                     bool_supersample: bool = False, 
                     float_blur_radius: float = 0.0) -> Image.Image:
        """Renders an individual shape (circle or square) based on given inputs."""
        int_width = max(1, int_width)
        int_height = max(1, int_height)

        if str_shape_type == "random":
            list_shape_options = ["square", "circle"]
            str_shape_type = secrets.choice(list_shape_options)

        int_scale = 4 if bool_supersample else 1
        int_draw_width = max(1, int_width * int_scale)
        int_draw_height = max(1, int_height * int_scale)

        tuple_transparent = (0, 0, 0, 0)
        image_shape_layer = Image.new("RGBA", (int_draw_width, int_draw_height), tuple_transparent)
        image_draw = ImageDraw.Draw(image_shape_layer)
        
        tuple_draw_box = (0, 0, int_draw_width, int_draw_height)
        if str_shape_type == "square":
            int_radius = int(int_draw_width * 0.2)
            image_draw.rounded_rectangle(tuple_draw_box, radius=int_radius, fill=tuple_color)
        else:
            image_draw.ellipse(tuple_draw_box, fill=tuple_color)

        if bool_supersample:
            tuple_final_size = (int_width, int_height)
            image_shape_layer = image_shape_layer.resize(tuple_final_size, Image.Resampling.LANCZOS)
        
        if float_blur_radius > 0:
            tuple_channels = image_shape_layer.split()
            channel_alpha = tuple_channels[3]
            channel_alpha_blurred = channel_alpha.filter(ImageFilter.GaussianBlur(radius=float_blur_radius))
            tuple_merged_channels = (tuple_channels[0], tuple_channels[1], tuple_channels[2], channel_alpha_blurred)
            image_shape_layer = Image.merge("RGBA", tuple_merged_channels)
            
        return image_shape_layer

    @staticmethod
    def generate_standard_grid(int_width: int, int_height: int, int_grid_size: int, float_blur_factor: float) -> Generator[tuple[tuple[int, int, int, int], tuple[int, int], tuple[int, int], float], None, None]:
        """Generator that yields coordinate properties for a standard uniform grid."""
        int_cols = int_width // int_grid_size
        int_rows = int_height // int_grid_size
        int_shape_width = int(int_grid_size * 0.9)
        float_blur_radius = max(1.0, float(int_grid_size) * float_blur_factor) if float_blur_factor > 0 else 0.0
        
        for row_idx in range(int_rows):
            for col_idx in range(int_cols):
                int_left = col_idx * int_grid_size
                int_top = row_idx * int_grid_size
                int_right = int_left + int_grid_size
                int_bottom = int_top + int_grid_size
                int_center_x = int_left + (int_grid_size // 2)
                int_center_y = int_top + (int_grid_size // 2)
                
                tuple_box = (int_left, int_top, int_right, int_bottom)
                tuple_center = (int_center_x, int_center_y)
                tuple_dims = (int_shape_width, int_shape_width)
                
                yield tuple_box, tuple_center, tuple_dims, float_blur_radius

    @staticmethod
    def generate_linear_gradient(int_width: int, int_height: int, int_start_size: int, int_end_size: int, 
                               float_blur_factor: float, str_axis: str = "x", str_gradient_style: str = "linear") -> Generator[tuple[tuple[int, int, int, int], tuple[int, int], tuple[int, int], float], None, None]:
        """Generator yielding box attributes for size-varying rectangular cells along an axis."""
        bool_is_x = (str_axis == "x")
        int_primary_dim = int_width if bool_is_x else int_height
        int_secondary_dim = int_height if bool_is_x else int_width
        float_center_pos = int_primary_dim / 2.0
        int_current_pos = 0

        while int_current_pos < int_primary_dim:
            if str_gradient_style in ("center_x", "center_y"):
                float_progress = abs(int_current_pos - float_center_pos) / (int_primary_dim / 2.0)
            else:
                float_progress = int_current_pos / float(int_primary_dim)
                
            float_progress = min(max(float_progress, 0.0), 1.0)
            
            float_calc_size = int_start_size + (int_end_size - int_start_size) * float_progress
            int_grid_size = max(1, int(round(float_calc_size)))
            float_blur_radius = max(1.0, float(int_grid_size) * float_blur_factor) if float_blur_factor > 0 else 0.0
            
            for int_sec_pos in range(0, int_secondary_dim, int_grid_size):
                int_left = int_current_pos if bool_is_x else int_sec_pos
                int_top = int_sec_pos if bool_is_x else int_current_pos
                int_right = int_left + int_grid_size
                int_bottom = int_top + int_grid_size
                
                int_center_x_pos = int_left + (int_grid_size // 2)
                int_center_y_pos = int_top + (int_grid_size // 2)
                
                float_size_factor = secrets.SystemRandom().uniform(0.85, 0.95)
                int_size_dim = int(int_grid_size * float_size_factor)
                
                tuple_box = (int_left, int_top, int_right, int_bottom)
                tuple_center = (int_center_x_pos, int_center_y_pos)
                tuple_dims = (int_size_dim, int_size_dim)
                
                yield tuple_box, tuple_center, tuple_dims, float_blur_radius
            
            int_current_pos += int_grid_size

    @staticmethod
    def generate_radial_grid(int_width: int, int_height: int, int_start_size: int, int_end_size: int, float_blur_factor: float) -> Generator[tuple[tuple[int, int, int, int], tuple[int, int], tuple[int, int], float], None, None]:
        """Generator yielding cells arranged in concentric circles."""
        float_cx, float_cy = int_width / 2.0, int_height / 2.0
        float_max_radius = math.sqrt((int_width/2.0)**2 + (int_height/2.0)**2) * 1.05
        float_current_r = 0.0
        
        while float_current_r < float_max_radius:
            float_progress = float_current_r / float_max_radius
            float_calc_width = int_start_size + (int_end_size - int_start_size) * float_progress
            float_radial_width = max(2.0, float_calc_width)
            float_blur_radius = max(1.0, float_radial_width * float_blur_factor) if float_blur_factor > 0 else 0.0
            
            if float_current_r < 1:
                int_num_spokes = 1
            else:
                float_circumference = 2.0 * math.pi * float_current_r
                int_num_spokes = max(4, int(float_circumference / float_radial_width))
            
            float_d_theta = 360.0 / int_num_spokes
            
            for index_spoke in range(int_num_spokes):
                float_theta_start = index_spoke * float_d_theta
                float_theta_end = (index_spoke + 1) * float_d_theta
                float_rc = float_current_r + (float_radial_width / 2.0)
                float_theta_c_rad = math.radians((float_theta_start + float_theta_end) / 2.0)
                
                float_cent_x = float_cx + float_rc * math.cos(float_theta_c_rad)
                float_cent_y = float_cy + float_rc * math.sin(float_theta_c_rad)
                
                int_sample_half_size = max(1, int(float_radial_width / 2.0))
                int_left = int(float_cent_x - int_sample_half_size)
                int_top = int(float_cent_y - int_sample_half_size)
                int_right = int(float_cent_x + int_sample_half_size)
                int_bottom = int(float_cent_y + int_sample_half_size)
                
                if int_right < 0 or int_bottom < 0 or int_left > int_width or int_top > int_height:
                    continue

                float_arc_len = float_rc * math.radians(float_d_theta)
                float_max_dim = min(float_radial_width, float_arc_len)
                
                float_size_factor = secrets.SystemRandom().uniform(0.85, 0.95)
                int_size_dim = int(float_max_dim * float_size_factor)
                
                tuple_box = (int_left, int_top, int_right, int_bottom)
                tuple_center = (int(float_cent_x), int(float_cent_y))
                tuple_dims = (int_size_dim, int_size_dim)
                
                yield tuple_box, tuple_center, tuple_dims, float_blur_radius
                
            float_current_r += float_radial_width

    def render_buffer(self, obj_settings: MosaicSettings) -> Image.Image:
        """Renders the mosaic based on settings and returns the RGB buffer as a Pillow Image."""
        tuple_white_color = (255, 255, 255, 255)
        tuple_size = (self.int_width, self.int_height)
        image_output = Image.new("RGBA", tuple_size, tuple_white_color)
        
        if obj_settings.int_spatial_interpolation_start is not None and obj_settings.int_spatial_interpolation_end is not None:
            int_start_size = obj_settings.int_spatial_interpolation_start
            int_end_size = obj_settings.int_spatial_interpolation_end
            if obj_settings.str_gradient_style == "radial":
                 obj_generator = self.generate_radial_grid(self.int_width, self.int_height, int_start_size, int_end_size, obj_settings.float_blur_factor)
            elif obj_settings.str_gradient_style == "center_y":
                 obj_generator = self.generate_linear_gradient(self.int_width, self.int_height, int_start_size, int_end_size, obj_settings.float_blur_factor, str_axis="y", str_gradient_style="center_y")
            elif obj_settings.str_gradient_style == "center_x":
                 obj_generator = self.generate_linear_gradient(self.int_width, self.int_height, int_start_size, int_end_size, obj_settings.float_blur_factor, str_axis="x", str_gradient_style="center_x")
            else:
                 obj_generator = self.generate_linear_gradient(self.int_width, self.int_height, int_start_size, int_end_size, obj_settings.float_blur_factor, str_axis="x", str_gradient_style="linear")
        else:
            obj_generator = self.generate_standard_grid(self.int_width, self.int_height, obj_settings.int_grid_size, obj_settings.float_blur_factor)
            
        for tuple_sample_box, tuple_paste_center, tuple_shape_dims, float_blur_radius in obj_generator:
            tuple_color = self.get_dominant_color(self.image_original, tuple_sample_box)
            image_shape = self.render_shape(tuple_color, tuple_shape_dims[0], tuple_shape_dims[1], bool_supersample=obj_settings.bool_supersample, float_blur_radius=float_blur_radius)
            
            int_paste_x = int(tuple_paste_center[0] - image_shape.width // 2)
            int_paste_y = int(tuple_paste_center[1] - image_shape.height // 2)
            tuple_paste_loc = (int_paste_x, int_paste_y)
            
            image_output.paste(image_shape, tuple_paste_loc, mask=image_shape)
            
        image_rgb_converted = image_output.convert("RGB")
        return image_rgb_converted
        
    def save_static_image(self, image_buffer: Image.Image, str_path: str, str_format: str = "png") -> None:
        """Saves a static image buffer to the specified path."""
        try:
            image_buffer.save(str_path, format=str_format)
            logger_app.info("Saved static image to %s", str_path)
        except Exception as exc_error:
            logger_app.error("Failed to save static image. Context: %s", exc_error)
            raise RuntimeError(f"Error saving image: {exc_error}") from exc_error

    def save_video_fragment(self, image_buffer: Image.Image, str_path: str, int_num_frames: int = 1, int_fps: int = 30) -> None:
        """Appends a buffer to the video file. Initializes VideoWriter on the first call."""
        if not bool_has_video_support:
            logger_app.error("Video fragment save called but opencv-python, numpy, or tqdm is missing.")
            raise RuntimeError("opencv-python, numpy, and tqdm are required for video generation.")

        if self._video_writer is None:
            obj_fourcc = cv2.VideoWriter_fourcc(*'mp4v') # type: ignore
            tuple_size = (image_buffer.width, image_buffer.height)
            self._video_writer = cv2.VideoWriter(str_path, obj_fourcc, int_fps, tuple_size) # type: ignore
            # logger_app.info("Initialized video writer for %s", str_path) # Removed to avoid cluttering tqdm

        array_open_cv = np.array(image_buffer)
        array_bgr = array_open_cv[:, :, ::-1].copy()

        for _ in range(int_num_frames):
            self._video_writer.write(array_bgr)

    def generate_video(self, obj_start_settings: MosaicSettings, obj_end_settings: MosaicSettings, int_duration: int, str_output_path: str, int_fps: int = 30) -> None:
        """Incrementally calls render_buffer and save_video_fragment, interpolating settings over the duration."""
        if int_duration <= 1:
            image_buffer = self.render_buffer(obj_start_settings)
            self.save_video_fragment(image_buffer, str_output_path, int_num_frames=1, int_fps=int_fps)
        else:
            if bool_has_video_support:
                str_filename_short = os.path.basename(str_output_path)
                if len(str_filename_short) > 30:
                    str_filename_short = str_filename_short[:27] + "..."
                obj_range = tqdm(range(int_duration), desc=f"Rendering {str_filename_short}", unit="frame")
            else:
                obj_range = range(int_duration) # Will crash cleanly in save_video_fragment immediately
                
            for int_step in obj_range:
                float_progress = int_step / float(int_duration - 1)
                obj_current_settings = obj_start_settings.interpolate(obj_end_settings, float_progress)
                image_buffer = self.render_buffer(obj_current_settings)
                self.save_video_fragment(image_buffer, str_output_path, int_num_frames=1, int_fps=int_fps)
                
        if self._video_writer is not None:
            self._video_writer.release()
            self._video_writer = None
            logger_app.info("\nVideo generation complete. Saved to: %s", str_output_path)


def main() -> None:
    """Entry point for the console script."""
    import argparse
    obj_parser = argparse.ArgumentParser(description="Mosaic Image and Video Generator")
    
    obj_parser.add_argument("input_image", type=str, help="Path to source image")
    obj_parser.add_argument("--version", "-v", action="store_true", help="Print version and exit")
    
    # Global Settings
    obj_parser.add_argument("--mode", type=str, default="standard", choices=["standard", "gradient", "supersample", "centervert", "centerhoriz", "radial"], help="Rendering mode for the mosaic.")
    obj_parser.add_argument("--grid_size", type=int, default=30, help="Standard grid size (used if mode=standard).")
    obj_parser.add_argument("--blur_factor", type=float, default=0.0, help="Blur factor to soften mosaic edges.")
    
    # Spatial Interpolation Settings (For Gradients/Radials)
    obj_parser.add_argument("--spatial_start_size", type=int, help="Starting cell size for spatial interpolation.")
    obj_parser.add_argument("--spatial_end_size", type=int, help="Ending cell size for spatial interpolation.")
    
    # Temporal Interpolation Settings (For Video Mode)
    obj_parser.add_argument("--video", type=int, nargs="?", const=60, help="Generate an mp4 video with specified frames (default 60).")
    obj_parser.add_argument("--grid_size_temporal_end", type=int, help="Grid size at the temporal end of the video generated.")
    obj_parser.add_argument("--spatial_start_size_temporal_end", type=int, help="Spatial start size at the temporal end of the video generated.")
    obj_parser.add_argument("--spatial_end_size_temporal_end", type=int, help="Spatial end size at the temporal end of the video generated.")

    obj_args = obj_parser.parse_args()

    if obj_args.version:
        str_version_text = f"mosaic v{get_version()} (Python {sys.version.split()[0]})"
        logger_app.info(str_version_text)
        sys.exit(0)

    try:
        obj_mosaic = Mosaic(obj_args.input_image)
    except Exception as exc_error:
        logger_app.error("Initialization error. Context: %s", exc_error)
        sys.exit(1)

    str_filename, str_ext = os.path.splitext(os.path.basename(obj_args.input_image))
    str_output_dir = "output"
    if not os.path.exists(str_output_dir):
        os.makedirs(str_output_dir)

    # Establish Base Settings (Temporal Start)
    obj_base_settings = MosaicSettings(
        int_grid_size=obj_args.grid_size,
        float_blur_factor=obj_args.blur_factor,
        bool_supersample=obj_args.mode in ["gradient", "supersample", "centervert", "centerhoriz", "radial"]
    )

    if obj_args.mode != "standard":
        if obj_args.spatial_start_size is None or obj_args.spatial_end_size is None:
            logger_app.error("%s mode requires --spatial_start_size and --spatial_end_size.", obj_args.mode)
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

    # Execution Mode
    if obj_args.video is not None:
        # User requested video output
        int_duration = obj_args.video
        
        # Build base name with temporal grid bounds
        int_temp_end_grid = obj_args.grid_size_temporal_end if obj_args.grid_size_temporal_end is not None else obj_base_settings.int_grid_size
        str_temporal_tag = f"tempG_{obj_args.grid_size}-{int_temp_end_grid}"
        
        if obj_args.mode != "standard":
            int_temp_start = obj_args.spatial_start_size_temporal_end if obj_args.spatial_start_size_temporal_end is not None else obj_base_settings.int_spatial_interpolation_start
            int_temp_end = obj_args.spatial_end_size_temporal_end if obj_args.spatial_end_size_temporal_end is not None else obj_base_settings.int_spatial_interpolation_end
            str_temporal_tag += f"_tempS_{obj_args.spatial_start_size}-{obj_args.spatial_end_size}_to_{int_temp_start}-{int_temp_end}"
            
        str_output_file = os.path.join(str_output_dir, f"{str_filename}_mosaic_{obj_args.mode}_{int_duration}f_{str_temporal_tag}_B{obj_args.blur_factor}.mp4")
        
        # Build Temporal End Settings
        obj_end_settings = MosaicSettings(
            int_grid_size=int_temp_end_grid,
            float_blur_factor=obj_base_settings.float_blur_factor,
            bool_supersample=obj_base_settings.bool_supersample,
            str_gradient_style=obj_base_settings.str_gradient_style
        )
        
        if obj_args.mode != "standard":
            obj_end_settings.int_spatial_interpolation_start = int_temp_start
            obj_end_settings.int_spatial_interpolation_end = int_temp_end

        logger_app.info("Rendering video (%d frames)... starting static/spatial rendering to target settings...", int_duration)
        try:
            obj_mosaic.generate_video(obj_base_settings, obj_end_settings, int_duration, str_output_file)
        except Exception as exc_error:
            logger_app.error("Video rendering failed. Context: %s", exc_error)
            sys.exit(1)

    else:
        # Static Image Rendering
        str_output_file = os.path.join(str_output_dir, f"{str_filename}_mosaic_static{str_ext}")
        logger_app.info("Rendering static image...")
        try:
            image_rendered_buffer = obj_mosaic.render_buffer(obj_base_settings)
            str_fmt = str_ext.replace(".", "") if str_ext else "png"
            obj_mosaic.save_static_image(image_rendered_buffer, str_output_file, str_format=str_fmt)
        except Exception as exc_error:
            logger_app.error("Rendering failed. Context: %s", exc_error)
            sys.exit(1)

if __name__ == "__main__":
    main()