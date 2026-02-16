import sys
import os
import random
import math
import importlib.metadata
from PIL import Image, ImageDraw, ImageFilter
from typing import Generator, Tuple, Optional

def get_version() -> str:
    """Retrieve version from pyproject.toml via Poetry."""
    try:
        return importlib.metadata.version("chuckclose")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"

def safe_parse(value: str, type_func: type, arg_name: str):
    """Safely parse a string value to a specific type, exiting on failure."""
    try:
        return type_func(value)
    except ValueError:
        print(f"Error: Invalid format for {arg_name}: '{value}'")
        print(f"       Expected a valid {type_func.__name__}.")
        sys.exit(1)

# --- Core Utilities ---

def get_dominant_color(image: Image.Image, region_box: Tuple[int, int, int, int]) -> Tuple[int, int, int]:
    """Extracts the dominant color from a region of the image."""
    try:
        # Validate region
        left, top, right, bottom = region_box
        
        # Clamp to image bounds to avoid sampling "void" (black) areas outside the image
        left = max(0, left)
        top = max(0, top)
        right = min(image.width, right)
        bottom = min(image.height, bottom)

        if right <= left or bottom <= top:
            return (0, 0, 0) # Invalid/Empty region, return black
            
        cell_img = image.crop((left, top, right, bottom))
        if cell_img.width == 0 or cell_img.height == 0:
            return (0, 0, 0)

        # Quantize to 1 color to find the dominant one
        dominant_img = cell_img.quantize(colors=1)
        palette = dominant_img.getpalette()
        if palette:
            return tuple(palette[:3])
        return (0, 0, 0)
    except Exception:
        return (128, 128, 128) # Fallback gray

def render_shape(color: Tuple[int, int, int], 
                 width: int, 
                 height: int, 
                 shape_type: str = "random", 
                 supersample: bool = False, 
                 blur_radius: float = 0.0) -> Image.Image:
    """
    Renders a single shape (circle or rounded square) with given parameters.
    Returns an RGBA Image object.
    """
    if width < 1: width = 1
    if height < 1: height = 1

    # Resolve shape type
    if shape_type == "random":
        shape_type = random.choice(["square", "circle"])

    # Determine drawing dimensions
    if supersample:
        scale = 4
        draw_w = width * scale
        draw_h = height * scale
    else:
        scale = 1
        draw_w = width
        draw_h = height
    
    if draw_w < 1: draw_w = 1
    if draw_h < 1: draw_h = 1

    # Create canvas
    shape_layer = Image.new("RGBA", (draw_w, draw_h), (0, 0, 0, 0))
    d = ImageDraw.Draw(shape_layer)
    
    # Draw logic
    if shape_type == "square":
        radius = int(draw_w * 0.2)
        d.rounded_rectangle((0, 0, draw_w, draw_h), radius=radius, fill=color)
    else:
        d.ellipse((0, 0, draw_w, draw_h), fill=color)

    # Post-processing (Resize / Blur)
    if supersample:
        shape_layer = shape_layer.resize((width, height), Image.Resampling.LANCZOS)
    
    # Apply Blur to Alpha Channel
    if blur_radius > 0:
        r, g, b, a = shape_layer.split()
        # Blur radius relative to the *output* size, not supersampled size
        a_blurred = a.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        shape_layer = Image.merge("RGBA", (r, g, b, a_blurred))
        
    return shape_layer


# --- Grid Generators ---
# Each generator yields tuples: (sample_box, paste_center_xy, shape_size_wh, blur_radius)

def generate_standard_grid(width: int, height: int, grid_size: int, blur_factor: float) -> Generator:
    """Standard uniform grid generator."""
    cols = width // grid_size
    rows = height // grid_size
    
    # Pre-calculate common shape attributes
    shape_w = int(grid_size * 0.9) # Slightly smaller than grid
    blur_r = max(1, grid_size * blur_factor) if blur_factor > 0 else 0
    
    for r in range(rows):
        for c in range(cols):
            left = c * grid_size
            top = r * grid_size
            right = left + grid_size
            bottom = top + grid_size
            
            # Center of the grid cell
            center_x = left + (grid_size // 2)
            center_y = top + (grid_size // 2)
            
            yield (left, top, right, bottom), (center_x, center_y), (shape_w, shape_w), blur_r

def generate_linear_gradient(width: int, height: int, start_size: int, end_size: int, 
                           blur_factor: float, axis: str = "x", gradient_style: str = "linear") -> Generator:
    """
    Generates rectangular cells with varying sizes along an axis.
    Supports: linear-x, center-x, center-y.
    """
    # X-Axis Logic (Columns)
    if axis == "x":
        current_x = 0
        center_x = width / 2
        
        while current_x < width:
            # Calculate t
            if gradient_style == "center_x":
                t = abs(current_x - center_x) / (width / 2)
            else: # linear
                t = current_x / width
            t = min(max(t, 0.0), 1.0)
            
            current_size = start_size + (end_size - start_size) * t
            grid_s = int(round(current_size))
            if grid_s < 1: grid_s = 1
            
            blur_r = max(1, grid_s * blur_factor) if blur_factor > 0 else 0
            
            # Iterate Y (Rows) for this column
            for y in range(0, height, grid_s):
                left = current_x
                top = y
                right = left + grid_s
                bottom = top + grid_s
                
                center_x_pos = left + (grid_s // 2)
                center_y_pos = top + (grid_s // 2)
                
                # Shape size randomized slightly per cell
                size_factor = random.uniform(0.85, 0.95)
                w = h = int(grid_s * size_factor)
                
                yield (left, top, right, bottom), (center_x_pos, center_y_pos), (w, h), blur_r
            
            current_x += grid_s

    # Y-Axis Logic (Rows)
    elif axis == "y":
        current_y = 0
        center_y = height / 2
        
        while current_y < height:
            # Calculate t
            if gradient_style == "center_y":
                 t = abs(current_y - center_y) / (height / 2)
            else:
                 t = current_y / height
            t = min(max(t, 0.0), 1.0)
            
            current_size = start_size + (end_size - start_size) * t
            grid_s = int(round(current_size))
            if grid_s < 1: grid_s = 1

            blur_r = max(1, grid_s * blur_factor) if blur_factor > 0 else 0

            # Iterate X (Columns) for this row
            for x in range(0, width, grid_s):
                left = x
                top = current_y
                right = left + grid_s
                bottom = top + grid_s
                
                center_x_pos = left + (grid_s // 2)
                center_y_pos = top + (grid_s // 2)
                
                size_factor = random.uniform(0.85, 0.95)
                w = h = int(grid_s * size_factor)

                yield (left, top, right, bottom), (center_x_pos, center_y_pos), (w, h), blur_r

            current_y += grid_s

def generate_radial_grid(width: int, height: int, start_size: int, end_size: int, blur_factor: float) -> Generator:
    """
    Generates concentric ring cells.
    Uses dynamic spoke calculation to keep cells roughly square.
    """
    cx, cy = width / 2, height / 2
    max_radius = math.sqrt((width/2)**2 + (height/2)**2) * 1.05
    
    current_r = 0.0
    while current_r < max_radius:
        # Calculate grid size at this radius
        t = current_r / max_radius
        current_radial_width = start_size + (end_size - start_size) * t
        if current_radial_width < 2: current_radial_width = 2
        
        blur_r = max(1, current_radial_width * blur_factor) if blur_factor > 0 else 0

        # Dynamic Spokes Calculation
        if current_r < 1:
            num_spokes = 1
        else:
            circumference = 2 * math.pi * current_r
            num_spokes = int(circumference / current_radial_width)
        
        if num_spokes < 4: num_spokes = 4
        
        d_theta = 360 / num_spokes
        
        for i in range(num_spokes):
            theta_start = i * d_theta
            theta_end = (i + 1) * d_theta
            
            r_in = current_r
            r_out = current_r + current_radial_width
            
            # Calculate Centroid (Polar to Cartesian)
            r_c = (r_in + r_out) / 2
            theta_c = (theta_start + theta_end) / 2
            theta_c_rad = math.radians(theta_c)
            
            cent_x = cx + r_c * math.cos(theta_c_rad)
            cent_y = cy + r_c * math.sin(theta_c_rad)
            
            # Sampling Box (Approximate region for color)
            sample_half_size = max(1, int(current_radial_width / 2))
            s_left = int(cent_x - sample_half_size)
            s_top = int(cent_y - sample_half_size)
            s_right = int(cent_x + sample_half_size)
            s_bottom = int(cent_y + sample_half_size)
            
            # Check bounds to skip off-screen cells early
            if s_right < 0 or s_bottom < 0 or s_left > width or s_top > height:
                continue

            # Check drawing size
            arc_len = r_c * math.radians(d_theta)
            max_d = min(current_radial_width, arc_len)
            
            size_factor = random.uniform(0.85, 0.95)
            w = h = int(max_d * size_factor)
            
            yield (s_left, s_top, s_right, s_bottom), (cent_x, cent_y), (w, h), blur_r
            
        current_r += current_radial_width


# --- Orchestrator ---

def create_chuck_close_effect(input_path: str, output_path: str, 
                              grid_size: int = 30, 
                              blur_factor: float = 0.0, 
                              gradient: tuple[int, int] = None, 
                              supersample: bool = False, 
                              gradient_style: str = "linear_x") -> None:
    """
    Main orchestrator function.
    Reads image, selects generator, renders cells, saves output.
    """
    try:
        original = Image.open(input_path).convert("RGB")
    except Exception as e:
        print(f"Error opening image: {e}")
        return

    width, height = original.size
    print(f"Analyzing image ({width}x{height})...")
    
    # Setup Canvas
    output_img = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    
    # Select Generator
    if gradient:
        start_size, end_size = gradient
        if gradient_style == "radial":
             generator = generate_radial_grid(width, height, start_size, end_size, blur_factor)
        elif gradient_style == "center_y":
             generator = generate_linear_gradient(width, height, start_size, end_size, blur_factor, axis="y", gradient_style="center_y")
        elif gradient_style == "center_x":
             generator = generate_linear_gradient(width, height, start_size, end_size, blur_factor, axis="x", gradient_style="center_x")
        else: # linear_x default
             generator = generate_linear_gradient(width, height, start_size, end_size, blur_factor, axis="x", gradient_style="linear")
    else:
        generator = generate_standard_grid(width, height, grid_size, blur_factor)
        
    print("Rendering...")
    
    count = 0
    for sample_box, paste_center, shape_dims, blur_r in generator:
        # 1. Color
        color = get_dominant_color(original, sample_box)
        
        # 2. Render Shape
        shape_w, shape_h = shape_dims
        shape_img = render_shape(color, shape_w, shape_h, supersample=supersample, blur_radius=blur_r)
        
        # 3. Paste
        cx, cy = paste_center
        paste_x = int(cx - shape_img.width // 2)
        paste_y = int(cy - shape_img.height // 2)
        
        output_img.paste(shape_img, (paste_x, paste_y), mask=shape_img)
        count += 1
        
    output_img.convert("RGB").save(output_path)
    print(f"Success! Rendered {count} cells. Saved to: {output_path}")

def main():
    """Entry point for the console script."""
    if len(sys.argv) > 1:
        if sys.argv[1] in ["--version", "-v"]:
            print(f"chuckclose v{get_version()} (Python {sys.version.split()[0]})")
            sys.exit(0)
            
        input_file = sys.argv[1]
        
        # Check for Special Modes
        special_modes = ["gradient", "supersample", "centervert", "centerhoriz", "radial"]
        
        if len(sys.argv) > 2 and sys.argv[2] in special_modes:
             mode = sys.argv[2]
             
             # Configuration
             # All special modes use supersampling by default in current design
             # or at least the user requested high quality for gradients/radial
             is_supersample = True 
             
             gradient_style = "linear_x"
             if mode == "centervert": gradient_style = "center_x"
             if mode == "centerhoriz": gradient_style = "center_y"
             if mode == "radial": gradient_style = "radial"
             if mode == "gradient": gradient_style = "linear_x" # Explicit
             
             if len(sys.argv) < 5:
                 print(f"Error: {mode} mode requires start and end sizes.")
                 print(f"Usage: chuckclose <input> {mode} <start> <end> [blur]")
                 sys.exit(1)
             
             start_g = safe_parse(sys.argv[3], int, "start_size")
             end_g = safe_parse(sys.argv[4], int, "end_size")
             blur_f = safe_parse(sys.argv[5], float, "blur_factor") if len(sys.argv) > 5 else 0.0
             
             filename, ext = os.path.splitext(os.path.basename(input_file))
             output_dir = "output"
             if not os.path.exists(output_dir):
                 os.makedirs(output_dir)
             
             output_file = os.path.join(output_dir, f"{filename}_chuckclose_{mode}_{start_g}-{end_g}_{blur_f}{ext}")
             
             create_chuck_close_effect(input_file, output_file, blur_factor=blur_f, gradient=(start_g, end_g), supersample=is_supersample, gradient_style=gradient_style)
             
        else:
            # Standard Mode
            grid_s = 30 # Default
            blur_f = 0.0 # Default

            if len(sys.argv) > 2:
                grid_s = safe_parse(sys.argv[2], int, "grid_size")
            
            if len(sys.argv) > 3:
                blur_f = safe_parse(sys.argv[3], float, "blur_factor")
    
            filename, ext = os.path.splitext(os.path.basename(input_file))
            output_dir = "output"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            output_file = os.path.join(output_dir, f"{filename}_chuckclose_{grid_s}_{blur_f}{ext}")
            
            create_chuck_close_effect(input_file, output_file, grid_size=grid_s, blur_factor=blur_f)
    else:
        print("Usage: chuckclose <input_image> [grid_size] [blur_factor]")
        print("       chuckclose <input_image> [mode] <start> <end> [blur]")
        print("Modes: gradient, centervert, centerhoriz, radial")

if __name__ == "__main__":
    main()