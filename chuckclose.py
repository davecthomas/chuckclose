import sys
import os
import random
import math
import importlib.metadata
from PIL import Image, ImageDraw, ImageFilter

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

def create_chuck_close_effect(input_path: str, output_path: str, grid_size: int = 30, blur_factor: float = 0.15, gradient: tuple[int, int] = None, supersample: bool = False, gradient_style: str = "linear_x") -> None:
    """
    Converts an image into a grid of fuzzy diffuse dots (Chuck Close style).
    
    Args:
        input_path (str): Path to source image.
        output_path (str): Path to save the result.
        grid_size (int): Size of the square grid area in pixels.
    """
    try:
        # Open image and ensure RGB to capture color (or grayscale as R=G=B)
        original = Image.open(input_path).convert("RGB")
    except FileNotFoundError:
        print(f"Error: Could not find file '{input_path}'")
        return
    except Exception as e:
        print(f"Error opening image: {e}")
        return

    width, height = original.size
    
    # 1. Calculate Grid Dimensions
    cols = width // grid_size
    rows = height // grid_size
    
    print(f"Analyzing image ({width}x{height}) with grid size {grid_size}...")
    
    # 2. Setup Output Canvas
    # We use RGBA for layering transparency, starting with a white background.
    output_img = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    
    # 3. Render the Grid
    # 3. Render the Grid (or Gradient Columns)
    print(f"Rendering...")

    if gradient:
        start_size, end_size = gradient
        
        # Define the loop axis
        # linear_x, center_x -> vary current_x (columns)
        # center_y -> vary current_y (rows)
        
        if gradient_style == "center_y":
            vary_axis = "y"
        elif gradient_style == "radial":
            vary_axis = "radial"
        else:
             vary_axis = "x"
        
        if vary_axis == "radial":
            # RADIAL LOOP (Concentric)
            cx, cy = width / 2, height / 2
            max_radius = math.sqrt((width/2)**2 + (height/2)**2) * 1.05 # Go slightly past corners
            
            # Heuristic for angular steps (Spokes) logic moved inside loop for dynamic density
            
            current_r = 0.0
            while current_r < max_radius:
                # Calculate current radial width
                t = current_r / max_radius
                current_radial_width = start_size + (end_size - start_size) * t
                
                # Check min size
                if current_radial_width < 2: current_radial_width = 2
                
                # DYNAMIC SPOKES CALCULATION
                # We want arc_length ~= current_radial_width to keep cells roughly square
                # dist = 2 * pi * r
                # n = dist / width
                if current_r < 1:
                     num_spokes = 1
                else:
                     circumference = 2 * math.pi * current_r
                     num_spokes = int(circumference / current_radial_width)
                
                if num_spokes < 4: num_spokes = 4
                
                d_theta = 360 / num_spokes
                
                # Loop Angles
                for i in range(num_spokes):
                    theta_start = i * d_theta
                    theta_end = (i + 1) * d_theta
                    
                    # Define Region (Annular Sector)
                    r_in = current_r
                    r_out = current_r + current_radial_width
                    
                    # 1. Extract Color
                    r_c = (r_in + r_out) / 2
                    theta_c = (theta_start + theta_end) / 2
                    theta_c_rad = math.radians(theta_c)
                    
                    cent_x = cx + r_c * math.cos(theta_c_rad)
                    cent_y = cy + r_c * math.sin(theta_c_rad)
                    
                    # Box size for color sampling (approx size)
                    sample_size = max(1, int(current_radial_width / 2))
                    left = int(cent_x - sample_size)
                    top = int(cent_y - sample_size)
                    right = int(cent_x + sample_size)
                    bottom = int(cent_y + sample_size)
                    
                    # Clamp
                    left = max(0, left); top = max(0, top)
                    right = min(width, right); bottom = min(height, bottom)
                    
                    if right <= left or bottom <= top:
                        continue # Off screen
                        
                    cell_img = original.crop((left, top, right, bottom))
                    
                    try:
                        dominant_img = cell_img.quantize(colors=1)
                        palette = dominant_img.getpalette()
                        if palette:
                            color = tuple(palette[:3])
                        else:
                            color = (0,0,0)
                    except Exception:
                        color = (128,128,128)
                        
                    # 2. Draw Shape
                    # Size?
                    # Approximate arc length at this radius
                    arc_len = r_c * math.radians(d_theta)
                    # Fit in the smaller dimension (radial width vs arc length)
                    max_d = min(current_radial_width, arc_len)

                    bbox_w = bbox_h = max_d * random.uniform(0.85, 0.95)
                    
                    draw_x = cent_x
                    draw_y = cent_y
                    
                    # Select Shape Type
                    shape_type = random.choice(["square", "circle"])
                    
                    # Drawing
                    if supersample:
                         scale = 4
                         s_w = int(bbox_w * scale)
                         s_h = int(bbox_h * scale)
                         if s_w < 1: s_w = 1
                         if s_h < 1: s_h = 1
                         
                         shape_layer = Image.new("RGBA", (s_w, s_h), (0, 0, 0, 0))
                         d = ImageDraw.Draw(shape_layer)
                         
                         if shape_type == "square":
                             radius = int(s_w * 0.2)
                             d.rounded_rectangle((0, 0, s_w, s_h), radius=radius, fill=color)
                         else:
                             d.ellipse((0, 0, s_w, s_h), fill=color)
                         
                         # Resize down
                         final_w = int(bbox_w)
                         final_h = int(bbox_h)
                         if final_w < 1: final_w = 1
                         if final_h < 1: final_h = 1
                         
                         shape_layer = shape_layer.resize((final_w, final_h), Image.Resampling.LANCZOS)
                         
                    else:
                         s_w = int(bbox_w)
                         s_h = int(bbox_h)
                         if s_w < 1: s_w = 1
                         if s_h < 1: s_h = 1
                         shape_layer = Image.new("RGBA", (s_w, s_h), (0, 0, 0, 0))
                         d = ImageDraw.Draw(shape_layer)
                         
                         if shape_type == "square":
                             radius = int(s_w * 0.2)
                             d.rounded_rectangle((0, 0, s_w, s_h), radius=radius, fill=color)
                         else:
                             d.ellipse((0, 0, s_w, s_h), fill=color)

                    # Blur
                    if blur_factor > 0:
                         r_ch, g_ch, b_ch, a_ch = shape_layer.split()
                         radius_val = max(1, current_radial_width * blur_factor)
                         a_blurred = a_ch.filter(ImageFilter.GaussianBlur(radius=radius_val))
                         shape_layer = Image.merge("RGBA", (r_ch, g_ch, b_ch, a_blurred))
                    
                    # Paste centered at cent_x, cent_y
                    p_x = int(draw_x - shape_layer.width // 2)
                    p_y = int(draw_y - shape_layer.height // 2)
                    
                    output_img.paste(shape_layer, (p_x, p_y), mask=shape_layer)

                current_r += current_radial_width

        elif vary_axis == "x":
            # COLUMN-BASED LOOP
            current_x = 0
            center_x = width / 2
            
            while current_x < width:
                # Calculate t based on style
                if gradient_style == "center_x":
                    # Center out: 0 at center, 1 at edges
                    # We use distance from center center_x
                    # Normalize by half width
                    dist = abs((current_x + (start_size if start_size > 0 else 1)/2) - center_x) 
                    # Actually just use current_x vs center
                    dist = abs(current_x - center_x)
                    t = dist / (width / 2)
                    # Clamp t?
                    t = min(max(t, 0.0), 1.0)
                else:
                    # linear_x
                    t = current_x / width
                
                current_grid_size = start_size + (end_size - start_size) * t
                
                # Use a threshold for "zero" or very small grid sizes to just copy original
                if current_grid_size < 4:
                    step = 1
                    box = (current_x, 0, min(current_x + step, width), height)
                    strip = original.crop(box)
                    output_img.paste(strip, (current_x, 0))
                    current_x += step
                    continue
                
                # Geometric Mode
                grid_s = int(round(current_grid_size))
                if grid_s < 1: grid_s = 1
                
                cols_width = grid_s
                
                for y in range(0, height, grid_s):
                    left = current_x
                    top = y
                    right = left + cols_width
                    bottom = top + grid_s
                    # ... Render Cell (Extracted to helper or inline)? 
                    # Inline for now to minimize diff risk, using existing logic copy
                    
                    cell_img = original.crop((left, top, right, bottom))
                    if cell_img.width == 0 or cell_img.height == 0: continue

                    # Dominant Color
                    try:
                        dominant_img = cell_img.quantize(colors=1)
                        palette = dominant_img.getpalette()
                        if palette:
                            color = tuple(palette[:3])
                        else:
                            color = (0,0,0)
                    except Exception:
                         color = (128,128,128)

                    cw, ch = cell_img.size
                    
                    # Drawing Logic
                    if supersample:
                        scale = 4
                        super_w, super_h = cw * scale, ch * scale
                        shape_layer = Image.new("RGBA", (super_w, super_h), (0, 0, 0, 0))
                        draw = ImageDraw.Draw(shape_layer)
                        
                        base_s = min(super_w, super_h)
                        size_factor = random.uniform(0.85, 0.95)
                        w = h = int(base_s * size_factor)
                        
                        center_x_draw = super_w // 2
                        center_y_draw = super_h // 2
                        bbox = (center_x_draw - w//2, center_y_draw - h//2, center_x_draw + w//2, center_y_draw + h//2)
                        
                        shape_type = random.choice(["square", "circle"])
                        if shape_type == "square":
                            radius = int(w * 0.2)
                            draw.rounded_rectangle(bbox, radius=radius, fill=color)
                        else:
                            draw.ellipse(bbox, fill=color)
                            
                        shape_layer = shape_layer.resize((cw, ch), Image.Resampling.LANCZOS)
                    else:
                        shape_layer = Image.new("RGBA", (cw, ch), (0, 0, 0, 0))
                        draw = ImageDraw.Draw(shape_layer)
                        shape_type = random.choice(["square", "circle"])
                        size_factor = random.uniform(0.85, 0.95)
                        base_s = min(cw, ch)
                        w = h = int(base_s * size_factor)
                        center_x_draw = cw // 2
                        center_y_draw = ch // 2
                        bbox = (center_x_draw - w//2, center_y_draw - h//2, center_x_draw + w//2, center_y_draw + h//2)
                        if shape_type == "square":
                            radius = int(w * 0.2)
                            draw.rounded_rectangle(bbox, radius=radius, fill=color)
                        else:
                            draw.ellipse(bbox, fill=color)

                    if blur_factor > 0:
                         r_ch, g_ch, b_ch, a_ch = shape_layer.split()
                         radius_val = max(1, grid_s * blur_factor)
                         a_blurred = a_ch.filter(ImageFilter.GaussianBlur(radius=radius_val))
                         shape_layer = Image.merge("RGBA", (r_ch, g_ch, b_ch, a_blurred))
                    
                    output_img.paste(shape_layer, (left, top), mask=shape_layer)
                    
                current_x += cols_width

        else:
            # ROW-BASED LOOP (center_y)
            current_y = 0
            center_y = height / 2
            
            while current_y < height:
                # Calculate t based on center_y
                dist = abs(current_y - center_y)
                t = dist / (height / 2)
                t = min(max(t, 0.0), 1.0)
                
                current_grid_size = start_size + (end_size - start_size) * t
                
                if current_grid_size < 4:
                    step = 1
                    box = (0, current_y, width, min(current_y + step, height))
                    strip = original.crop(box)
                    output_img.paste(strip, (0, current_y))
                    current_y += step
                    continue
                
                grid_s = int(round(current_grid_size))
                if grid_s < 1: grid_s = 1
                
                rows_height = grid_s
                
                # Iterate columns for this row
                for x in range(0, width, grid_s):
                    left = x
                    top = current_y
                    right = left + grid_s
                    bottom = top + rows_height
                    
                    cell_img = original.crop((left, top, right, bottom))
                    if cell_img.width == 0 or cell_img.height == 0: continue

                    # Dominant Color
                    try:
                        dominant_img = cell_img.quantize(colors=1)
                        palette = dominant_img.getpalette()
                        if palette:
                            color = tuple(palette[:3])
                        else:
                            color = (0,0,0)
                    except Exception:
                         color = (128,128,128)

                    cw, ch = cell_img.size
                    
                    # Drawing Logic (Same as above, forcing supersample usually for center gradients)
                    # Note: User request implied supersampling for center gradients
                    
                    if supersample:
                        scale = 4
                        super_w, super_h = cw * scale, ch * scale
                        shape_layer = Image.new("RGBA", (super_w, super_h), (0, 0, 0, 0))
                        draw = ImageDraw.Draw(shape_layer)
                        base_s = min(super_w, super_h)
                        size_factor = random.uniform(0.85, 0.95)
                        w = h = int(base_s * size_factor)
                        center_x_draw = super_w // 2
                        center_y_draw = super_h // 2
                        bbox = (center_x_draw - w//2, center_y_draw - h//2, center_x_draw + w//2, center_y_draw + h//2)
                        shape_type = random.choice(["square", "circle"])
                        if shape_type == "square":
                            radius = int(w * 0.2)
                            draw.rounded_rectangle(bbox, radius=radius, fill=color)
                        else:
                            draw.ellipse(bbox, fill=color)
                        shape_layer = shape_layer.resize((cw, ch), Image.Resampling.LANCZOS)
                    else:
                        # Fallback if someone somehow calls without supersample, though CLI enforces
                        shape_layer = Image.new("RGBA", (cw, ch), (0, 0, 0, 0))
                        draw = ImageDraw.Draw(shape_layer)
                        shape_type = random.choice(["square", "circle"])
                        size_factor = random.uniform(0.85, 0.95)
                        base_s = min(cw, ch)
                        w = h = int(base_s * size_factor)
                        center_x_draw = cw // 2
                        center_y_draw = ch // 2
                        bbox = (center_x_draw - w//2, center_y_draw - h//2, center_x_draw + w//2, center_y_draw + h//2)
                        if shape_type == "square":
                            radius = int(w * 0.2)
                            draw.rounded_rectangle(bbox, radius=radius, fill=color)
                        else:
                            draw.ellipse(bbox, fill=color)

                    if blur_factor > 0:
                         r_ch, g_ch, b_ch, a_ch = shape_layer.split()
                         radius_val = max(1, grid_s * blur_factor)
                         a_blurred = a_ch.filter(ImageFilter.GaussianBlur(radius=radius_val))
                         shape_layer = Image.merge("RGBA", (r_ch, g_ch, b_ch, a_blurred))
                    
                    output_img.paste(shape_layer, (left, top), mask=shape_layer)
                
                current_y += rows_height

    else:
        # Standard Fixed Grid Logic
        cols = width // grid_size
        rows = height // grid_size
        print(f"Rendering {cols * rows} grid cells...")
        
        for r in range(rows):
            for c in range(cols):
                # Calculate grid cell coordinates
                left = c * grid_size
                top = r * grid_size
                right = left + grid_size
                bottom = top + grid_size
                
                # Crop the cell from original
                cell_img = original.crop((left, top, right, bottom))
                
                # Find Dominant Color
                # Quantize to 1 color (this finds the centroid of the main cluster)
                # This effectively gives the "dominant" visual color
                dominant_img = cell_img.quantize(colors=1)
                # Get palette (r, g, b) from the single color index 0
                palette = dominant_img.getpalette()[:3]
                color = tuple(palette)
                
                # --- Rendering Logic ---
                # GEOMETRIC SHAPE MODE (Default)
                # Create a localized canvas for this shape
                # We make it slightly larger for anti-aliasing safety, but no rotation needed now
                shape_size = int(grid_size * 1.5)
                center = shape_size // 2
                
                # Transparent layer for the shape
                shape_layer = Image.new("RGBA", (shape_size, shape_size), (0, 0, 0, 0))
                draw = ImageDraw.Draw(shape_layer)
                
                # Randomize Shape Attributes: Rounded Square or Circle
                shape_type = random.choice(["square", "circle"])
                
                # Dimensions - Tighter fit!
                # Use 85-95% of the grid size
                size_factor = random.uniform(0.85, 0.95)
                w = h = int(grid_size * size_factor)
                
                # Bounding box centered
                bbox = (center - w//2, center - h//2, center + w//2, center + h//2)
                
                # Drawing Logic - Filled Only
                if shape_type == "square":
                    # Rounded corners - radius about 20% of size
                    radius = int(w * 0.2)
                    draw.rounded_rectangle(bbox, radius=radius, fill=color)
                else: # circle
                    draw.ellipse(bbox, fill=color)
                
                # No Rotation needed
                
                # Apply Blur
                # We blur the alpha channel of the shape itself to soften edges based on blur_factor
                if blur_factor > 0:
                     # Extract alpha
                     r_ch, g_ch, b_ch, a_ch = shape_layer.split()
                     # Blur alpha
                     a_blurred = a_ch.filter(ImageFilter.GaussianBlur(radius=grid_size * blur_factor))
                     # Merge back
                     shape_layer = Image.merge("RGBA", (r_ch, g_ch, b_ch, a_blurred))

                # Paste onto output
                # Calculate global position
                grid_center_x = (c * grid_size) + (grid_size // 2)
                grid_center_y = (r * grid_size) + (grid_size // 2)
                
                paste_x = grid_center_x - (shape_size // 2)
                paste_y = grid_center_y - (shape_size // 2)
                
                output_img.paste(shape_layer, (paste_x, paste_y), mask=shape_layer)

    # 5. Save Result
    output_img.convert("RGB").save(output_path)
    print(f"Success! Saved processed image to: {output_path}")

def main():
    """Entry point for the console script."""
    if len(sys.argv) > 1:
        if sys.argv[1] in ["--version", "-v"]:
            print(f"chuckclose v{get_version()} (Python {sys.version.split()[0]})")
            sys.exit(0)
            
        input_file = sys.argv[1]
        
        # Check for Special Modes
        if len(sys.argv) > 2 and sys.argv[2] in ["gradient", "supersample", "centervert", "centerhoriz", "radial"]:
             mode = sys.argv[2]
             
             # Configuration based on mode
             is_supersample = mode in ["supersample", "centervert", "centerhoriz", "radial"]
             gradient_style = "linear_x"
             if mode == "centervert": gradient_style = "center_x"
             if mode == "centerhoriz": gradient_style = "center_y"
             if mode == "radial": gradient_style = "radial"
             
             # Usage: chuckclose input [mode] start end [blur]
             if len(sys.argv) < 5:
                 print(f"Error: {mode} mode requires start and end sizes.")
                 print(f"Usage: chuckclose <input> {mode} <start> <end> [blur]")
                 sys.exit(1)
             
             start_g = int(sys.argv[3])
             end_g = int(sys.argv[4])
             blur_f = float(sys.argv[5]) if len(sys.argv) > 5 else 0.15
             
             filename, ext = os.path.splitext(input_file)
             # Map mode to filename tag
             mode_tag = mode # directly use mode name
             output_file = f"{filename}_chuckclose_{mode_tag}_{start_g}-{end_g}_{blur_f}{ext}"
             
             create_chuck_close_effect(input_file, output_file, blur_factor=blur_f, gradient=(start_g, end_g), supersample=is_supersample, gradient_style=gradient_style)
             
        else:
            # Standard Mode
            try:
                grid_s = int(sys.argv[2])
            except ValueError:
                print(f"Error: Invalid argument '{sys.argv[2]}'. Expected an integer grid size or a valid mode (gradient, supersample, centervert, centerhoriz, radial).")
                sys.exit(1)
            
            # Blur factor is sys.argv[3]
            blur_f = float(sys.argv[3]) if len(sys.argv) > 3 else 0.15
    
            # Determine output filename automatically
            # Generate default: input_name_chuckclose_grid_blur.ext
            filename, ext = os.path.splitext(input_file)
            output_file = f"{filename}_chuckclose_{grid_s}_{blur_f}{ext}"
            
            create_chuck_close_effect(input_file, output_file, grid_size=grid_s, blur_factor=blur_f)
    else:
        print("Usage: chuckclose <input_image> [grid_size] [blur_factor]")
        print("Example: chuckclose selfie.jpg 20 0.1")

if __name__ == "__main__":
    main()