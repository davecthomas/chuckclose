import sys
import os
import random
import importlib.metadata
from PIL import Image, ImageDraw, ImageFilter

def get_version() -> str:
    """Retrieve version from pyproject.toml via Poetry."""
    try:
        return importlib.metadata.version("chuckclose")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"

def create_chuck_close_effect(input_path: str, output_path: str, grid_size: int = 30, blur_factor: float = 0.15, gradient: tuple[int, int] = None, supersample: bool = False) -> None:
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
        current_x = 0
        while current_x < width:
            # Calculate current grid size
            t = current_x / width
            current_grid_size = start_size + (end_size - start_size) * t
            
            # Use a threshold for "zero" or very small grid sizes to just copy original
            if current_grid_size < 4:
                # Copy a 1-pixel wide strip (or whatever the step is to advance)
                step = 1
                box = (current_x, 0, min(current_x + step, width), height)
                strip = original.crop(box)
                output_img.paste(strip, (current_x, 0))
                current_x += step
                continue
            
            # Geometric Mode
            grid_s = int(round(current_grid_size))
            if grid_s < 1: grid_s = 1 # Safety
            
            # Render a column
            cols_width = grid_s
            # Ensure we don't go out of bounds? crop handles it, but let's be clean
            
            for y in range(0, height, grid_s):
                # Coordinates
                left = current_x
                top = y
                right = left + cols_width
                bottom = top + grid_s
                
                # Crop
                # Note: crop handles out of bounds by clipping, which is what we want
                cell_img = original.crop((left, top, right, bottom))
                if cell_img.width == 0 or cell_img.height == 0: continue

                # Dominant Color
                try:
                    dominant_img = cell_img.quantize(colors=1)
                    palette = dominant_img.getpalette()
                    if palette:
                        color = tuple(palette[:3])
                    else:
                        color = (0,0,0) # Fallback
                except Exception:
                    # In case quantize fails on tiny empty image
                     color = (128,128,128)

                # --- Geometric Shape Drawing ---
                # Local canvas
                # We need actual width/height of this cell (might be clipped at edges)
                cw, ch = cell_img.size
                
                # SUPERSAMPLING LOGIC
                if supersample:
                    scale = 4
                    # Create scaled-up canvas
                    super_w, super_h = cw * scale, ch * scale
                    shape_layer = Image.new("RGBA", (super_w, super_h), (0, 0, 0, 0))
                    draw = ImageDraw.Draw(shape_layer)
                    
                    # Target size (scaled)
                    # We want the shape to take up 85-95% of the cell
                    base_s = min(super_w, super_h)
                    size_factor = random.uniform(0.85, 0.95)
                    w = h = int(base_s * size_factor)
                    
                    center_x = super_w // 2
                    center_y = super_h // 2
                    bbox = (center_x - w//2, center_y - h//2, center_x + w//2, center_y + h//2)
                    
                    shape_type = random.choice(["square", "circle"])
                    if shape_type == "square":
                        radius = int(w * 0.2)
                        draw.rounded_rectangle(bbox, radius=radius, fill=color)
                    else:
                        draw.ellipse(bbox, fill=color)
                        
                    # Downsample back to original size
                    # LANCZOS is best for downsampling (high quality antialiasing)
                    shape_layer = shape_layer.resize((cw, ch), Image.Resampling.LANCZOS)

                else:
                    # STANDARD LOGIC (No Supersampling)
                    shape_layer = Image.new("RGBA", (cw, ch), (0, 0, 0, 0))
                    draw = ImageDraw.Draw(shape_layer)
                    
                    shape_type = random.choice(["square", "circle"])
                    
                    # Size factor
                    size_factor = random.uniform(0.85, 0.95)
                    # target size based on grid_s, but constrained by actual cell size
                    base_s = min(cw, ch)
                    w = h = int(base_s * size_factor)
                    
                    center_x = cw // 2
                    center_y = ch // 2
                    
                    bbox = (center_x - w//2, center_y - h//2, center_x + w//2, center_y + h//2)
                    
                    if shape_type == "square":
                        radius = int(w * 0.2)
                        draw.rounded_rectangle(bbox, radius=radius, fill=color)
                    else:
                        draw.ellipse(bbox, fill=color)

                # Blur (Apply to the final sized layer)
                if blur_factor > 0:
                     r_ch, g_ch, b_ch, a_ch = shape_layer.split()
                     # Blur radius depends on current grid size
                     radius_val = max(1, grid_s * blur_factor)
                     a_blurred = a_ch.filter(ImageFilter.GaussianBlur(radius=radius_val))
                     shape_layer = Image.merge("RGBA", (r_ch, g_ch, b_ch, a_blurred))
                
                # Paste
                output_img.paste(shape_layer, (left, top), mask=shape_layer)
                
            current_x += cols_width

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
        
        # Check for Gradient Mode
        if len(sys.argv) > 2 and (sys.argv[2] == "gradient" or sys.argv[2] == "supersample"):
             mode = sys.argv[2]
             is_supersample = (mode == "supersample")
             
             # Usage: chuckclose input [gradient|supersample] start end [blur]
             if len(sys.argv) < 5:
                 print(f"Error: {mode} mode requires start and end sizes.")
                 print(f"Usage: chuckclose <input> {mode} <start> <end> [blur]")
                 sys.exit(1)
             
             start_g = int(sys.argv[3])
             end_g = int(sys.argv[4])
             blur_f = float(sys.argv[5]) if len(sys.argv) > 5 else 0.15
             
             filename, ext = os.path.splitext(input_file)
             mode_tag = "supersample" if is_supersample else "gradient"
             output_file = f"{filename}_chuckclose_{mode_tag}_{start_g}-{end_g}_{blur_f}{ext}"
             
             create_chuck_close_effect(input_file, output_file, blur_factor=blur_f, gradient=(start_g, end_g), supersample=is_supersample)
             
        else:
            # Standard Mode
            # Grid size is sys.argv[2]
            grid_s = int(sys.argv[2]) if len(sys.argv) > 2 else 30
            
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