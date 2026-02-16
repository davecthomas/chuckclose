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

def create_chuck_close_effect(input_path: str, output_path: str, grid_size: int = 30, blur_factor: float = 0.15) -> None:
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
        
        # Grid size is sys.argv[2]
        grid_s = int(sys.argv[2]) if len(sys.argv) > 2 else 30
        
        # Blur factor is sys.argv[3]
        blur_f = float(sys.argv[3]) if len(sys.argv) > 3 else 0.15

        # Determine output filename automatically
        # Generate default: input_name_chuckclose_grid_blur.ext
        filename, ext = os.path.splitext(input_file)
        output_file = f"{filename}_chuckclose_{grid_s}_{blur_f}{ext}"
        
        create_chuck_close_effect(input_file, output_file, grid_s, blur_f)
    else:
        print("Usage: chuckclose <input_image> [grid_size] [blur_factor]")
        print("Example: chuckclose selfie.jpg 20 0.1")

if __name__ == "__main__":
    main()