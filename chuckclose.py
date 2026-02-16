import sys
import os
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
    
    # 2. Generate Density/Color Map
    # Resize the image to the size of the grid using BOX resampling.
    # This automatically calculates the average color of the pixels 
    # that form each grid square.
    color_map = original.resize((cols, rows), resample=Image.Resampling.BOX)
    color_pixels = color_map.load()

    # 3. Create Output Canvas
    # We use RGBA for layering transparency, starting with a white background.
    output_img = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    
    # 4. Create the Master Fuzzy Mask
    # The dot canvas is larger than the grid_size to allow the fuzziness to 
    # overlap slightly with neighbors, creating a smooth diffused look.
    # Tuned for more distinct circles:
    dot_radius = int(grid_size * 0.45)      # Core size ~90% of grid width
    canvas_size = int(grid_size * 1.5)      # Total area reduced to minimize wash-out
    
    # Create a white circle on black background for the mask
    mask = Image.new("L", (canvas_size, canvas_size), 0)
    draw = ImageDraw.Draw(mask)
    
    center = canvas_size // 2
    draw.ellipse(
        (center - dot_radius, center - dot_radius, center + dot_radius, center + dot_radius),
        fill=255
    )
    
    # Apply Gaussian Blur to create the diffuse/fuzzy effect
    # blur_factor controls the fuzziness relative to grid size (default 0.15)
    fuzzy_mask = mask.filter(ImageFilter.GaussianBlur(radius=grid_size * blur_factor))

    # 5. Render the Grid
    print(f"Rendering {cols * rows} grid cells...")
    
    for r in range(rows):
        for c in range(cols):
            # Get average color of this grid square
            color = color_pixels[c, r] # (R, G, B)
            
            # Create a solid "brush tip" of the target color
            brush = Image.new("RGBA", (canvas_size, canvas_size), color)
            
            # Apply the master fuzzy mask to the alpha channel of the brush
            brush.putalpha(fuzzy_mask)
            
            # Calculate position to paste
            # We center the large fuzzy dot over the grid square
            grid_center_x = (c * grid_size) + (grid_size // 2)
            grid_center_y = (r * grid_size) + (grid_size // 2)
            
            paste_x = grid_center_x - (canvas_size // 2)
            paste_y = grid_center_y - (canvas_size // 2)
            
            # Paste onto the output
            # We use the brush itself as the mask to ensure proper alpha blending
            output_img.paste(brush, (paste_x, paste_y), mask=brush)

    # 6. Save Result
    output_img.convert("RGB").save(output_path)
    print(f"Success! Saved processed image to: {output_path}")

def main():
    """Entry point for the console script."""
    if len(sys.argv) > 1:
        if sys.argv[1] in ["--version", "-v"]:
            print(f"chuckclose v{get_version()} (Python {sys.version.split()[0]})")
            sys.exit(0)
            
        input_file = sys.argv[1]
        
        # Determine output filename automatically
        # Generate default: input_name_chuckclose.ext
        filename, ext = os.path.splitext(input_file)
        output_file = f"{filename}_chuckclose{ext}"

        # Grid size is sys.argv[2]
        grid_s = int(sys.argv[2]) if len(sys.argv) > 2 else 30
        
        # Blur factor is sys.argv[3]
        blur_f = float(sys.argv[3]) if len(sys.argv) > 3 else 0.15
        
        create_chuck_close_effect(input_file, output_file, grid_s, blur_f)
    else:
        print("Usage: chuckclose <input_image> [grid_size] [blur_factor]")
        print("Example: chuckclose selfie.jpg 20 0.1")

if __name__ == "__main__":
    main()