"""Shared pytest configuration and fixtures for the mosaic test suite."""

from pathlib import Path
import sys

from PIL import Image
import pytest


path_project_root = Path(__file__).resolve().parents[1]
path_src = path_project_root / "src"
if str(path_src) not in sys.path:
    sys.path.insert(0, str(path_src))


@pytest.fixture
def path_input_image(tmp_path: Path) -> Path:
    """Create a small deterministic RGB test image and return its path."""
    path_image = tmp_path / "input_image.png"
    image_input = Image.new("RGB", (20, 20), (200, 200, 200))
    for int_x in range(10):
        for int_y in range(20):
            image_input.putpixel((int_x, int_y), (10, 30, 80))
    image_input.save(path_image)
    return path_image
