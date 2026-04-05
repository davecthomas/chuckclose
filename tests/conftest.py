"""Shared pytest configuration and fixtures for the mosaic test suite."""

from pathlib import Path
import sys

from PIL import Image
import pytest


REAL_VIDEO_PYTEST_OPTION: str = "--run-real-video"
REAL_VIDEO_MARKER_NAME: str = "real_video"
REAL_VIDEO_MARKER_DESCRIPTION: str = (
    "run non-mocked MP4 generation and decode integration tests"
)


path_project_root = Path(__file__).resolve().parents[1]
path_src = path_project_root / "src"
if str(path_src) not in sys.path:
    sys.path.insert(0, str(path_src))


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register custom pytest CLI options for the mosaic test suite.

    Purpose:
    - Add one explicit opt-in switch for the non-mocked MP4 generation tests so
      normal validation runs do not execute them automatically.

    Inputs:
    - `parser`: Pytest parser used to register additional command-line
      options for the current test session.

    Output:
    - Returns `None` after registering `--run-real-video`.
    """
    parser.addoption(
        REAL_VIDEO_PYTEST_OPTION,
        action="store_true",
        default=False,
        help="Run the non-mocked real MP4 generation integration tests.",
    )
    # Normal return after registering the real-video test option.
    return None


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers used by the mosaic test suite.

    Purpose:
    - Declare the `real_video` marker so pytest output and help text describe
      the non-mocked MP4 integration tests clearly.

    Inputs:
    - `config`: Active pytest configuration object for the current test
      session.

    Output:
    - Returns `None` after registering the custom marker metadata.
    """
    config.addinivalue_line(
        "markers",
        f"{REAL_VIDEO_MARKER_NAME}: {REAL_VIDEO_MARKER_DESCRIPTION}",
    )
    # Normal return after registering the real-video marker metadata.
    return None


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Deselect real-video integration tests unless the caller opts in.

    Purpose:
    - Keep the normal pytest run fast and deterministic.
    - Preserve the non-mocked MP4 tests as a first-class selectable test mode
      rather than skipping them at runtime.

    Inputs:
    - `config`: Active pytest configuration object.
    - `items`: Mutable list of collected pytest items for the session.

    Output:
    - Returns `None` after optionally deselecting items marked `real_video`.
    """
    if config.getoption(REAL_VIDEO_PYTEST_OPTION):
        # Early return because the caller explicitly requested real-video tests.
        return None

    list_real_video_items: list[pytest.Item] = []
    list_default_items: list[pytest.Item] = []
    # Partition collected items so real-video integration tests can be deselected cleanly.
    for obj_item in items:
        if obj_item.get_closest_marker(REAL_VIDEO_MARKER_NAME) is not None:
            list_real_video_items.append(obj_item)
        else:
            list_default_items.append(obj_item)

    if list_real_video_items:
        config.hook.pytest_deselected(items=list_real_video_items)
        items[:] = list_default_items
    # Normal return after applying real-video test deselection rules.
    return None


@pytest.fixture
def path_input_image(tmp_path: Path) -> Path:
    """Create a small deterministic RGB test image and return its path."""
    path_image = tmp_path / "input_image.png"
    image_input = Image.new("RGB", (20, 20), (200, 200, 200))
    # Paint the left half with a distinct color so mosaic output remains visually non-uniform in tests.
    for int_x in range(10):
        for int_y in range(20):
            image_input.putpixel((int_x, int_y), (10, 30, 80))
    image_input.save(path_image)
    # Normal return with the generated deterministic input-image fixture path.
    return path_image
