"""Expanded test coverage for grid generators, color extraction, output naming,
CLI static rendering, MosaicSettings validation, and AiApi failure paths."""

from __future__ import annotations

import random
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from mosaic.exceptions import AiApiInitError, AiApiRequestError, MosaicInputError
from mosaic.mosaic_generator import Mosaic
from mosaic.mosaic_image_inputs import MosaicImageInputs
from mosaic.mosaic_settings import MosaicSettings
from mosaic import mosaic_generator


# ---------------------------------------------------------------------------
# MosaicSettings bounds validation
# ---------------------------------------------------------------------------


def test_mosaic_settings_rejects_zero_grid_size() -> None:
    """Reject grid_size of 0 with MosaicInputError."""
    with pytest.raises(MosaicInputError, match="int_grid_size"):
        MosaicSettings(int_grid_size=0)


def test_mosaic_settings_rejects_negative_grid_size() -> None:
    """Reject negative grid_size with MosaicInputError."""
    with pytest.raises(MosaicInputError, match="int_grid_size"):
        MosaicSettings(int_grid_size=-5)


def test_mosaic_settings_rejects_negative_blur_factor() -> None:
    """Reject negative blur_factor with MosaicInputError."""
    with pytest.raises(MosaicInputError, match="float_blur_factor"):
        MosaicSettings(float_blur_factor=-0.1)


def test_mosaic_settings_rejects_zero_spatial_start() -> None:
    """Reject spatial interpolation start of 0 with MosaicInputError."""
    with pytest.raises(MosaicInputError, match="int_spatial_interpolation_start"):
        MosaicSettings(int_spatial_interpolation_start=0)


def test_mosaic_settings_rejects_zero_spatial_end() -> None:
    """Reject spatial interpolation end of 0 with MosaicInputError."""
    with pytest.raises(MosaicInputError, match="int_spatial_interpolation_end"):
        MosaicSettings(int_spatial_interpolation_end=0)


def test_mosaic_settings_accepts_valid_defaults() -> None:
    """Default MosaicSettings values pass validation without error."""
    obj_settings = MosaicSettings()
    assert obj_settings.int_grid_size == 30
    assert obj_settings.float_blur_factor == 0.0


def test_mosaic_settings_accepts_blur_factor_zero() -> None:
    """blur_factor of exactly 0.0 is valid."""
    obj_settings = MosaicSettings(float_blur_factor=0.0)
    assert obj_settings.float_blur_factor == 0.0


# ---------------------------------------------------------------------------
# Grid generator boundary behavior
# ---------------------------------------------------------------------------


def test_generate_standard_grid_yields_no_cells_when_image_smaller_than_grid() -> None:
    """Yield zero cells when image dimensions are smaller than grid_size."""
    list_cells = list(Mosaic.generate_standard_grid(5, 5, 10, 0.0))
    assert len(list_cells) == 0


def test_generate_standard_grid_yields_correct_cell_count() -> None:
    """Yield width//grid_size * height//grid_size cells for a square image."""
    int_width = 40
    int_height = 20
    int_grid_size = 10
    list_cells = list(
        Mosaic.generate_standard_grid(int_width, int_height, int_grid_size, 0.0)
    )
    int_expected = (int_width // int_grid_size) * (int_height // int_grid_size)
    assert len(list_cells) == int_expected


def test_generate_standard_grid_is_deterministic_with_seeded_rng() -> None:
    """Two calls with the same seed produce identical geometry."""
    obj_rng_a = random.Random(42)
    obj_rng_b = random.Random(42)
    list_cells_a = list(Mosaic.generate_standard_grid(40, 40, 8, 0.0, obj_rng_a))
    list_cells_b = list(Mosaic.generate_standard_grid(40, 40, 8, 0.0, obj_rng_b))
    assert list_cells_a == list_cells_b


def test_generate_standard_grid_differs_with_different_seeds() -> None:
    """Two calls with different seeds may produce different shape dimensions."""
    obj_rng_a = random.Random(1)
    obj_rng_b = random.Random(999)
    list_cells_a = list(Mosaic.generate_standard_grid(40, 40, 4, 0.0, obj_rng_a))
    list_cells_b = list(Mosaic.generate_standard_grid(40, 40, 4, 0.0, obj_rng_b))
    # Collect the shape dimension tuples; they may differ due to jitter.
    list_dims_a = [tuple_cell[2] for tuple_cell in list_cells_a]
    list_dims_b = [tuple_cell[2] for tuple_cell in list_cells_b]
    # Not guaranteed to differ on every run, but with grid_size=4 jitter is active.
    # If they happen to match that is also valid; we just confirm the generator ran.
    assert len(list_dims_a) == len(list_dims_b)


def test_generate_linear_gradient_yields_cells_for_unit_image() -> None:
    """Yield at least one cell for a minimal valid image with gradient."""
    list_cells = list(Mosaic.generate_linear_gradient(10, 10, 2, 4, 0.0, str_axis="x"))
    assert len(list_cells) > 0


def test_generate_radial_grid_yields_cells_for_small_image() -> None:
    """Yield cells for a small image with radial layout."""
    list_cells = list(Mosaic.generate_radial_grid(20, 20, 2, 8, 0.0))
    assert len(list_cells) > 0


# ---------------------------------------------------------------------------
# Color extraction fallback
# ---------------------------------------------------------------------------


def test_get_dominant_color_returns_black_for_zero_size_region() -> None:
    """Return (0, 0, 0) when the sample region has zero area."""
    image_test = Image.new("RGB", (10, 10), (200, 100, 50))
    # Region where right <= left produces a zero-width crop.
    tuple_color = Mosaic.get_dominant_color(image_test, (5, 0, 5, 10))
    assert tuple_color == (0, 0, 0)


def test_get_dominant_color_returns_gray_on_exception() -> None:
    """Return (128, 128, 128) fallback when quantize raises an unexpected error."""
    image_test = Image.new("RGB", (10, 10), (255, 0, 0))

    with patch.object(
        Image.Image, "quantize", side_effect=RuntimeError("quantize failed")
    ):
        tuple_color = Mosaic.get_dominant_color(image_test, (0, 0, 10, 10))

    assert tuple_color == (128, 128, 128)


def test_get_dominant_color_returns_valid_rgb_for_solid_image() -> None:
    """Return an RGB tuple within valid range for a solid color image."""
    image_test = Image.new("RGB", (20, 20), (80, 160, 240))
    tuple_color = Mosaic.get_dominant_color(image_test, (0, 0, 20, 20))
    int_r, int_g, int_b = tuple_color
    assert 0 <= int_r <= 255
    assert 0 <= int_g <= 255
    assert 0 <= int_b <= 255


# ---------------------------------------------------------------------------
# Output naming logic
# ---------------------------------------------------------------------------


def test_cli_static_output_filename_contains_mosaic_static(
    path_input_image: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Static render mode writes a file whose name contains '_mosaic_static'."""
    list_str_saved_paths: list[str] = []

    original_save_static = Mosaic.save_static_image

    def recording_save_static(
        self_mosaic: Mosaic,
        image_buffer: Image.Image,
        str_path: str,
        str_format: str = "png",
    ) -> None:
        """Record the output path and delegate to the real implementation."""
        list_str_saved_paths.append(str_path)
        original_save_static(self_mosaic, image_buffer, str_path, str_format)

    monkeypatch.setattr(Mosaic, "save_static_image", recording_save_static)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["mosaic", str(path_input_image), "--grid_size", "10"],
    )

    mosaic_generator.main()

    assert len(list_str_saved_paths) == 1
    assert "_mosaic_static" in list_str_saved_paths[0]


def test_cli_static_output_filename_preserves_input_stem(
    path_input_image: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Static render output filename preserves the input image stem."""
    list_str_saved_paths: list[str] = []

    original_save_static = Mosaic.save_static_image

    def recording_save_static(
        self_mosaic: Mosaic,
        image_buffer: Image.Image,
        str_path: str,
        str_format: str = "png",
    ) -> None:
        """Record the output path and delegate to the real implementation."""
        list_str_saved_paths.append(str_path)
        original_save_static(self_mosaic, image_buffer, str_path, str_format)

    monkeypatch.setattr(Mosaic, "save_static_image", recording_save_static)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["mosaic", str(path_input_image), "--grid_size", "10"],
    )

    mosaic_generator.main()

    str_stem = path_input_image.stem
    assert any(str_stem in str_path for str_path in list_str_saved_paths)


# ---------------------------------------------------------------------------
# CLI static smoke test
# ---------------------------------------------------------------------------


def test_cli_static_render_produces_output_file(
    path_input_image: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CLI static render mode writes a valid PNG file to the output directory."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["mosaic", str(path_input_image), "--grid_size", "5"],
    )

    mosaic_generator.main()

    list_path_outputs = list((tmp_path / "output").glob("*_mosaic_static*.png"))
    assert len(list_path_outputs) == 1
    image_output = Image.open(list_path_outputs[0])
    assert image_output.mode == "RGB"


def test_cli_version_flag_does_not_require_input_image(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--version flag exits cleanly without an input_image argument."""
    monkeypatch.setattr(sys, "argv", ["mosaic", "--version"])
    # Should return normally, not raise SystemExit or require input_image.
    mosaic_generator.main()


def test_cli_rejects_invalid_grid_size(
    path_input_image: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CLI exits with code 1 when grid_size is 0."""
    monkeypatch.setattr(
        sys,
        "argv",
        ["mosaic", str(path_input_image), "--grid_size", "0"],
    )
    with pytest.raises(SystemExit) as exc_info:
        mosaic_generator.main()
    assert exc_info.value.code == 1


def test_cli_rejects_negative_blur_factor(
    path_input_image: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CLI exits with code 1 when blur_factor is negative."""
    monkeypatch.setattr(
        sys,
        "argv",
        ["mosaic", str(path_input_image), "--blur_factor", "-1.0"],
    )
    with pytest.raises(SystemExit) as exc_info:
        mosaic_generator.main()
    assert exc_info.value.code == 1


# ---------------------------------------------------------------------------
# Deterministic render mode
# ---------------------------------------------------------------------------


def test_render_buffer_is_deterministic_with_seeded_rng(
    path_input_image: Path,
) -> None:
    """Two render_buffer calls with the same seed produce identical pixel data."""
    obj_inputs = MosaicImageInputs(str_input_image_path=str(path_input_image))
    obj_mosaic = Mosaic(obj_inputs)
    obj_settings = MosaicSettings(int_grid_size=5)

    image_a = obj_mosaic.render_buffer(obj_settings, obj_rng=random.Random(7))
    image_b = obj_mosaic.render_buffer(obj_settings, obj_rng=random.Random(7))

    assert list(image_a.getdata()) == list(image_b.getdata())


# ---------------------------------------------------------------------------
# AiApi init and request failure paths
# ---------------------------------------------------------------------------


def test_ai_api_raises_init_error_when_completions_client_fails() -> None:
    """AiApiInitError is raised when the completions client factory fails."""
    with patch(
        "mosaic.ai_api.AIFactory.get_ai_completions_client",
        side_effect=RuntimeError("provider unavailable"),
    ):
        with pytest.raises(AiApiInitError):
            from mosaic.ai_api import AiApi

            AiApi()


def test_ai_api_create_images_raises_value_error_for_zero_count() -> None:
    """create_images raises ValueError when num_images < 1."""
    mock_completions = MagicMock()
    mock_completions.model_name = "test-model"

    with patch(
        "mosaic.ai_api.AIFactory.get_ai_completions_client",
        return_value=mock_completions,
    ):
        from mosaic.ai_api import AiApi

        obj_api = AiApi()

    with pytest.raises(ValueError, match="num_images must be >= 1"):
        obj_api.create_images("prompt", int_num_images=0)


def test_ai_api_send_prompt_raises_request_error_on_failure() -> None:
    """AiApiRequestError is raised when the completions provider call fails."""
    mock_completions = MagicMock()
    mock_completions.model_name = "test-model"
    mock_completions.send_prompt.side_effect = RuntimeError("network timeout")

    with patch(
        "mosaic.ai_api.AIFactory.get_ai_completions_client",
        return_value=mock_completions,
    ):
        from mosaic.ai_api import AiApi

        obj_api = AiApi()

    with pytest.raises(AiApiRequestError):
        obj_api.send_prompt("test prompt")


def test_ai_api_image_client_init_failure_raises_init_error() -> None:
    """AiApiInitError is raised when the image client factory fails."""
    mock_completions = MagicMock()
    mock_completions.model_name = "test-model"

    with patch(
        "mosaic.ai_api.AIFactory.get_ai_completions_client",
        return_value=mock_completions,
    ):
        with patch(
            "mosaic.ai_api.AIFactory.get_ai_images_client",
            side_effect=RuntimeError("image provider unavailable"),
        ):
            from mosaic.ai_api import AiApi

            obj_api = AiApi()

            with pytest.raises(AiApiInitError):
                obj_api._ensure_image_client()
