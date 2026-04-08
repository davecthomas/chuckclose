# Mosaic Image and Video Generator

A Python 3.13 tool that converts input images into an artistic grid of Rounded Squares and Circles, outputting either static images or smooth video transitions between different mosaic configurations. It analyzes the color of grid areas in your image and renders filled shapes to recreate the image with a modern, geometric aesthetic.

Current Version: `1.6.1`

![Mosaic Example](output/test_chuckclose_50_0.0.png)

## Prerequisites

- **Python 3.13** (Required)
- **Poetry** (Recommended package manager)

## Installation

This project uses [Poetry](https://python-poetry.org/) to manage dependencies and virtual environments automatically.

1. **Clone or Download** this project folder.
2. **Open your terminal** in the project folder.
3. **Bootstrap the environment**:

```bash
./setup.sh
```

This script installs Python dependencies, verifies `ffmpeg`, seeds `.env` from `env_template.txt` when needed, and validates the Google Veo settings required for storyboard video mode.

4. **Or install dependencies manually**:

```bash
poetry install
```

_(This command automatically creates a virtual environment and installs Pillow, numpy, opencv-python, and the `ai-api-unified` Google video dependencies into it.)_

If you are not using Poetry, you can install the dependencies via pip using the provided `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

## Usage

Because the project is configured with Poetry scripts, you can run the tool directly using `poetry run mosaic`.

### Syntax

**Static Image Mode** (Outputs `.png` or original format)

```bash
# Standard Mode
poetry run mosaic <input_image> [--grid_size SIZE] [--blur_factor BLUR]

# Gradient & Special Modes
poetry run mosaic <input_image> --mode gradient --spatial_start_size START --spatial_end_size END [--blur_factor BLUR]
poetry run mosaic <input_image> --mode supersample --spatial_start_size START --spatial_end_size END
poetry run mosaic <input_image> --mode centervert --spatial_start_size START --spatial_end_size END
poetry run mosaic <input_image> --mode centerhoriz --spatial_start_size START --spatial_end_size END
poetry run mosaic <input_image> --mode radial --spatial_start_size START --spatial_end_size END
```

**Video Animation Mode** (Outputs `.mp4`)

```bash
# Standard Temporal Interpolation (Constant grid size across the image, zooming in/out over time)
poetry run mosaic <input_image> --video [FRAMES] --grid_size START_SIZE --grid_size_temporal_end END_SIZE

# Spatial + Temporal Interpolation (Morphing a gradient pattern over time)
poetry run mosaic <input_image> --mode radial --video [FRAMES] \
  --spatial_start_size S_START --spatial_end_size S_END \
  --spatial_start_size_temporal_end T_START --spatial_end_size_temporal_end T_END

# Storyboard prompt driven video mode (default path: Google Veo -> exact frame extraction -> mosaic)
poetry run mosaic --storyboard_prompt "An extreme closeup of a fair-skinned woman's right eye. She looks straight ahead, then left, blinks, and looks ahead again. " \
  --storyboard_num_frames 30 --storyboard_mode video --video_duration 8 --video_resolution 1080p \
  --mode radial \
  --spatial_start_size 10 --spatial_end_size 40

# Storyboard prompt mode with explicit source-video persistence
poetry run mosaic --storyboard_prompt "Extreme closeup face of a latino male, broadly smiling, wearing a cowboy hat, eyes squinting in the sunlight." \
  --storyboard_num_frames 24 --storyboard_mode video --save_source_video \
  --mode radial \
  --spatial_start_size 10 --spatial_end_size 40 --fps 24

# Force the legacy image decomposition path
poetry run mosaic --storyboard_prompt "A portrait turns slightly and smiles." \
  --storyboard_num_frames 24 --storyboard_mode image --storyboard_frames_per_image 3 \
  --mode radial \
  --spatial_start_size 10 --spatial_end_size 40
```

### Arguments

- **`input_image`**: (Optional in storyboard mode) Path to source image (jpg, png, etc). Required unless `--storyboard_prompt` is provided.
- **`--storyboard_prompt`**: (Optional) Storyboard prompt used to generate source material before mosaic rendering.
- **`--storyboard_num_frames`**: (Optional) Total final mosaic video frame count in storyboard mode. Default: 24.
- **`--storyboard_mode`**: (Optional) `video` or `image`. Default: `video`.
- **`--storyboard_frames_per_image`**: (Optional) Image-mode-only control for how many output frames reuse one generated image. Default: 3.
- **`--video_model`**: (Optional) Override the default Google Veo model for storyboard video mode.
- **`--video_duration`**: (Optional) Source-video generation duration in seconds for storyboard video mode. Default: 8.
- **`--video_aspect_ratio`**: (Optional) Source-video aspect ratio. Choices: `16:9`, `9:16`. Default: `16:9`.
- **`--video_resolution`**: (Optional) Source-video resolution. Choices: `720p`, `1080p`, `4k`. Default: `1080p`.
- **`--save_source_video`**: (Optional) Persist the intermediate source video under `output/source_videos`.
- **`--mode`**: (Optional) Rendering mode. Choices: `standard`, `gradient`, `supersample`, `centervert`, `centerhoriz`, `radial`. Default is `standard`.
- **`--grid_size`**: (Optional) Integer. Uniform grid size used in `standard` mode. Default: 30.
- **`--blur_factor`**: (Optional) Float. Controls softness of dots (0.0 = sharp, 0.4 = fuzzy).
- **`--fps`**: (Optional) Output video frames per second. Default: 30. This was added as part of the storyboard+video CLI updates, so existing commands can omit it.

#### Spatial Interpolation Arguments (For Gradient/Radial modes)

Controls how sizes interpolate _across the canvas on a single frame_.

- **`--spatial_start_size`**: Grid size at the **starting point** (Left for gradient, **Center** for center/radial modes).
- **`--spatial_end_size`**: Grid size at the **ending point** (Right for gradient, **Edges/Corners** for center/radial modes).

#### Temporal Interpolation Arguments (For Video generation)

Controls how sizes interpolate _across time from the first frame to the last frame_.

- **`--video [frames]`**: Generates an `.mp4` video. Optional frame count (default 60).
- **`--grid_size_temporal_end`**: The target `grid_size` for the _final frame_ of the video (used in `standard` mode).
- **`--spatial_start_size_temporal_end`**: The target `spatial_start_size` for the _final frame_ of the video.
- **`--spatial_end_size_temporal_end`**: The target `spatial_end_size` for the _final frame_ of the video.

_(Note: If a temporal end argument is omitted, the video will keep that parameter constant throughout the duration, matching your starting parameters)._

### Video Generation (API Only)

While the CLI generates video exports, you can also leverage the Python API directly.

```python
from mosaic import Mosaic, MosaicImageInputs, MosaicSettings

obj_mosaic_inputs = MosaicImageInputs(str_input_image_path="portrait.jpg")
mosaic = Mosaic(obj_mosaic_inputs)

start_settings = MosaicSettings(int_grid_size=10, float_blur_factor=0.0)
end_settings = MosaicSettings(int_grid_size=80, float_blur_factor=0.2)
duration_frames = 60

mosaic.generate_video(start_settings, end_settings, duration_frames, "output.mp4")
```

### Storyboard Source Generation (API Only)

You can generate storyboard source frames with either the image path or the video path, use those frames as the mosaic source sequence, and render with full mosaic settings interpolation.

```python
from ai_api_unified import AIBaseVideoProperties
from mosaic import Mosaic, MosaicImageInputs, MosaicSettings, VideoStoryboard

storyboard = VideoStoryboard(
    str_storyboard_prompt=(
        "An extreme closeup of a fair-skinned woman's right eye. "
        "She looks straight ahead, then left, blinks, and looks ahead again."
    ),
    int_num_frames=24,
)

video_properties = AIBaseVideoProperties(
    duration_seconds=8,
    aspect_ratio="16:9",
    resolution="1080p",
)

list_bytes_frames = storyboard.generate_storyboard(
    str_storyboard_mode="video",
    obj_video_properties=video_properties,
)
obj_mosaic_inputs = MosaicImageInputs(list_bytes_frame_image_buffers=list_bytes_frames)
mosaic = Mosaic(obj_mosaic_inputs)
obj_start_settings = MosaicSettings(int_grid_size=12, float_blur_factor=0.05)
obj_end_settings = MosaicSettings(int_grid_size=32, float_blur_factor=0.10)
obj_end_settings.str_gradient_style = "radial"
obj_end_settings.int_spatial_interpolation_start = 8
obj_end_settings.int_spatial_interpolation_end = 42

mosaic.generate_video(
    obj_start_settings=obj_start_settings,
    obj_end_settings=obj_end_settings,
    int_duration=24,
    str_output_path="output/storyboard_sequence_mosaic.mp4",
    int_fps=24,
)
```

### Examples

**1. Basic conversion (uses defaults):**

```bash
poetry run mosaic portrait.jpg
```

**2. Specifying grid size:**

```bash
poetry run mosaic landscape.png --grid_size 40
```

**3. High detail (small dots):**

```bash
poetry run mosaic dog.jpg --grid_size 8
```

**4. Abstract (large dots):**

```bash
poetry run mosaic profile.jpg --grid_size 50
```

---

**5. Gradient (Variable Grid):**

```bash
poetry run mosaic landscape.jpg --mode gradient --spatial_start_size 0 --spatial_end_size 80 --blur_factor 0.1
```

_Transitions from full detail (left) to 80px blocks (right)._

**6. Supersample (High Quality Gradient):**

```bash
poetry run mosaic detail.jpg --mode supersample --spatial_start_size 0 --spatial_end_size 80 --blur_factor 0.1
```

_Same as gradient, but smoother edges at small sizes._

**7. Center Vertical (Symmetric Columns):**

```bash
poetry run mosaic face.jpg --mode centervert --spatial_start_size 10 --spatial_end_size 60 --blur_factor 0.1
```

_Small detailed 10px cells in the center column, growing to 60px at left/right edges._

**8. Center Horizontal (Symmetric Rows):**

```bash
poetry run mosaic landscape.jpg --mode centerhoriz --spatial_start_size 5 --spatial_end_size 40 --blur_factor 0.1
```

_Small 5px cells in the middle horizon, growing to 40px at top/bottom._

**9. Radial Mode (Concentric):**

```bash
poetry run mosaic eye.jpg --mode radial --spatial_start_size 10 --spatial_end_size 50 --blur_factor 0.2
```

_Small 10px cells in the center (pupil), expanding to 50px at the corners. The grid follows a circular pattern with radial spokes._

---

**10. Generating a Video Animation (Temporal Morphing):**

```bash
poetry run mosaic portrait.jpg --video 60 --grid_size 1 --grid_size_temporal_end 50
```

_Generates a 60-frame `.mp4` video starting at a 1px grid (original image) and morphing into a 50px-blocked mosaic._

**11. Generating a Video Animation (Temporal + Spatial Morphing):**

```bash
poetry run mosaic test1.png --mode radial --video 90 \
  --spatial_start_size 10 --spatial_end_size 10 \
  --spatial_start_size_temporal_end 10 --spatial_end_size_temporal_end 110 \
  --blur_factor 0.0
```

_Generates a 90-frame video. Starts as a flat 10px mosaic, and animates the outer edges blooming outwards into a 110px radial gradient, while the center stays at 10px._

**12. Storyboard Prompt Video (No Input Image Path):**

```bash
poetry run mosaic --storyboard_prompt "An extreme closeup of a fair-skinned woman's right eye. She looks straight ahead, then left, blinks, and looks ahead again." \
  --storyboard_num_frames 24 --storyboard_mode video --video_duration 8 --video_resolution 1080p \
  --mode radial \
  --spatial_start_size 10 --spatial_end_size 40
```

_Generates one coherent Google Veo source video, extracts 24 clean source frames by exact frame index, and renders mosaic video frames from that sequence. Uses default `--fps 30` unless overridden._

---

## Version Check

To verify which version of the tool and Python you are running:

```bash
poetry run mosaic --version
```

## Troubleshooting

**"command not found: poetry"**
You need to install Poetry first. The recommended way is using `pipx`:

```bash
pip install pipx
pipx install poetry
```

**"Python 3.13 not found"**
Poetry tries to use the Python version specified in `pyproject.toml`. Ensure you have Python 3.13 installed on your system. You can tell Poetry to use a specific executable with:

```bash
poetry env use /path/to/python3.13
```

## Gallery

### Source Images

|         Test 1         |         Test          |
| :--------------------: | :-------------------: |
| ![Source 1](test1.png) | ![Source 2](test.png) |

### Generated Samples

#### Standard

**Grid Size 50, No Blur**  
`poetry run mosaic test1.png 50 0.0`
![Standard 50](test1_chuckclose_50_0.0.png)

#### Center Gradients

**Center Horizontal (4px to 90px)**  
`poetry run mosaic test1.png centerhoriz 4 90 0.0`
![CenterHoriz 4-90](output/test1_chuckclose_centerhoriz_4-90_0.0.png)

**Center Vertical (4px to 90px)**  
`poetry run mosaic test1.png centervert 4 90 0.0`
![CenterVert 4-90](output/test1_chuckclose_centervert_4-90_0.0.png)

**Center Horizontal (Test Image)**  
`poetry run mosaic test.png centerhoriz 4 90 0.0`
![CenterHoriz Test](output/test_chuckclose_centerhoriz_4-90_0.0.png)

#### Radial Gradients

**Radial (6px to 130px)**  
`poetry run mosaic test1.png radial 6 130 0.0`
![Radial 6-130](output/test1_chuckclose_radial_6-130_0.0.png)

**Radial (30px to 50px)**  
`poetry run mosaic test1.png radial 30 50 0.0`
![Radial 30-50](output/test1_chuckclose_radial_30-50_0.0.png)

**Radial Sample (10px to 110px)**  
`poetry run mosaic test.png radial 10 110 0.0`
![Radial Sample](output/test_chuckclose_radial_10-110_0.0.png)

#### Video Animation

**Radial Animation (Temporal + Spatial Morphing, 90 frames)**  
`poetry run mosaic test1.png --mode radial --video 90 --spatial_start_size 10 --spatial_end_size 10 --spatial_start_size_temporal_end 10 --spatial_end_size_temporal_end 110 --blur_factor 0.0`

<video src="output/test_mosaic_radial_90f_tempG_30-30_tempS_110-10_to_10-110_B0.0.mp4" controls="controls" muted="muted" style="max-width:100%; max-height:640px;"></video>
