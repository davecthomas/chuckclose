# ChuckClose Image Converter

A Python 3.13 tool that converts input images into an artistic grid of **Rounded Squares** and **Circles**. It analyzes the color of grid areas in your image and renders filled shapes to recreate the image with a modern, geometric aesthetic.

## Prerequisites

* **Python 3.13** (Required)
* **Poetry** (Recommended package manager)

## Installation

This project uses [Poetry](https://python-poetry.org/) to manage dependencies and virtual environments automatically.

1. **Clone or Download** this project folder.
2. **Open your terminal** in the project folder.
3. **Install dependencies**:
   ```bash
   poetry install

```

*(This command automatically creates a virtual environment and installs Pillow into it.)*

---

## Usage

Because the project is configured with Poetry scripts, you can run the tool directly using `poetry run chuckclose`.

### Syntax

```bash
# Standard Mode
poetry run chuckclose <input_image> [grid_size] [blur_factor]

# Gradient Mode
poetry run chuckclose <input_image> gradient <start_size> <end_size> [blur_factor]

# Supersample Mode (High Quality Gradient)
poetry run chuckclose <input_image> supersample <start_size> <end_size> [blur_factor]

# Center Gradient Modes (Symmetric)
poetry run chuckclose <input_image> centervert <center_size> <edge_size> [blur_factor]
poetry run chuckclose <input_image> centerhoriz <center_size> <edge_size> [blur_factor]

# Radial Mode (Concentric)
poetry run chuckclose <input_image> radial <center_size> <edge_size> [blur_factor]

```

### Arguments

* **`input_image`**: (Required) Path to source image (jpg, png, etc).
* **`grid_size`**: (Optional) Integer (Default: 30). Size of grid squares.
* **`blur_factor`**: (Optional) Float (Default: 0.15). Controls softness of dots.
  * *Lower (0.1)* = Sharp, distinct shapes.
  * *Higher (0.4)* = Fuzzy, blended edges.
  * *Zero (0)* = No blur (sharpest edges).

### Gradient & Center Mode Arguments
* **`gradient`**: Standard variable grid (left-to-right).
* **`supersample`**: High-quality variable grid (left-to-right).
* **`centervert`**: Symmetrical gradient. Columns change size from center to left/right edges.
* **`centerhoriz`**: Symmetrical gradient. Rows change size from center to top/bottom edges.
* **`radial`**: Concentric gradient. Rings change size from center to corners.
* **`start_size`**: Grid size at the **starting point** (Left for gradient, **Center** for center/radial modes).
* **`end_size`**: Grid size at the **ending point** (Right for gradient, **Edges/Corners** for center/radial modes).
  * *Note:* If size is 0, the original high-resolution image is shown.



### Examples

**1. Basic conversion (uses defaults):**

```bash
poetry run chuckclose portrait.jpg

```

**2. Specifying grid size:**

```bash
poetry run chuckclose landscape.png 40

```

**3. High detail (small dots):**

```bash
poetry run chuckclose dog.jpg 8

```

**4. Abstract (large dots):**

```bash
poetry run chuckclose profile.jpg 50

```

---

**5. Gradient (Variable Grid):**

```bash
poetry run chuckclose landscape.jpg gradient 0 80 0.1
```
*Transitions from full detail (left) to 80px blocks (right).*

**6. Supersample (High Quality Gradient):**

```bash
poetry run chuckclose detail.jpg supersample 0 80 0.1
```
*Same as gradient, but smoother edges at small sizes.*

**7. Center Vertical (Symmetric Columns):**

```bash
poetry run chuckclose face.jpg centervert 10 60 0.1
```
*Small detailed 10px cells in the center column, growing to 60px at left/right edges.*

**8. Center Horizontal (Symmetric Rows):**

```bash
poetry run chuckclose landscape.jpg centerhoriz 5 40 0.1
```
*Small 5px cells in the middle horizon, growing to 40px at top/bottom.*

**9. Radial Mode (Concentric):**

```bash
poetry run chuckclose eye.jpg radial 10 50 0.2
```
*Small 10px cells in the center (pupil), expanding to 50px at the corners. The grid follows a circular pattern with radial spokes.*

---

## Version Check

To verify which version of the tool and Python you are running:

```bash
poetry run chuckclose --version

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
