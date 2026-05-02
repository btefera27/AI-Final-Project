"""

Utility script for visually inspecting the extracted Sudoku digit dataset.

This script is part of the computer vision pipeline for the Sudoku solver
project. It does not train a model and does not perform prediction. Instead,
it creates contact-sheet preview images for the digit crops stored in:

    data/processed/digit_cells/

The digit dataset is expected to have the following structure:

    data/processed/digit_cells/
    ├── 1/
    ├── 2/
    ├── 3/
    ├── 4/
    ├── 5/
    ├── 6/
    ├── 7/
    ├── 8/
    └── 9/

Each folder contains cropped cell images labeled with the corresponding digit.
For example:

    data/processed/digit_cells/7/image30_r4_c6.jpg

means the crop came from image30, row 4, column 6, and the ground-truth label
from the .dat file was digit 7.

The purpose of this script is to create larger preview grids such as:

    data/processed/digit_previews/digit_1_preview.jpg
    data/processed/digit_previews/digit_2_preview.jpg
    ...
    data/processed/digit_previews/digit_9_preview.jpg

These preview sheets are useful because individual digit crops are only 28x28
pixels and are difficult to inspect manually. The preview sheet enlarges each
crop and arranges multiple examples into a grid.

Why this matters:
    A digit classifier can only perform well if its training crops are mostly
    clean and correctly labeled. Before training or debugging the SVM digit
    classifier, we should visually inspect these previews to catch common
    problems such as:
        - blank crops saved under a digit label,
        - crops shifted into the wrong cell,
        - severe grid-line artifacts,
        - mislabeled-looking digits caused by bad warping,
        - blurry or unreadable digits.

This file is intentionally simple and safe:
    - It does not modify the original digit crops.
    - It only reads existing images and writes preview sheets.
    - It can be rerun any time after rebuilding digit_cells/.
"""

from pathlib import Path
import cv2
import numpy as np


def make_contact_sheet(digit_dir, output_path, max_images=50, cell_size=80):
    """
    Create one contact-sheet image for a single digit class.

    The function reads up to max_images crops from one digit folder, enlarges
    each crop for readability, and arranges them into a grid. The resulting
    preview image is saved to output_path.

    Parameters
    ----------
    digit_dir : str or pathlib.Path
        Folder containing cropped images for one digit class.

        Example:
            data/processed/digit_cells/7/

    output_path : str or pathlib.Path
        File path where the preview contact sheet should be saved.

        Example:
            data/processed/digit_previews/digit_7_preview.jpg

    max_images : int, default=50
        Maximum number of digit crops to include in the preview sheet.
        Keeping this limited makes the preview easy to inspect quickly.

    cell_size : int, default=80
        Size, in pixels, of each enlarged crop in the preview sheet.
        The original classifier crops are usually 28x28, so enlarging them
        makes visual inspection much easier.

    Returns
    -------
    None
        The function writes the preview image to disk and prints the saved path.
    """
    digit_dir = Path(digit_dir)
    output_path = Path(output_path)

    # Ensure the output folder exists before trying to save the preview image.
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Use a deterministic order so repeated previews are stable and comparable.
    image_paths = sorted(digit_dir.glob("*.jpg"))[:max_images]

    if not image_paths:
        print(f"No images found in {digit_dir}")
        return

    processed = []

    for path in image_paths:
        # Load the crop as grayscale because digit crops are single-channel
        # features for the classifier.
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

        # Skip unreadable images instead of crashing the whole preview script.
        if img is None:
            continue

        # Enlarge the crop for human inspection.
        # INTER_NEAREST preserves the blocky pixel structure, which is useful
        # for seeing exactly what the classifier sees.
        img = cv2.resize(img, (cell_size, cell_size), interpolation=cv2.INTER_NEAREST)

        # Convert grayscale to BGR so all preview tiles have 3 channels.
        # This makes horizontal/vertical stacking consistent.
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        processed.append(img)

    # Arrange the preview as 10 columns. With max_images=50, this gives
    # at most 5 rows, which is compact and easy to scan visually.
    cols = 10
    rows = int(np.ceil(len(processed) / cols))

    # Add blank tiles if the final row is not full.
    blank = np.zeros((cell_size, cell_size, 3), dtype=np.uint8)

    while len(processed) < rows * cols:
        processed.append(blank.copy())

    row_imgs = []

    for r in range(rows):
        row = processed[r * cols:(r + 1) * cols]
        row_imgs.append(np.hstack(row))

    # Stack all rows into one preview sheet and save it.
    sheet = np.vstack(row_imgs)
    cv2.imwrite(str(output_path), sheet)

    print(f"Saved preview: {output_path}")


def main():
    """
    Generate preview sheets for all digit classes 1 through 9.

    Expected input:
        data/processed/digit_cells/1/
        ...
        data/processed/digit_cells/9/

    Generated output:
        data/processed/digit_previews/digit_1_preview.jpg
        ...
        data/processed/digit_previews/digit_9_preview.jpg

    Run from the project root using:

        python -m cv.preview_digit_dataset
    """
    base_dir = Path("data/processed/digit_cells")
    preview_dir = Path("data/processed/digit_previews")

    for digit in range(1, 10):
        make_contact_sheet(
            digit_dir=base_dir / str(digit),
            output_path=preview_dir / f"digit_{digit}_preview.jpg",
        )


if __name__ == "__main__":
    main()