from pathlib import Path
import cv2

from cv.parse_dat import parse_dat_file
from cv.cell_extract import split_grid_into_cells, crop_cell_margin
from cv.grid_extract_outline import extract_warped_grid_from_outline

def clear_output_folder(output_dir: Path):
    """
    Create folders 1 through 9 for storing labeled digit cell crops.

    This does not delete existing files automatically, to avoid accidental data loss.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for digit in range(1, 10):
        digit_dir = output_dir / str(digit)
        digit_dir.mkdir(parents=True, exist_ok=True)


def preprocess_digit_crop(cell):
    """
    Prepare one digit cell crop before saving.

    We crop out the cell margin to remove most grid lines, then resize to a
    consistent size for later classifier training.
    """
    cropped = crop_cell_margin(cell, margin_ratio=0.18)

    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

    resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)

    return resized


def build_digit_dataset(image_dir, output_dir):
    """
    Build a labeled dataset of Sudoku digit cell crops.

    For each imageX.jpg:
        - find matching imageX.dat
        - parse ground-truth board from .dat file
        - warp the Sudoku grid
        - split it into 81 cells
        - save only cells whose label is 1 through 9

    Output format:
        data/processed/digit_cells/1/
        data/processed/digit_cells/2/
        ...
        data/processed/digit_cells/9/
    """
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)

    clear_output_folder(output_dir)

    image_paths = sorted(image_dir.glob("*.jpg"))

    total_images = 0
    successful_images = 0
    failed_images = 0
    total_saved_cells = 0

    for image_path in image_paths:
        dat_path = image_path.with_suffix(".dat")

        if not dat_path.exists():
            print(f"Skipping {image_path}: missing .dat file")
            continue

        total_images += 1

        try:
            board = parse_dat_file(dat_path)

            warped_grid, _ = extract_warped_grid_from_outline(image_path, output_size=450)
            cells = split_grid_into_cells(warped_grid)

            if len(cells) != 9 or any(len(row) != 9 for row in cells):
                raise ValueError("Expected cells to be a 9x9 nested list.")

            for r in range(9):
                for c in range(9):
                    label = board[r][c]

                    if label == 0:
                        continue

                    cell_img = cells[r][c]
                    processed_cell = preprocess_digit_crop(cell_img)

                    save_name = f"{image_path.stem}_r{r + 1}_c{c + 1}.jpg"
                    save_path = output_dir / str(label) / save_name

                    cv2.imwrite(str(save_path), processed_cell)
                    total_saved_cells += 1

            successful_images += 1

        except Exception as e:
            failed_images += 1
            print(f"Failed on {image_path}: {e}")

    print()
    print("Digit dataset build complete.")
    print(f"Images found: {len(image_paths)}")
    print(f"Images with .dat attempted: {total_images}")
    print(f"Successful images: {successful_images}")
    print(f"Failed images: {failed_images}")
    print(f"Saved nonempty digit cells: {total_saved_cells}")
    print(f"Output folder: {output_dir}")


def main():
    image_dir = "data/raw/sudoku_dataset/images"
    output_dir = "data/processed/digit_cells"

    build_digit_dataset(image_dir, output_dir)


if __name__ == "__main__":
    main()