from pathlib import Path
import cv2

from cv.grid_extract import extract_warped_grid


def split_grid_into_cells(warped_grid, grid_size=9):
    """
    Split a square warped Sudoku image into 81 cells.

    Returns:
        cells: 9x9 list of cell images
    """
    height, width = warped_grid.shape[:2]

    if height != width:
        raise ValueError(f"Warped grid must be square, got {width}x{height}")

    cell_size = height // grid_size

    cells = []

    for row in range(grid_size):
        row_cells = []

        for col in range(grid_size):
            y1 = row * cell_size
            y2 = (row + 1) * cell_size
            x1 = col * cell_size
            x2 = (col + 1) * cell_size

            cell = warped_grid[y1:y2, x1:x2]
            row_cells.append(cell)

        cells.append(row_cells)

    return cells


def crop_cell_margin(cell, margin_ratio=0.15):
    """
    Remove grid-line borders from a cell.
    """
    h, w = cell.shape[:2]

    margin_y = int(h * margin_ratio)
    margin_x = int(w * margin_ratio)

    return cell[margin_y:h - margin_y, margin_x:w - margin_x]


def save_cell_debug(image_path, output_dir="data/processed/debug_cells"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    warped, _, _ = extract_warped_grid(image_path, output_size=450)

    cells = split_grid_into_cells(warped)

    for row in range(9):
        for col in range(9):
            cell = cells[row][col]
            cropped = crop_cell_margin(cell)

            filename = f"cell_r{row + 1}_c{col + 1}.jpg"
            cv2.imwrite(str(output_dir / filename), cropped)

    print(f"Saved 81 cell images to: {output_dir.resolve()}")


if __name__ == "__main__":
    image_path = "data/raw/sudoku_dataset/images/image1.jpg"
    save_cell_debug(image_path)