from pathlib import Path
import csv
import cv2
import numpy as np

from cv.grid_extract import load_image, warp_grid


def normalize_dataset_filepath(filepath):
    """
    Convert CSV filepath like './images/image32.jpg' into a normalized relative path:
        images/image32.jpg
    """
    filepath = filepath.replace("\\", "/")

    if filepath.startswith("./"):
        filepath = filepath[2:]

    return filepath


def load_outlines_csv(outlines_csv_path):
    """
    Load outlines_sorted.csv into a dictionary.

    Returns:
        outlines: dict mapping normalized dataset filepath to 4 corner points.

    Example key:
        images/image32.jpg

    Example value:
        np.array([
            [p1_x, p1_y],
            [p2_x, p2_y],
            [p3_x, p3_y],
            [p4_x, p4_y],
        ], dtype=np.float32)
    """
    outlines_csv_path = Path(outlines_csv_path)
    outlines = {}

    with open(outlines_csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            filepath = normalize_dataset_filepath(row["filepath"])

            points = np.array(
                [
                    [float(row["p1_x"]), float(row["p1_y"])],
                    [float(row["p2_x"]), float(row["p2_y"])],
                    [float(row["p3_x"]), float(row["p3_y"])],
                    [float(row["p4_x"]), float(row["p4_y"])],
                ],
                dtype=np.float32,
            )

            outlines[filepath] = points

    return outlines


def image_path_to_outline_key(image_path, dataset_root):
    """
    Convert a full project path like:
        data/raw/sudoku_dataset/images/image30.jpg

    into the outline CSV key:
        images/image30.jpg
    """
    image_path = Path(image_path)
    dataset_root = Path(dataset_root)

    relative_path = image_path.relative_to(dataset_root)

    return normalize_dataset_filepath(str(relative_path))


def get_outline_points_for_image(
    image_path,
    dataset_root="data/raw/sudoku_dataset",
    outlines_csv_path="data/raw/sudoku_dataset/outlines_sorted.csv",
):
    outlines = load_outlines_csv(outlines_csv_path)
    key = image_path_to_outline_key(image_path, dataset_root)

    if key not in outlines:
        raise KeyError(f"No outline found for image key: {key}")

    return outlines[key]


def extract_warped_grid_from_outline(
    image_path,
    dataset_root="data/raw/sudoku_dataset",
    outlines_csv_path="data/raw/sudoku_dataset/outlines_sorted.csv",
    output_size=450,
):
    """
    Extract a warped Sudoku grid using dataset-provided outline annotations.
    """
    img = load_image(image_path)

    points = get_outline_points_for_image(
        image_path=image_path,
        dataset_root=dataset_root,
        outlines_csv_path=outlines_csv_path,
    )

    warped = warp_grid(img, points, output_size=output_size)

    return warped, points


def save_outline_debug(
    image_path,
    output_dir="data/processed/debug_grid_outline",
    dataset_root="data/raw/sudoku_dataset",
    outlines_csv_path="data/raw/sudoku_dataset/outlines_sorted.csv",
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    img = load_image(image_path)

    points = get_outline_points_for_image(
        image_path=image_path,
        dataset_root=dataset_root,
        outlines_csv_path=outlines_csv_path,
    )

    debug = img.copy()
    cv2.drawContours(debug, [points.astype(int).reshape(-1, 1, 2)], -1, (0, 255, 0), 4)

    warped = warp_grid(img, points, output_size=450)

    cv2.imwrite(str(output_dir / "01_detected_grid_outline.jpg"), debug)
    cv2.imwrite(str(output_dir / "02_warped_grid_outline.jpg"), warped)

    print(f"Saved outline debug images to: {output_dir.resolve()}")


if __name__ == "__main__":
    image_path = "data/raw/sudoku_dataset/images/image30.jpg"
    save_outline_debug(image_path)