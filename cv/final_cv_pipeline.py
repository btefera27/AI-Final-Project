"""
Final integrated Computer Vision pipeline for the Sudoku solvers.

This file combines all CV stages into one clean pipeline:

    Input image
        ↓
    Outline-based Sudoku grid extraction
        ↓
    Perspective warp to square grid
        ↓
    Split warped grid into 81 cells
        ↓
    Empty/nonempty detection for each cell
        ↓
    Digit classification for nonempty cells
        ↓
    9x9 integer Sudoku board

The output board uses the same representation expected by the solver modules:

    0 = empty cell
    1-9 = given Sudoku digit

Example output:

    [
        [0, 0, 0, 7, 0, 0, 0, 8, 0],
        [0, 9, 0, 0, 0, 3, 1, 0, 0],
        ...
    ]

Current final configuration based on evaluation:

    - Grid extraction: dataset-provided outlines from outlines_sorted.csv
    - Digit classifier: trained SVM model saved as digit_classifier_svm.joblib
    - Empty/nonempty threshold: 0.08
    - Confidence threshold: 0.0

Important note:
    The confidence threshold is kept in the code for extensibility, but our
    evaluation showed that confidence rejection did not improve exact-board
    match rate. Therefore, the default is 0.0, meaning no confidence rejection.
"""

from pathlib import Path
import csv

import cv2
import joblib
import numpy as np


# ============================================================
# Configuration
# ============================================================

# Root folder of the raw Sudoku dataset.
DATASET_ROOT = Path("data/raw/sudoku_dataset")

# Dataset-provided file containing four-corner grid annotations.
OUTLINES_CSV_PATH = DATASET_ROOT / "outlines_sorted.csv"

# Trained SVM digit classifier.
DEFAULT_MODEL_PATH = Path("data/processed/digit_classifier_svm.joblib")

# Output size for the perspective-corrected Sudoku grid.
# 450 works well because it is divisible by 9, giving 50x50 cells.
DEFAULT_GRID_SIZE = 450

# Final chosen empty/nonempty threshold.
# This was selected because it maximized exact-board match rate.
DEFAULT_EMPTY_THRESHOLD = 0.08

# Final chosen confidence threshold.
# 0.0 means no confidence rejection.
DEFAULT_CONFIDENCE_THRESHOLD = 0.0


# ============================================================
# Basic utility functions
# ============================================================

def load_image(image_path):
    """
    Load an image from disk using OpenCV.

    Parameters
    ----------
    image_path : str or pathlib.Path
        Path to the input Sudoku image.

    Returns
    -------
    image : numpy.ndarray
        Loaded BGR image.

    Raises
    ------
    FileNotFoundError
        If OpenCV cannot read the image.
    """
    image_path = Path(image_path)

    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    return image


def print_board(board):
    """
    Print a 9x9 Sudoku board in a simple readable format.

    Parameters
    ----------
    board : list[list[int]]
        9x9 Sudoku board.
    """
    for row in board:
        print(row)


# ============================================================
# Ground-truth .dat parsing
# ============================================================

def parse_dat_file(dat_path):
    """
    Parse one Sudoku .dat metadata file.

    The dataset's .dat files have the following structure:

        line 1: phone/camera model
        line 2: image metadata
        next 9 lines: Sudoku board

    In the Sudoku board:
        0 means empty cell
        1-9 means given digit

    Parameters
    ----------
    dat_path : str or pathlib.Path
        Path to the .dat file.

    Returns
    -------
    board : list[list[int]]
        9x9 integer Sudoku board.
    """
    dat_path = Path(dat_path)

    with open(dat_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    if len(lines) < 11:
        raise ValueError(f"{dat_path} does not contain enough lines.")

    board_lines = lines[2:11]
    board = []

    for line in board_lines:
        row = [int(x) for x in line.split()]

        if len(row) != 9:
            raise ValueError(f"Invalid row length in {dat_path}: {line}")

        for value in row:
            if value < 0 or value > 9:
                raise ValueError(f"Invalid Sudoku value {value} in {dat_path}")

        board.append(row)

    if len(board) != 9:
        raise ValueError(f"{dat_path} does not contain a 9x9 board.")

    return board


# ============================================================
# Outline CSV parsing
# ============================================================

def normalize_dataset_filepath(filepath):
    """
    Normalize a filepath from outlines_sorted.csv.

    The CSV uses paths like:

        ./images/image32.jpg

    This function converts them to:

        images/image32.jpg

    Normalizing paths makes lookup consistent across Windows and Unix-style paths.

    Parameters
    ----------
    filepath : str
        Filepath string from outlines_sorted.csv.

    Returns
    -------
    filepath : str
        Normalized relative filepath.
    """
    filepath = filepath.replace("\\", "/")

    if filepath.startswith("./"):
        filepath = filepath[2:]

    return filepath


def load_outlines_csv(outlines_csv_path=OUTLINES_CSV_PATH):
    """
    Load dataset-provided Sudoku grid outlines.

    Each row in outlines_sorted.csv contains:

        filepath,
        p1_x, p1_y,
        p2_x, p2_y,
        p3_x, p3_y,
        p4_x, p4_y

    We interpret these points as:

        p1 = top-left
        p2 = top-right
        p3 = bottom-right
        p4 = bottom-left

    Returns
    -------
    outlines : dict[str, numpy.ndarray]
        Dictionary mapping normalized relative image path to a 4x2 array
        of corner points.
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


def image_path_to_outline_key(image_path, dataset_root=DATASET_ROOT):
    """
    Convert a project image path into the key used by outlines_sorted.csv.

    Example
    -------
    Input:

        data/raw/sudoku_dataset/images/image30.jpg

    Output:

        images/image30.jpg

    Parameters
    ----------
    image_path : str or pathlib.Path
        Full or relative path to an image inside the dataset root.

    dataset_root : str or pathlib.Path
        Root of the Sudoku dataset.

    Returns
    -------
    key : str
        Normalized relative key used in outlines_sorted.csv.
    """
    image_path = Path(image_path)
    dataset_root = Path(dataset_root)

    relative_path = image_path.relative_to(dataset_root)

    return normalize_dataset_filepath(str(relative_path))


def get_outline_points_for_image(
    image_path,
    dataset_root=DATASET_ROOT,
    outlines_csv_path=OUTLINES_CSV_PATH,
):
    """
    Retrieve the four annotated grid corner points for one image.

    Parameters
    ----------
    image_path : str or pathlib.Path
        Path to the Sudoku image.

    dataset_root : str or pathlib.Path
        Root folder of the dataset.

    outlines_csv_path : str or pathlib.Path
        Path to outlines_sorted.csv.

    Returns
    -------
    points : numpy.ndarray
        4x2 array of corner points in the order:
        top-left, top-right, bottom-right, bottom-left.
    """
    outlines = load_outlines_csv(outlines_csv_path)
    key = image_path_to_outline_key(image_path, dataset_root)

    if key not in outlines:
        raise KeyError(f"No outline found for image key: {key}")

    return outlines[key]


# ============================================================
# Perspective warping and grid extraction
# ============================================================

def warp_grid(image, points, output_size=DEFAULT_GRID_SIZE):
    """
    Apply perspective correction to the Sudoku grid.

    The input image may be photographed at an angle. The four annotated
    grid corners define a quadrilateral in the original image. We map that
    quadrilateral to a square image of size output_size x output_size.

    Parameters
    ----------
    image : numpy.ndarray
        Original BGR image.

    points : numpy.ndarray
        4x2 array of source points in this order:
        top-left, top-right, bottom-right, bottom-left.

    output_size : int
        Width and height of the warped square grid.

    Returns
    -------
    warped : numpy.ndarray
        Perspective-corrected Sudoku grid image.
    """
    destination = np.array(
        [
            [0, 0],
            [output_size - 1, 0],
            [output_size - 1, output_size - 1],
            [0, output_size - 1],
        ],
        dtype=np.float32,
    )

    transform = cv2.getPerspectiveTransform(points.astype(np.float32), destination)

    warped = cv2.warpPerspective(
        image,
        transform,
        (output_size, output_size),
    )

    return warped


def extract_warped_grid_from_outline(
    image_path,
    dataset_root=DATASET_ROOT,
    outlines_csv_path=OUTLINES_CSV_PATH,
    output_size=DEFAULT_GRID_SIZE,
):
    """
    Extract a perspective-corrected Sudoku grid using dataset outlines.

    This is the final grid extraction method used by the project. We use
    the provided outline annotations because contour-based detection can
    fail on blurry images, partial borders, or images with nearby printed
    text and other grid-like objects.

    Parameters
    ----------
    image_path : str or pathlib.Path
        Path to the input Sudoku image.

    dataset_root : str or pathlib.Path
        Dataset root folder.

    outlines_csv_path : str or pathlib.Path
        Path to outlines_sorted.csv.

    output_size : int
        Size of the square warped grid.

    Returns
    -------
    warped_grid : numpy.ndarray
        Perspective-corrected Sudoku grid.

    points : numpy.ndarray
        4x2 source corner points used for the transform.
    """
    image = load_image(image_path)

    points = get_outline_points_for_image(
        image_path=image_path,
        dataset_root=dataset_root,
        outlines_csv_path=outlines_csv_path,
    )

    warped_grid = warp_grid(image, points, output_size=output_size)

    return warped_grid, points


# ============================================================
# Cell extraction
# ============================================================

def split_grid_into_cells(warped_grid):
    """
    Split a warped Sudoku grid into a 9x9 nested list of cell images.

    Because the grid has already been perspective-corrected into a square,
    cell boundaries can be computed by simple equal spacing.

    Parameters
    ----------
    warped_grid : numpy.ndarray
        Square Sudoku grid image.

    Returns
    -------
    cells : list[list[numpy.ndarray]]
        9x9 nested list where cells[r][c] is the image for row r, column c.
    """
    height, width = warped_grid.shape[:2]

    cell_height = height // 9
    cell_width = width // 9

    cells = []

    for r in range(9):
        row = []

        for c in range(9):
            y1 = r * cell_height
            y2 = (r + 1) * cell_height
            x1 = c * cell_width
            x2 = (c + 1) * cell_width

            cell = warped_grid[y1:y2, x1:x2]
            row.append(cell)

        cells.append(row)

    return cells


def crop_cell_margin(cell, margin_ratio=0.18):
    """
    Remove cell boundary margins to reduce grid-line artifacts.

    Sudoku cell crops often include vertical and horizontal grid lines.
    These lines can confuse the empty detector and digit classifier. We
    therefore crop away a fixed fraction from each side.

    Parameters
    ----------
    cell : numpy.ndarray
        Original cell image.

    margin_ratio : float
        Fraction of cell width/height to remove from each side.

    Returns
    -------
    cropped : numpy.ndarray
        Center crop of the cell.
    """
    height, width = cell.shape[:2]

    margin_y = int(height * margin_ratio)
    margin_x = int(width * margin_ratio)

    cropped = cell[
        margin_y:height - margin_y,
        margin_x:width - margin_x,
    ]

    return cropped


# ============================================================
# Empty/nonempty detection
# ============================================================

def cell_ink_score(cell):
    """
    Compute an ink-density score for a Sudoku cell.

    The score estimates how much dark foreground content is present in the
    center of the cell. Empty cells should have low ink density, while cells
    containing printed digits should have higher ink density.

    Steps:
        1. Crop away margins to remove grid lines.
        2. Convert to grayscale.
        3. Apply adaptive thresholding.
        4. Count dark/foreground pixels.
        5. Return foreground fraction.

    Parameters
    ----------
    cell : numpy.ndarray
        BGR cell image.

    Returns
    -------
    score : float
        Approximate fraction of foreground ink pixels.
    """
    cropped = crop_cell_margin(cell, margin_ratio=0.18)

    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

    # Blur slightly to reduce small noise.
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Invert binary threshold:
    # digits become white foreground pixels, background becomes black.
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11,
        2,
    )

    foreground_pixels = np.count_nonzero(binary)
    total_pixels = binary.size

    score = foreground_pixels / total_pixels

    return score


def is_cell_nonempty(cell, empty_threshold=DEFAULT_EMPTY_THRESHOLD):
    """
    Decide whether a cell contains a digit.

    Parameters
    ----------
    cell : numpy.ndarray
        BGR cell image.

    empty_threshold : float
        Minimum ink score required to classify a cell as nonempty.

    Returns
    -------
    bool
        True if the cell is predicted to contain a digit, False otherwise.
    """
    score = cell_ink_score(cell)

    return score > empty_threshold


# ============================================================
# Digit preprocessing and classification
# ============================================================

def preprocess_digit_for_classifier(cell):
    """
    Convert a cell image into the feature vector expected by the SVM.

    This preprocessing must match the training pipeline used in
    train_digit_classifier.py:

        crop margin
        convert to grayscale
        resize to 28x28
        normalize to [0, 1]
        flatten into 784 features

    Parameters
    ----------
    cell : numpy.ndarray
        BGR cell image.

    Returns
    -------
    features : numpy.ndarray
        Shape (1, 784), suitable for model.predict().
    """
    cropped = crop_cell_margin(cell, margin_ratio=0.18)

    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

    resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)

    normalized = resized.astype(np.float32) / 255.0

    features = normalized.flatten().reshape(1, -1)

    return features


def classify_digit(
    cell,
    model,
    confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD,
):
    """
    Classify the digit in a nonempty Sudoku cell.

    Parameters
    ----------
    cell : numpy.ndarray
        BGR cell image.

    model : sklearn model or pipeline
        Trained digit classifier.

    confidence_threshold : float
        If probability estimates are available, reject predictions whose
        maximum class probability is below this threshold by returning 0.
        Our final evaluated setting is 0.0, meaning no rejection.

    Returns
    -------
    digit : int
        Predicted digit from 1 to 9, or 0 if rejected by confidence threshold.
    """
    features = preprocess_digit_for_classifier(cell)

    # If the classifier supports probability estimates, use them.
    # Our SVM was trained with probability=True, so this branch should run.
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[0]

        best_index = int(np.argmax(probabilities))
        best_probability = float(probabilities[best_index])
        predicted_digit = int(model.classes_[best_index])

        if best_probability < confidence_threshold:
            return 0

        return predicted_digit

    # Fallback for classifiers that do not support predict_proba.
    predicted_digit = int(model.predict(features)[0])

    return predicted_digit


# ============================================================
# Final image-to-board pipeline
# ============================================================

def image_to_board(
    image_path,
    model_path=DEFAULT_MODEL_PATH,
    empty_threshold=DEFAULT_EMPTY_THRESHOLD,
    confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD,
    dataset_root=DATASET_ROOT,
    outlines_csv_path=OUTLINES_CSV_PATH,
    output_size=DEFAULT_GRID_SIZE,
):
    """
    Convert one Sudoku image into a solver-compatible 9x9 board.

    This is the main function of the finalized CV pipeline.

    Parameters
    ----------
    image_path : str or pathlib.Path
        Path to the Sudoku image.

    model_path : str or pathlib.Path
        Path to the trained digit classifier.

    empty_threshold : float
        Threshold for empty/nonempty detection.

    confidence_threshold : float
        Optional threshold for rejecting low-confidence digit predictions.

    dataset_root : str or pathlib.Path
        Root folder of the Sudoku dataset.

    outlines_csv_path : str or pathlib.Path
        Path to dataset-provided grid outlines.

    output_size : int
        Size of the warped Sudoku grid.

    Returns
    -------
    board : list[list[int]]
        9x9 Sudoku board where 0 means empty and 1-9 are given digits.
    """
    image_path = Path(image_path)
    model_path = Path(model_path)

    # Load trained digit classifier once for this image.
    model = joblib.load(model_path)

    # Stage 1: Extract and perspective-correct the Sudoku grid.
    warped_grid, _ = extract_warped_grid_from_outline(
        image_path=image_path,
        dataset_root=dataset_root,
        outlines_csv_path=outlines_csv_path,
        output_size=output_size,
    )

    # Stage 2: Split the corrected grid into 81 cells.
    cells = split_grid_into_cells(warped_grid)

    # Stage 3: For each cell, first decide whether it is empty.
    # If it is nonempty, classify the digit.
    board = []

    for r in range(9):
        row = []

        for c in range(9):
            cell = cells[r][c]

            if not is_cell_nonempty(cell, empty_threshold=empty_threshold):
                row.append(0)
            else:
                digit = classify_digit(
                    cell=cell,
                    model=model,
                    confidence_threshold=confidence_threshold,
                )
                row.append(digit)

        board.append(row)

    return board


# ============================================================
# Evaluation helpers
# ============================================================

def compare_boards(predicted, truth):
    """
    Compare a predicted board against a ground-truth board.

    Parameters
    ----------
    predicted : list[list[int]]
        Predicted 9x9 board from the CV pipeline.

    truth : list[list[int]]
        Ground-truth 9x9 board from the .dat file.

    Returns
    -------
    accuracy : float
        Fraction of correctly predicted cells.

    errors : list[tuple]
        List of errors in the form:
            (row, col, predicted_value, true_value)
        Rows and columns are 1-indexed for readability.
    """
    correct = 0
    total = 81
    errors = []

    for r in range(9):
        for c in range(9):
            predicted_value = predicted[r][c]
            true_value = truth[r][c]

            if predicted_value == true_value:
                correct += 1
            else:
                errors.append(
                    (r + 1, c + 1, predicted_value, true_value)
                )

    accuracy = correct / total

    return accuracy, errors


def classify_board_errors(predicted, truth):
    """
    Break board errors into meaningful categories.

    Categories:
        false_positive:
            Truth is empty, but prediction is a digit.

        false_negative:
            Truth is a digit, but prediction is empty.

        wrong_digit:
            Truth is a digit, prediction is also a digit, but the digit is wrong.

    This is useful because it tells us which stage of the CV pipeline is failing:
        - false positives and false negatives reflect empty/nonempty detection
        - wrong_digit reflects digit classification

    Parameters
    ----------
    predicted : list[list[int]]
        Predicted board.

    truth : list[list[int]]
        Ground-truth board.

    Returns
    -------
    summary : dict
        Dictionary with counts for each error type.
    """
    false_positive = 0
    false_negative = 0
    wrong_digit = 0

    for r in range(9):
        for c in range(9):
            p = predicted[r][c]
            t = truth[r][c]

            if p == t:
                continue

            if t == 0 and p != 0:
                false_positive += 1
            elif t != 0 and p == 0:
                false_negative += 1
            else:
                wrong_digit += 1

    return {
        "false_positive": false_positive,
        "false_negative": false_negative,
        "wrong_digit": wrong_digit,
        "total_errors": false_positive + false_negative + wrong_digit,
    }


# ============================================================
# Debug output helpers
# ============================================================

def save_debug_outputs(
    image_path,
    output_dir="data/processed/final_cv_debug",
    dataset_root=DATASET_ROOT,
    outlines_csv_path=OUTLINES_CSV_PATH,
    output_size=DEFAULT_GRID_SIZE,
):
    """
    Save visual debug outputs for one image.

    This function is useful for reports, presentations, and debugging.
    It saves:
        1. Original image with outline drawn
        2. Warped Sudoku grid
        3. Individual cell crops

    Parameters
    ----------
    image_path : str or pathlib.Path
        Path to input image.

    output_dir : str or pathlib.Path
        Directory where debug images should be saved.
    """
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image = load_image(image_path)

    points = get_outline_points_for_image(
        image_path=image_path,
        dataset_root=dataset_root,
        outlines_csv_path=outlines_csv_path,
    )

    # Draw the annotated Sudoku outline on the original image.
    outline_debug = image.copy()
    cv2.drawContours(
        outline_debug,
        [points.astype(int).reshape(-1, 1, 2)],
        -1,
        (0, 255, 0),
        4,
    )

    warped_grid = warp_grid(image, points, output_size=output_size)
    cells = split_grid_into_cells(warped_grid)

    cv2.imwrite(str(output_dir / "01_original_with_outline.jpg"), outline_debug)
    cv2.imwrite(str(output_dir / "02_warped_grid.jpg"), warped_grid)

    cells_dir = output_dir / "cells"
    cells_dir.mkdir(parents=True, exist_ok=True)

    for r in range(9):
        for c in range(9):
            cell_path = cells_dir / f"cell_r{r + 1}_c{c + 1}.jpg"
            cv2.imwrite(str(cell_path), cells[r][c])

    print(f"Saved debug outputs to: {output_dir.resolve()}")


# ============================================================
# Demo / command-line test
# ============================================================

def main():
    """
    Run the final CV pipeline on one example image.

    This main function is mainly for quick testing. For large-scale evaluation,
    use the separate evaluation scripts.
    """
    image_path = DATASET_ROOT / "images" / "image1.jpg"
    dat_path = DATASET_ROOT / "images" / "image1.dat"

    predicted_board = image_to_board(image_path)
    truth_board = parse_dat_file(dat_path)

    print("Predicted board:")
    print_board(predicted_board)

    print("\nGround-truth board:")
    print_board(truth_board)

    accuracy, errors = compare_boards(predicted_board, truth_board)
    error_summary = classify_board_errors(predicted_board, truth_board)

    print(f"\nFull board cell accuracy: {accuracy:.4f}")
    print(f"Number of cell errors: {len(errors)}")

    print("\nError summary:")
    print(error_summary)

    if errors:
        print("\nErrors: row, col, predicted, truth")
        for error in errors:
            print(error)

    # Optional: save visual debug outputs for the example image.
    save_debug_outputs(image_path)


if __name__ == "__main__":
    main()