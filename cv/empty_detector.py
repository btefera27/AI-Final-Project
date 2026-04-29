from pathlib import Path
import cv2
import numpy as np

from cv.grid_extract import extract_warped_grid
from cv.cell_extract import split_grid_into_cells, crop_cell_margin
from cv.parse_dat import parse_dat_file, print_board


def preprocess_cell(cell):
    """
    Convert one cropped cell into a clean binary image.
    White pixels should correspond mostly to digit strokes.
    """
    gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)

    # Blur slightly to reduce noise.
    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    # Invert threshold: digit strokes become white.
    thresh = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11,
        2,
    )

    return thresh


def cell_ink_score(cell):
    """
    Estimate whether a cell contains a digit using connected components.

    Instead of counting all white pixels, we keep only meaningful components.
    This reduces false positives caused by grid residue and background noise.
    """
    cropped = crop_cell_margin(cell, margin_ratio=0.20)
    thresh = preprocess_cell(cropped)

    # Remove tiny noise.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        cleaned,
        connectivity=8,
    )

    h, w = cleaned.shape
    total_area = h * w

    meaningful_area = 0
    largest_component_area = 0

    for label in range(1, num_labels):  # skip background
        x, y, comp_w, comp_h, area = stats[label]

        # Ignore tiny noise.
        if area < 8:
            continue

        # Ignore long thin fragments, often remaining grid lines.
        aspect_ratio = comp_w / comp_h if comp_h != 0 else 999
        if aspect_ratio > 5 or aspect_ratio < 0.2:
            continue

        meaningful_area += area
        largest_component_area = max(largest_component_area, area)

    # Combine total meaningful ink and largest blob.
    score = max(
        meaningful_area / total_area,
        largest_component_area / total_area,
    )

    return score


def predict_empty_mask_from_image(image_path, threshold=0.12):
    """
    Returns a 9x9 board where:
        0 = predicted empty
        1 = predicted nonempty
    """
    warped, _, _ = extract_warped_grid(image_path, output_size=450)
    cells = split_grid_into_cells(warped)

    mask = []

    for row in range(9):
        mask_row = []

        for col in range(9):
            score = cell_ink_score(cells[row][col])
            is_nonempty = score > threshold
            mask_row.append(1 if is_nonempty else 0)

        mask.append(mask_row)

    return mask


def ground_truth_nonempty_mask(dat_path):
    """
    Converts the .dat board into a 0/1 nonempty mask.
    """
    board = parse_dat_file(dat_path)

    mask = []

    for row in board:
        mask.append([1 if value != 0 else 0 for value in row])

    return mask


def compare_masks(predicted, truth):
    correct = 0
    total = 81
    errors = []

    for r in range(9):
        for c in range(9):
            if predicted[r][c] == truth[r][c]:
                correct += 1
            else:
                errors.append((r + 1, c + 1, predicted[r][c], truth[r][c]))

    accuracy = correct / total

    return accuracy, errors


def main():
    image_path = "data/raw/sudoku_dataset/images/image1.jpg"
    dat_path = "data/raw/sudoku_dataset/images/image1.dat"

    predicted = predict_empty_mask_from_image(image_path, threshold=0.12)
    truth = ground_truth_nonempty_mask(dat_path)

    print("Predicted nonempty mask:")
    print_board(predicted)

    print("\nGround-truth nonempty mask:")
    print_board(truth)

    accuracy, errors = compare_masks(predicted, truth)

    print(f"\nEmpty/nonempty accuracy: {accuracy:.3f}")
    print(f"Number of errors: {len(errors)}")

    if errors:
        print("\nErrors: row, col, predicted, truth")
        for error in errors:
            print(error)


if __name__ == "__main__":
    main()