from pathlib import Path
import joblib
import cv2
import numpy as np

from cv.grid_extract_outline import extract_warped_grid_from_outline
from cv.parse_dat import parse_dat_file, print_board
from cv.grid_extract import extract_warped_grid
from cv.cell_extract import split_grid_into_cells, crop_cell_margin
from cv.empty_detector import cell_ink_score


def preprocess_digit_for_classifier(cell):
    """
    This must match the preprocessing used when building digit_cells.

    Current training pipeline:
        crop margin
        convert to grayscale
        resize to 28x28
        normalize to [0, 1]
        flatten to 784 features
    """
    cropped = crop_cell_margin(cell, margin_ratio=0.18)

    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

    resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)

    features = resized.astype(np.float32) / 255.0

    return features.flatten().reshape(1, -1)


def image_to_board(
    image_path,
    model_path="data/processed/digit_classifier_svm.joblib",
    empty_threshold=0.08,
    use_outline=True,
    confidence_threshold=0.0,
):
    """
    Convert one Sudoku image into a 9x9 integer board.

    Returns:
        board: 9x9 list where 0 means empty.
    """
    model = joblib.load(model_path)

    if use_outline:
        warped_grid, _ = extract_warped_grid_from_outline(image_path, output_size=450)
    else:
        warped_grid, _, _ = extract_warped_grid(image_path, output_size=450)
    
    cells = split_grid_into_cells(warped_grid)

    board = []

    for r in range(9):
        row = []

        for c in range(9):
            cell = cells[r][c]

            score = cell_ink_score(cell)

            if score <= empty_threshold:
                row.append(0)
            else:
                features = preprocess_digit_for_classifier(cell)
                
                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba(features)[0]
                    best_index = int(np.argmax(probabilities))
                    best_prob = float(probabilities[best_index])
                    predicted_digit = int(model.classes_[best_index])

                    if best_prob < confidence_threshold:
                        row.append(0)
                    else:
                        row.append(predicted_digit)
                else:
                    predicted_digit = int(model.predict(features)[0])
                    row.append(predicted_digit)

        board.append(row)

    return board


def compare_boards(predicted, truth):
    """
    Compare predicted 9x9 board against ground truth.

    Returns:
        accuracy: cell-level board accuracy
        errors: list of row, col, predicted, truth
    """
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
    image_path = "data/raw/sudoku_dataset/images/image30.jpg"
    dat_path = "data/raw/sudoku_dataset/images/image30.dat"

    predicted_board = image_to_board(image_path)
    truth_board = parse_dat_file(dat_path)

    print("Predicted board:")
    print_board(predicted_board)

    print("\nGround-truth board:")
    print_board(truth_board)

    accuracy, errors = compare_boards(predicted_board, truth_board)

    print(f"\nFull board cell accuracy: {accuracy:.4f}")
    print(f"Number of cell errors: {len(errors)}")

    if errors:
        print("\nErrors: row, col, predicted, truth")
        for error in errors:
            print(error)


if __name__ == "__main__":
    main()