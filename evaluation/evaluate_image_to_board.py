from pathlib import Path
import csv

from cv.image_to_board import image_to_board, compare_boards
from cv.parse_dat import parse_dat_file


def find_image_dat_pairs(dataset_folder):
    dataset_folder = Path(dataset_folder)

    pairs = []

    for image_path in sorted(dataset_folder.glob("*.jpg")):
        dat_path = image_path.with_suffix(".dat")

        if dat_path.exists():
            pairs.append((image_path, dat_path))

    return pairs


def main():
    dataset_folder = Path("data/raw/sudoku_dataset/images")
    output_csv = Path("data/processed/image_to_board_results.csv")

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    pairs = find_image_dat_pairs(dataset_folder)

    rows = []

    total_correct = 0
    total_cells = 0
    exact_match_count = 0
    failed_count = 0

    for image_path, dat_path in pairs:
        row = {
            "image": str(image_path),
            "dat": str(dat_path),
            "status": "ok",
            "cell_accuracy": "",
            "num_errors": "",
            "exact_match": "",
            "error_message": "",
        }

        try:
            predicted = image_to_board(image_path)
            truth = parse_dat_file(dat_path)

            accuracy, errors = compare_boards(predicted, truth)

            correct = 81 - len(errors)
            total_correct += correct
            total_cells += 81

            exact_match = len(errors) == 0
            if exact_match:
                exact_match_count += 1

            row.update(
                {
                    "cell_accuracy": accuracy,
                    "num_errors": len(errors),
                    "exact_match": exact_match,
                }
            )

        except Exception as e:
            failed_count += 1
            row["status"] = "failed"
            row["error_message"] = str(e)

        rows.append(row)

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "image",
            "dat",
            "status",
            "cell_accuracy",
            "num_errors",
            "exact_match",
            "error_message",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    overall_cell_accuracy = total_correct / total_cells if total_cells else 0

    print(f"Saved results to: {output_csv}")
    print(f"Total pairs: {len(pairs)}")
    print(f"Failed images: {failed_count}")
    print(f"Evaluated images: {len(pairs) - failed_count}")
    print(f"Overall board cell accuracy: {overall_cell_accuracy:.4f}")
    print(f"Exact board matches: {exact_match_count}")
    print(
        f"Exact board match rate: "
        f"{exact_match_count / (len(pairs) - failed_count):.4f}"
    )

    ok_rows = [r for r in rows if r["status"] == "ok"]
    worst = sorted(ok_rows, key=lambda r: float(r["cell_accuracy"]))[:10]

    print("\nWorst images:")
    for r in worst:
        print(
            f"{r['image']} | "
            f"accuracy={float(r['cell_accuracy']):.3f} | "
            f"errors={r['num_errors']}"
        )


if __name__ == "__main__":
    main()