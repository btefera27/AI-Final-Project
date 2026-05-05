from pathlib import Path
import csv

from cv.empty_detector import (
    predict_empty_mask_from_image,
    ground_truth_nonempty_mask,
)


def find_image_dat_pairs(dataset_folder):
    dataset_folder = Path(dataset_folder)
    pairs = []

    for jpg_path in sorted(dataset_folder.glob("*.jpg")):
        dat_path = jpg_path.with_suffix(".dat")
        if dat_path.exists():
            pairs.append((jpg_path, dat_path))

    return pairs


def count_confusion(predicted, truth):
    tp = tn = fp = fn = 0

    for r in range(9):
        for c in range(9):
            p = predicted[r][c]
            t = truth[r][c]

            if p == 1 and t == 1:
                tp += 1
            elif p == 0 and t == 0:
                tn += 1
            elif p == 1 and t == 0:
                fp += 1
            elif p == 0 and t == 1:
                fn += 1

    return tp, tn, fp, fn


def main():
    dataset_folder = Path("data/raw/sudoku_dataset/images")
    output_csv = Path("data/processed/empty_detector_results.csv")
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    threshold = 0.12
    pairs = find_image_dat_pairs(dataset_folder)

    rows = []

    total_tp = total_tn = total_fp = total_fn = 0
    failures = 0

    for image_path, dat_path in pairs:
        row = {
            "image": str(image_path),
            "dat": str(dat_path),
            "threshold": threshold,
            "status": "ok",
            "accuracy": "",
            "tp": "",
            "tn": "",
            "fp": "",
            "fn": "",
            "error_message": "",
        }

        try:
            predicted = predict_empty_mask_from_image(image_path, threshold=threshold)
            truth = ground_truth_nonempty_mask(dat_path)

            tp, tn, fp, fn = count_confusion(predicted, truth)
            accuracy = (tp + tn) / 81

            total_tp += tp
            total_tn += tn
            total_fp += fp
            total_fn += fn

            row.update(
                {
                    "accuracy": accuracy,
                    "tp": tp,
                    "tn": tn,
                    "fp": fp,
                    "fn": fn,
                }
            )

        except Exception as e:
            failures += 1
            row["status"] = "failed"
            row["error_message"] = str(e)

        rows.append(row)

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "image",
            "dat",
            "threshold",
            "status",
            "accuracy",
            "tp",
            "tn",
            "fp",
            "fn",
            "error_message",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    total_cells = total_tp + total_tn + total_fp + total_fn
    overall_accuracy = (total_tp + total_tn) / total_cells

    print(f"Saved results to: {output_csv}")
    print(f"Images total: {len(pairs)}")
    print(f"Images failed: {failures}")
    print(f"Overall accuracy: {overall_accuracy:.4f}")
    print(f"TP={total_tp}, TN={total_tn}, FP={total_fp}, FN={total_fn}")


if __name__ == "__main__":
    main()