from pathlib import Path

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


def evaluate_threshold(dataset_folder, threshold):
    pairs = find_image_dat_pairs(dataset_folder)

    total_correct = 0
    total_cells = 0
    total_tp = total_tn = total_fp = total_fn = 0
    failed = 0

    for image_path, dat_path in pairs:
        try:
            predicted = predict_empty_mask_from_image(image_path, threshold=threshold)
            truth = ground_truth_nonempty_mask(dat_path)

            tp, tn, fp, fn = count_confusion(predicted, truth)

            total_tp += tp
            total_tn += tn
            total_fp += fp
            total_fn += fn

            total_correct += tp + tn
            total_cells += 81

        except Exception:
            failed += 1

    accuracy = total_correct / total_cells if total_cells else 0

    return {
        "threshold": threshold,
        "accuracy": accuracy,
        "tp": total_tp,
        "tn": total_tn,
        "fp": total_fp,
        "fn": total_fn,
        "failed": failed,
    }


def main():
    dataset_folder = "data/raw/sudoku_dataset/images"

    thresholds = [
        0.01,
        0.02,
        0.03,
        0.04,
        0.05,
        0.06,
        0.07,
        0.08,
        0.09,
        0.10,
        0.12,
        0.15,
        0.18,
        0.20,
    ]

    results = []

    for threshold in thresholds:
        result = evaluate_threshold(dataset_folder, threshold)
        results.append(result)

    print("threshold | accuracy | FP | FN | failed")
    print("----------------------------------------")

    for r in results:
        print(
            f"{r['threshold']:>9.2f} | "
            f"{r['accuracy']:.4f}   | "
            f"{r['fp']:>4} | "
            f"{r['fn']:>4} | "
            f"{r['failed']:>6}"
        )

    best = max(results, key=lambda x: x["accuracy"])

    print("\nBest threshold:")
    print(best)


if __name__ == "__main__":
    main()