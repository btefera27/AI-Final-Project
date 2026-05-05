from pathlib import Path

from cv.empty_detector import (
    predict_empty_mask_from_image,
    ground_truth_nonempty_mask,
    compare_masks,
)


def find_image_dat_pairs(dataset_folder):
    """
    Find all matching .jpg/.dat pairs in one dataset folder.
    Example:
        image1.jpg + image1.dat
    """
    dataset_folder = Path(dataset_folder)

    pairs = []

    for jpg_path in sorted(dataset_folder.glob("*.jpg")):
        dat_path = jpg_path.with_suffix(".dat")

        if dat_path.exists():
            pairs.append((jpg_path, dat_path))

    return pairs


def evaluate_folder(dataset_folder, threshold=0.07, max_images=None):
    pairs = find_image_dat_pairs(dataset_folder)

    if max_images is not None:
        pairs = pairs[:max_images]

    if not pairs:
        raise ValueError(f"No .jpg/.dat pairs found in {dataset_folder}")

    total_correct = 0
    total_cells = 0
    failed_images = []
    image_results = []

    for image_path, dat_path in pairs:
        try:
            predicted = predict_empty_mask_from_image(image_path, threshold=threshold)
            truth = ground_truth_nonempty_mask(dat_path)

            accuracy, errors = compare_masks(predicted, truth)

            correct = 81 - len(errors)
            total_correct += correct
            total_cells += 81

            image_results.append(
                {
                    "image": str(image_path),
                    "accuracy": accuracy,
                    "num_errors": len(errors),
                    "errors": errors,
                }
            )

        except Exception as e:
            failed_images.append((str(image_path), str(e)))

    overall_accuracy = total_correct / total_cells if total_cells else 0

    return overall_accuracy, image_results, failed_images


def main():
    dataset_folder = "data/raw/sudoku_dataset/images"
    threshold = 0.07

    overall_accuracy, image_results, failed_images = evaluate_folder(
        dataset_folder,
        threshold=threshold,
        max_images=None,
    )

    print(f"Dataset folder: {dataset_folder}")
    print(f"Threshold: {threshold}")
    print(f"Images evaluated: {len(image_results)}")
    print(f"Overall cell accuracy: {overall_accuracy:.4f}")

    worst = sorted(image_results, key=lambda x: x["accuracy"])[:10]

    print("\nWorst images:")
    for result in worst:
        print(
            f"{result['image']} | "
            f"accuracy={result['accuracy']:.3f} | "
            f"errors={result['num_errors']}"
        )

    if failed_images:
        print("\nFailed images:")
        for image_path, error in failed_images:
            print(f"{image_path}: {error}")


if __name__ == "__main__":
    main()