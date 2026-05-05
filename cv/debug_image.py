from pathlib import Path
import cv2


def load_image(image_path):
    image_path = Path(image_path)

    img = cv2.imread(str(image_path))

    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    return img


def save_debug_images(image_path, output_dir="data/processed/debug"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    img = load_image(image_path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    thresh = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11,
        2,
    )

    cv2.imwrite(str(output_dir / "01_original.jpg"), img)
    cv2.imwrite(str(output_dir / "02_gray.jpg"), gray)
    cv2.imwrite(str(output_dir / "03_blur.jpg"), blur)
    cv2.imwrite(str(output_dir / "04_thresh.jpg"), thresh)

    print(f"Saved debug images to: {output_dir.resolve()}")


if __name__ == "__main__":
    image_path = "data/raw/sudoku_dataset/images/image1.jpg"
    save_debug_images(image_path)