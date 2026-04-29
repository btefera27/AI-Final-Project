from pathlib import Path
import cv2
import numpy as np


def load_image(image_path):
    image_path = Path(image_path)
    img = cv2.imread(str(image_path))

    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    return img


def preprocess_for_grid(img):
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

    return thresh


def order_points(pts):
    """
    Order four points as:
    top-left, top-right, bottom-right, bottom-left
    """
    pts = np.array(pts, dtype="float32")

    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    top_left = pts[np.argmin(s)]
    bottom_right = pts[np.argmax(s)]
    top_right = pts[np.argmin(diff)]
    bottom_left = pts[np.argmax(diff)]

    return np.array([top_left, top_right, bottom_right, bottom_left], dtype="float32")


def find_largest_quad(thresh):
    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < 1000:
            continue

        perimeter = cv2.arcLength(contour, True)

        for epsilon_factor in [0.02, 0.03, 0.04, 0.05, 0.08]:
            approx = cv2.approxPolyDP(
                contour,
                epsilon_factor * perimeter,
                True,
            )

            if len(approx) == 4:
                return approx.reshape(4, 2)

    raise ValueError("Could not find a 4-corner Sudoku grid.")


def warp_grid(img, quad_points, output_size=450):
    ordered = order_points(quad_points)

    dst = np.array(
        [
            [0, 0],
            [output_size - 1, 0],
            [output_size - 1, output_size - 1],
            [0, output_size - 1],
        ],
        dtype="float32",
    )

    transform = cv2.getPerspectiveTransform(ordered, dst)
    warped = cv2.warpPerspective(img, transform, (output_size, output_size))

    return warped


def extract_warped_grid(image_path, output_size=450):
    img = load_image(image_path)
    thresh = preprocess_for_grid(img)
    quad = find_largest_quad(thresh)
    warped = warp_grid(img, quad, output_size=output_size)

    return warped, thresh, quad


def save_grid_debug(image_path, output_dir="data/processed/debug_grid"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    img = load_image(image_path)
    thresh = preprocess_for_grid(img)
    quad = find_largest_quad(thresh)

    debug_contour = img.copy()
    cv2.drawContours(debug_contour, [quad.reshape(-1, 1, 2)], -1, (0, 255, 0), 3)

    warped = warp_grid(img, quad, output_size=450)

    cv2.imwrite(str(output_dir / "01_thresh.jpg"), thresh)
    cv2.imwrite(str(output_dir / "02_detected_grid.jpg"), debug_contour)
    cv2.imwrite(str(output_dir / "03_warped_grid.jpg"), warped)

    print(f"Saved grid debug images to: {output_dir.resolve()}")


if __name__ == "__main__":
    image_path = "data/raw/sudoku_dataset/images/image1005.jpg"
    save_grid_debug(image_path)