from pathlib import Path
import joblib
import cv2
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def load_digit_dataset(base_dir):
    """
    Load digit crops from:

        data/processed/digit_cells/1/
        ...
        data/processed/digit_cells/9/

    Returns:
        X: flattened image features
        y: digit labels 1-9
        paths: file paths for debugging
    """
    base_dir = Path(base_dir)

    X = []
    y = []
    paths = []

    for digit in range(1, 10):
        digit_dir = base_dir / str(digit)

        if not digit_dir.exists():
            raise FileNotFoundError(f"Missing digit folder: {digit_dir}")

        for image_path in sorted(digit_dir.glob("*.jpg")):
            img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

            if img is None:
                continue

            # Ensure consistent size.
            img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)

            # Normalize to [0, 1].
            img = img.astype(np.float32) / 255.0

            # Flatten 28x28 into 784 features.
            features = img.flatten()

            X.append(features)
            y.append(digit)
            paths.append(str(image_path))

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    return X, y, paths


def train_classifier(X_train, y_train):
    """
    Train a simple SVM classifier.

    StandardScaler helps because SVMs are sensitive to feature scale.
    """
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", C=10, gamma="scale", probability=True)),
        ]
    )

    model.fit(X_train, y_train)
    return model


def main():
    digit_dir = "data/processed/digit_cells"
    model_output = Path("data/processed/digit_classifier_svm.joblib")
    model_output.parent.mkdir(parents=True, exist_ok=True)

    X, y, paths = load_digit_dataset(digit_dir)

    print(f"Loaded digit samples: {len(X)}")
    print(f"Feature shape: {X.shape}")

    unique, counts = np.unique(y, return_counts=True)
    print("\nClass counts:")
    for digit, count in zip(unique, counts):
        print(f"{digit}: {count}")

    X_train, X_test, y_train, y_test, paths_train, paths_test = train_test_split(
        X,
        y,
        paths,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    model = train_classifier(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    print(f"\nTest accuracy: {acc:.4f}")

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, digits=4))

    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, y_pred, labels=list(range(1, 10))))

    joblib.dump(model, model_output)
    print(f"\nSaved model to: {model_output}")

    # Print a few mistakes for debugging.
    print("\nExample mistakes:")
    mistake_count = 0
    for true_label, pred_label, path in zip(y_test, y_pred, paths_test):
        if true_label != pred_label:
            print(f"true={true_label}, pred={pred_label}, file={path}")
            mistake_count += 1

        if mistake_count >= 20:
            break


if __name__ == "__main__":
    main()