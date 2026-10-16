"""Illustrative circle/square classifier using generated images.

Small synthetic tasks do not demonstrate robust real-world image recognition.
"""
from PIL import Image, ImageDraw
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def create_image(shape: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    image = Image.new("L", (32, 32), 255)
    draw = ImageDraw.Draw(image)
    margin = int(rng.integers(5, 11))
    bounds = (margin, margin, 31 - margin, 31 - margin)
    if shape == "circle":
        draw.ellipse(bounds, fill=0)
    else:
        draw.rectangle(bounds, fill=0)
    pixels = np.asarray(image, dtype=np.float32) / 255.0
    return pixels.flatten()


def main() -> None:
    rows, labels = [], []
    for shape, label in (("circle", 0), ("square", 1)):
        for seed in range(100):
            rows.append(create_image(shape, 1000 * label + seed))
            labels.append(label)
    X_train, X_test, y_train, y_test = train_test_split(
        np.asarray(rows), np.asarray(labels), test_size=0.25,
        random_state=42, stratify=labels
    )
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    model.fit(X_train, y_train)
    print(f"Synthetic shape test accuracy: {accuracy_score(y_test, model.predict(X_test)):.3f}")
    print("Limitation: generated, centered shapes only; not a real-world vision benchmark.")


if __name__ == "__main__":
    main()
