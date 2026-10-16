"""Detect faces in an image supplied by the user; no personal photos bundled."""
import argparse
from pathlib import Path
import cv2


def detect_faces(image_file: Path, output_file: Path) -> int:
    image = cv2.imread(str(image_file))
    if image is None:
        raise ValueError(f"Could not read image: {image_file}")
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        raise RuntimeError("Could not load the OpenCV face detection cascade")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    boxes = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    for x, y, w, h in boxes:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 180, 0), 2)
    if not cv2.imwrite(str(output_file), image):
        raise OSError(f"Could not save result to {output_file}")
    return len(boxes)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Photo you have permission to process")
    parser.add_argument("--output", type=Path, default=Path("faces_detected.jpg"))
    args = parser.parse_args()
    try:
        count = detect_faces(args.image, args.output)
        print(f"Found {count} face(s); saved to {args.output}")
    except (ValueError, OSError, RuntimeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
