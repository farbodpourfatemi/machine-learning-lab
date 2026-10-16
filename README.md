# Machine Learning Lab

Introductory machine-learning coursework, organized into short reproducible experiments. These examples began as Aiolearn class exercises and were refactored with AI assistance. **They are learning demonstrations, not claims of production-level AI research.**

| Script | Focus |
| --- | --- |
| `kmeans_blobs.py` | K-Means clustering on synthetic data |
| `kmeans_digits.py` | Unsupervised clustering with label-invariant metrics |
| `digit_classifier.py` | Supervised digit classification and evaluation |
| `binary_digit_classifier.py` | Binary classification: 0 vs. 1 |
| `shape_classifier.py` | Simple synthetic circle/square classification |
| `face_detection.py` | OpenCV face detection from a user-provided image |

## Get started

```bash
python -m pip install -r requirements.txt
python digit_classifier.py
python kmeans_digits.py
python kmeans_blobs.py
python shape_classifier.py
```

Face detection needs OpenCV installed separately:

```bash
python -m pip install "opencv-python>=4.8,<5"
python face_detection.py path/to/your-own-photo.jpg
```

**Methodology notes**
- Test splits use fixed random seeds and stratification for classification where appropriate.
- Digit clustering is **unsupervised**: arbitrary cluster IDs cannot be mapped to handwritten digits by guessing a fixed list. Adjusted Rand Index and Normalized Mutual Information are used instead.
- Shape-classification results use procedurally generated toy data and should not be generalized to real images.
- Original classmates' or friends' photographs are not redistributed.
