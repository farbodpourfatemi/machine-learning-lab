"""Evaluate supervised handwritten-digit classification with held-out test data."""
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def main() -> None:
    digits = load_digits()
    X_train, X_test, y_train, y_test = train_test_split(
        digits.data, digits.target, test_size=0.2, random_state=42, stratify=digits.target
    )
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    model.fit(X_train, y_train)
    predicted = model.predict(X_test)
    print(f"Test accuracy: {accuracy_score(y_test, predicted):.3f}")
    print(classification_report(y_test, predicted, zero_division=0))
    print("Confusion matrix:\n", confusion_matrix(y_test, predicted))


if __name__ == "__main__":
    main()
