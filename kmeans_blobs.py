"""Cluster synthetic 2D data and save a plot of the result."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score


def main() -> None:
    X, true_labels = make_blobs(n_samples=400, centers=4, cluster_std=0.6, random_state=0)
    model = KMeans(n_clusters=4, random_state=42, n_init=10)
    clusters = model.fit_predict(X)
    print(f"Adjusted Rand Index (synthetic labels): {adjusted_rand_score(true_labels, clusters):.3f}")
    plt.figure(figsize=(6, 5))
    plt.scatter(X[:, 0], X[:, 1], c=clusters, s=20)
    plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1],
                marker="x", s=120, color="black", label="Cluster centers")
    plt.title("K-Means on synthetic blobs")
    plt.legend()
    plt.tight_layout()
    out = Path("kmeans_blobs.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Plot saved to {out}")


if __name__ == "__main__":
    main()
