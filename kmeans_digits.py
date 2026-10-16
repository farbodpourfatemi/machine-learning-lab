"""Demonstrate unsupervised digit clustering using permutation-invariant metrics."""
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler


def main() -> None:
    digits = load_digits()
    X = StandardScaler().fit_transform(digits.data)
    cluster_ids = KMeans(n_clusters=10, random_state=42, n_init=10).fit_predict(X)
    print(f"Adjusted Rand Index: {adjusted_rand_score(digits.target, cluster_ids):.3f}")
    print(f"Normalized Mutual Information: {normalized_mutual_info_score(digits.target, cluster_ids):.3f}")
    print("Note: Cluster IDs are arbitrary and do NOT correspond directly to digit labels.")


if __name__ == "__main__":
    main()
