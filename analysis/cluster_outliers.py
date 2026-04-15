"""
Cluster outliers detected by finding_outliers.py using k-means.

This script loads outliers from the outliers.csv file and performs k-means 
clustering in PCA space.

Example usage:
    python analysis/cluster_outliers.py \
        --outliers-csv analysis/outlier_plots/outliers.csv \
        --pca-root analysis/pca_output \
        --output-root analysis/outlier_clusters \
        --n-clusters 3
"""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def kmeans(X: np.ndarray, n_clusters: int, max_iter: int = 100, random_state: int = 0):
    """
    K-means clustering.
    
    Args:
        X: (n_samples, n_features) array
        n_clusters: number of clusters
        max_iter: maximum iterations
        random_state: random seed for centroid initialization
        
    Returns:
        labels: (n_samples,) cluster assignments
        centroids: (n_clusters, n_features) cluster centers
        inertia: sum of squared distances
    """
    if n_clusters < 1:
        raise ValueError("n_clusters must be at least 1")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if X.shape[0] == 0:
        raise ValueError("X has no samples")
    if X.shape[0] < n_clusters:
        raise ValueError("n_clusters cannot exceed the number of samples")

    rng = np.random.default_rng(random_state)
    seeds = rng.choice(X.shape[0], size=n_clusters, replace=False)
    centroids = X[seeds].copy()

    labels = np.zeros(X.shape[0], dtype=np.int64)
    for iteration in range(max_iter):
        # Assign to nearest centroid
        distances = np.sum((X[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        new_labels = distances.argmin(axis=1)

        if np.array_equal(new_labels, labels):
            labels = new_labels
            break

        labels = new_labels
        
        # Update centroids
        new_centroids = centroids.copy()
        for cluster_idx in range(n_clusters):
            members = X[labels == cluster_idx]
            if len(members) > 0:
                new_centroids[cluster_idx] = members.mean(axis=0)
            else:
                # Reinitialize empty clusters
                new_centroids[cluster_idx] = X[rng.integers(0, X.shape[0])]

        if np.allclose(new_centroids, centroids):
            centroids = new_centroids
            break

        centroids = new_centroids

    # Compute inertia
    distances = np.sum((X - centroids[labels]) ** 2, axis=1)
    inertia = float(distances.sum())
    
    return labels, centroids, inertia


def plot_outlier_clusters(X_pca: np.ndarray, labels: np.ndarray, centroids: np.ndarray, 
                         evr: np.ndarray, output_path: Path):
    """Plot outlier clusters in PCA space."""
    if X_pca.shape[1] < 2:
        print("[warning] Need at least 2 PCA dimensions for plotting")
        return

    fig, ax = plt.subplots(figsize=(10, 7))
    n_clusters = int(labels.max()) + 1 if len(labels) > 0 else 1
    cmap = plt.get_cmap("tab10", max(n_clusters, 1))

    for cluster_idx in range(n_clusters):
        mask = labels == cluster_idx
        count = int(mask.sum())
        ax.scatter(
            X_pca[mask, 0],
            X_pca[mask, 1],
            s=50,
            alpha=0.7,
            color=cmap(cluster_idx),
            label=f"Cluster {cluster_idx} ({count} outliers)",
        )

    # Plot centroids
    ax.scatter(
        centroids[:, 0],
        centroids[:, 1],
        c="black",
        s=200,
        marker="*",
        linewidths=1,
        edgecolors="white",
        label="Centroids",
        zorder=5,
    )

    ax.set_xlabel(f"PC1 ({evr[0] * 100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({evr[1] * 100:.1f}% variance)")
    ax.set_title("K-means Clusters of Outliers")
    ax.legend(fontsize=10, loc="best")
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {output_path}")


def main():
    ap = argparse.ArgumentParser(
        description="Cluster outliers detected by finding_outliers.py using k-means."
    )
    ap.add_argument("--outliers-csv", required=True, help="Path to outliers.csv from finding_outliers.py")
    ap.add_argument("--pca-root", required=True, help="Directory containing pca_results.npz")
    ap.add_argument("--output-root", required=True, help="Where to write clustering results")
    ap.add_argument("--n-clusters", type=int, default=3, help="Number of clusters (default: 3)")
    ap.add_argument("--cluster-dims", type=int, default=10, 
                    help="How many leading PCA dimensions to use (default: 10)")
    ap.add_argument("--max-iter", type=int, default=100, help="Max k-means iterations (default: 100)")
    ap.add_argument("--random-state", type=int, default=0, help="Random seed (default: 0)")
    ap.add_argument("--no-plot", action="store_true", help="Skip plotting")
    ap.add_argument("--verbose", action="store_true", help="Verbose output")
    args = ap.parse_args()

    # Load paths
    outliers_csv = Path(args.outliers_csv)
    pca_root = Path(args.pca_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if not outliers_csv.exists():
        raise FileNotFoundError(f"Outliers CSV not found: {outliers_csv}")

    # Determine PCA file path
    if pca_root.is_dir():
        pca_file = pca_root / "pca_results.npz"
    else:
        pca_file = pca_root

    if not pca_file.exists():
        raise FileNotFoundError(f"PCA results not found: {pca_file}")

    # Load outliers CSV
    outlier_df = pd.read_csv(outliers_csv)
    outlier_indices = outlier_df["Index"].values
    
    if len(outlier_indices) == 0:
        raise ValueError("No outliers found in CSV")

    if args.verbose:
        print(f"Loaded {len(outlier_indices)} outliers from {outliers_csv}")

    # Load PCA results
    pca_data = np.load(pca_file, allow_pickle=True)
    X_pca_full = pca_data["X_pca"]
    evr = pca_data["evr"]

    # Extract outlier points only
    X_pca = X_pca_full[outlier_indices]

    if args.verbose:
        print(f"PCA shape: {X_pca_full.shape}")
        print(f"Outliers shape: {X_pca.shape}")

    # Use leading PCA dimensions for clustering
    cluster_dims = min(args.cluster_dims, X_pca.shape[1])
    X_cluster = X_pca[:, :cluster_dims]

    if args.verbose:
        print(f"Clustering on {cluster_dims} PCA dimensions")
        print(f"Requested clusters: {args.n_clusters}")

    # Run k-means
    labels, centroids, inertia = kmeans(
        X_cluster,
        n_clusters=args.n_clusters,
        max_iter=args.max_iter,
        random_state=args.random_state,
    )

    # Summary statistics
    cluster_sizes = np.bincount(labels, minlength=args.n_clusters)
    print(f"\nClustering complete!")
    print(f"Inertia: {inertia:.3f}")
    for cluster_idx, size in enumerate(cluster_sizes):
        pct = 100 * size / len(labels)
        print(f"  Cluster {cluster_idx}: {size} outliers ({pct:.1f}%)")

    # Add cluster assignments to outlier dataframe
    outlier_df["Cluster"] = labels

    # Save results
    out_csv = output_root / "outliers_clustered.csv"
    outlier_df.to_csv(out_csv, index=False)
    print(f"\nSaved to {out_csv}")

    out_npz = output_root / "outlier_cluster_results.npz"
    np.savez_compressed(
        out_npz,
        labels=labels,
        centroids=centroids,
        inertia=inertia,
        cluster_dims=int(cluster_dims),
        n_clusters=int(args.n_clusters),
        outlier_indices=outlier_indices,
    )
    print(f"Saved to {out_npz}")

    # Plot clusters
    if not args.no_plot:
        plot_path = output_root / "outlier_clusters.png"
        plot_outlier_clusters(X_pca, labels, centroids, evr, plot_path)

    print("\n[Done]")


if __name__ == "__main__":
    main()
