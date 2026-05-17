"""
Cluster windows from the PCA pipeline.

Two modes:
  --outliers-csv   Cluster only the detected outliers from outliers.csv
  (omit)           Cluster every window in pca_results.npz

Algorithms:  kmeans | gmm | agglomerative | hdbscan | optics

Usage:
    # Cluster detected outliers
    python clustering/cluster.py \
        --pca-root output/pipeline_results/pca \
        --outliers-csv output/pipeline_results/outliers/outliers.csv \
        --output-root output/pipeline_results/clusters \
        --algorithm kmeans --n-clusters 5

    # Cluster all windows
    python clustering/cluster.py \
        --pca-root output/pipeline_results/pca \
        --output-root output/pipeline_results/all_clusters \
        --algorithm kmeans --n-clusters 6 \
        --npz-root output/pipeline_results/npz
"""

import argparse
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

CLUSTER_DIR = Path(__file__).resolve().parent
ROOT        = CLUSTER_DIR.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CLUSTER_DIR))

import utilities.configs as configs
from clustering_core  import (NEEDS_K, compute_distances,
                               run_clustering, compute_metrics, compute_validation_recall)
from clustering_plots import (plot_clusters, plot_cluster_sizes, plot_silhouette,
                               plot_distance_boxplot, plot_recorder_distribution,
                               plot_validation_recall, plot_elbow, save_cluster_grid)


def main():
    ap = argparse.ArgumentParser(
        description="Cluster PCA windows — outliers-only or all windows.")
    # Paths
    ap.add_argument("--pca-root",     required=True,  help="Directory containing pca_results.npz")
    ap.add_argument("--output-root",  required=True)
    ap.add_argument("--outliers-csv", default=None,
                    help="If given, cluster only these detected outliers; "
                         "if omitted, cluster every window in pca_results.npz")
    ap.add_argument("--npz-root",     default=None,
                    help="Spectrogram .npz directory (needed for grids and acoustic features)")
    # Algorithm
    ap.add_argument("--algorithm",   default="kmeans",
                    choices=["kmeans", "gmm", "agglomerative", "hdbscan", "optics"])
    ap.add_argument("--n-clusters",  type=int, default=5)
    ap.add_argument("--cluster-dims", type=int, default=10,
                    help="PCA dimensions to use for clustering (default: 10)")
    ap.add_argument("--n-init",       type=int, default=10, help="k-means restarts")
    ap.add_argument("--seed",         type=int, default=42)
    ap.add_argument("--min-cluster-size", type=int, default=10)
    ap.add_argument("--min-samples",      type=int, default=None)
    # All-windows mode extras
    ap.add_argument("--distance-metric", choices=["euclidean", "mahalanobis"],
                    default="euclidean",
                    help="Distance metric when loading all windows (default: euclidean)")
    ap.add_argument("--grid-samples", type=int, default=20,
                    help="Max spectrogram thumbnails per cluster (default: 20)")
    # Spectrogram
    ap.add_argument("--mel-start",   type=int,   default=None)
    ap.add_argument("--mel-end",     type=int,   default=None)
    ap.add_argument("--window-secs", type=float, default=5.0)
    # Validation
    ap.add_argument("--validation-csv", default=None,
                    help="validatedChristerCalls.csv for per-cluster recall")
    ap.add_argument("--tolerance",   type=float, default=5.0)
    # Elbow
    ap.add_argument("--elbow-max-k", type=int, default=None)
    ap.add_argument("--no-plot", action="store_true", default=False)
    args = ap.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    # ── Load PCA ────────────────────────────────────────────────────────────
    pca_file = Path(args.pca_root) / "pca_results.npz"
    if not pca_file.exists():
        raise FileNotFoundError(f"pca_results.npz not found in {args.pca_root}")
    pca_data = np.load(pca_file, allow_pickle=True)
    X_pca = pca_data["X_pca"]

    # ── Build DataFrame ──────────────────────────────────────────────────────
    if args.outliers_csv:
        df      = pd.read_csv(args.outliers_csv)
        if "Index" not in df.columns:
            raise ValueError("outliers.csv must have an 'Index' column.")
        indices = df["Index"].to_numpy(dtype=int)
        out_csv = "outliers_clustered.csv"
        mode    = "outliers"
        print(f"Mode: outliers ({len(df)} windows from {args.outliers_csv})")
    else:
        window_files  = np.array(pca_data["window_files"], dtype=str)
        window_secs   = pca_data["window_start_secs"].astype(float)
        window_frames = pca_data["window_start_frames"].astype(int)
        indices = np.arange(len(X_pca))
        print(f"Computing {args.distance_metric} distances for {len(X_pca)} windows...")
        distances = compute_distances(X_pca, args.distance_metric)
        df = pd.DataFrame({
            "File":           window_files,
            "Start Frame":    window_frames,
            "Start Time (s)": window_secs,
            "PC1":            X_pca[:, 0],
            "PC2":            X_pca[:, 1] if X_pca.shape[1] > 1 else 0.0,
            "Distance":       distances,
        })
        out_csv = "all_windows_clustered.csv"
        mode    = "all_windows"
        print(f"Mode: all windows ({len(df)} windows from pca_results.npz)")

    n = len(df)
    if args.algorithm in NEEDS_K and n < args.n_clusters:
        raise ValueError(f"Fewer windows ({n}) than clusters ({args.n_clusters}).")

    # ── Build feature matrix ─────────────────────────────────────────────────
    dims         = min(args.cluster_dims, X_pca.shape[1])
    X_feat       = X_pca[indices, :dims].astype(np.float32)
    feature_desc = f"pca({dims})"

    X_norm = StandardScaler().fit_transform(X_feat)

    # ── Elbow search (before clustering) ────────────────────────────────────
    if args.elbow_max_k and args.algorithm == "kmeans":
        print(f"\nElbow search k=2..{args.elbow_max_k}...")
        plot_elbow(X_norm, args.elbow_max_k, args.seed, args.n_init,
                   output_root / "elbow.png")
        if args.no_plot:
            print("Elbow saved. Exiting (--no-plot, clustering skipped).")
            return

    # ── Cluster ──────────────────────────────────────────────────────────────
    print(f"\nClustering {n} windows  features={feature_desc}  algorithm={args.algorithm}  k={args.n_clusters}")
    labels  = run_clustering(X_norm, args)
    k       = int(labels.max()) + 1
    n_noise = int((labels == -1).sum())

    for j in range(k):
        c = int((labels == j).sum())
        print(f"  Cluster {j}: {c:5d} windows  ({100*c/n:.1f}%)")
    if n_noise:
        print(f"  Noise:     {n_noise:5d} windows  ({100*n_noise/n:.1f}%)")

    # ── Metrics ──────────────────────────────────────────────────────────────
    metrics = compute_metrics(X_norm, labels, args.algorithm)
    print(f"\nClustering metrics:")
    print(f"  Silhouette:        {metrics['silhouette']:.4f}  (higher = better)")
    print(f"  Davies-Bouldin:    {metrics['davies_bouldin']:.4f}  (lower = better)")
    print(f"  Calinski-Harabasz: {metrics['calinski_harabasz']:.1f}  (higher = better)")

    # ── Save CSVs ────────────────────────────────────────────────────────────
    df["cluster"] = labels
    df.to_csv(output_root / out_csv, index=False)
    print(f"\n[csv] {output_root / out_csv}")

    summary = (df.groupby("cluster")
                 .agg(count=("cluster", "count"),
                      mean_distance=("Distance", "mean"),
                      max_distance=("Distance", "max"))
                 .reset_index())
    summary.to_csv(output_root / "cluster_summary.csv", index=False)
    pd.DataFrame([metrics]).to_csv(output_root / "metrics.csv", index=False)
    print(f"\n{summary.to_string(index=False)}")

    # ── Plots ────────────────────────────────────────────────────────────────
    if not args.no_plot:
        labeled = labels != -1
        plot_clusters(X_pca[indices, :2], labels, k, args.algorithm,
                      output_root / "cluster_scatter.png", n_total=n)
        plot_cluster_sizes(labels, k, output_root / "cluster_sizes.png")
        plot_distance_boxplot(df, k, output_root / "distance_boxplot.png")
        if mode == "all_windows":
            plot_recorder_distribution(df, k, output_root / "recorder_distribution.png")
        if labeled.sum() > k > 1:
            plot_silhouette(X_norm, labels, k, output_root / "silhouette.png")

    # ── Validation recall ────────────────────────────────────────────────────
    if args.validation_csv:
        val_path = Path(args.validation_csv)
        if not val_path.exists():
            print(f"[warn] --validation-csv not found: {val_path}")
        else:
            print(f"\nValidation recall (tolerance={args.tolerance}s)...")
            val_df = compute_validation_recall(df, val_path, args.tolerance)
            val_df.to_csv(output_root / "validation_recall.csv", index=False)
            print(val_df.to_string(index=False))
            if not args.no_plot:
                plot_validation_recall(val_df, k, args.tolerance,
                                       output_root / "validation_recall.png")

    # ── Spectrogram grids ────────────────────────────────────────────────────
    if not args.no_plot and args.npz_root:
        npz_root = Path(args.npz_root)
        if not npz_root.exists():
            print(f"[warn] --npz-root not found: {npz_root}")
        else:
            spec_cfg  = configs.get_specgram_config()
            wf_frames = max(1, round(args.window_secs * spec_cfg["sample_rate"] / spec_cfg["hop_length"]))
            all_labels = list(range(k)) + ([-1] if n_noise else [])
            print(f"\nSaving spectrogram grids ({args.grid_samples} samples per cluster)...")
            from tqdm import tqdm
            for j in tqdm(all_labels, desc="Saving grids", unit="cluster"):
                label       = "noise" if j == -1 else f"cluster_{j}"
                cluster_dir = output_root / label
                cluster_dir.mkdir(exist_ok=True)
                save_cluster_grid(
                    df[df["cluster"] == j],
                    j, npz_root, wf_frames, spec_cfg,
                    cluster_dir / "spectrogram_grid.png",
                    mel_start=args.mel_start, mel_end=args.mel_end,
                    max_samples=args.grid_samples,
                )

    print(f"\n[done] {output_root}")


if __name__ == "__main__":
    main()
