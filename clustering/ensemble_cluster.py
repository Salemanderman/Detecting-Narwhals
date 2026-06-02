"""
Ensemble clustering via co-association matrix  (Fred & Jain, 2005).

  Generation : k-means (varied k + seed) or DPMM (varied seed, infers k)
  Consensus  : C[i,j] = fraction of runs where i and j share a cluster
  Final      : HDBSCAN on distance matrix D = 1 - C

Two modes (same as cluster.py):
  --outliers-csv   cluster only detected outliers
  (omit)           cluster every window in pca_results.npz

Usage:
    # Outliers only
    python clustering/ensemble_cluster.py \
        --pca-root    output/pipeline_results/pca \
        --outliers-csv output/pipeline_results/outliers/outliers.csv \
        --output-root output/pipeline_results/ensemble \
        --n-runs 100 --k-min 2 --k-max 10

    # All windows
    python clustering/ensemble_cluster.py \
        --pca-root    output/pipeline_results/pca \
        --output-root output/pipeline_results/ensemble_all \
        --n-runs 100 --k-min 5 --k-max 20
"""

import argparse
import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.cluster import KMeans, HDBSCAN
from sklearn.mixture import BayesianGaussianMixture
from sklearn.preprocessing import StandardScaler

CLUSTER_DIR = Path(__file__).resolve().parent
ROOT        = CLUSTER_DIR.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CLUSTER_DIR))

import utilities.configs as configs
from clustering_core  import (compute_distances,
                               compute_metrics, compute_validation_recall)
from clustering_plots import (plot_clusters, plot_cluster_sizes, plot_silhouette,
                               plot_distance_boxplot, plot_recorder_distribution,
                               plot_validation_recall, plot_coassociation_heatmap,
                               save_cluster_grid)


# ── Co-association matrix ─────────────────────────────────────────────────────

def build_coassociation(all_labels: list[np.ndarray], n: int) -> np.ndarray:
    """
    C[i,j] = fraction of runs where windows i and j belong to the same cluster.
    Noise points (-1) do not contribute to any co-occurrence.
    Diagonal is set to 1 (a point always co-occurs with itself).
    """
    C = np.zeros((n, n), dtype=np.float32)
    for labels in all_labels:
        for j in np.unique(labels):
            if j == -1:
                continue
            idx = np.where(labels == j)[0]
            C[np.ix_(idx, idx)] += 1.0
    C /= len(all_labels)
    np.fill_diagonal(C, 1.0)
    return C


def run_ensemble(X_norm: np.ndarray, n_runs: int, base_algorithm: str,
                 k_min: int, k_max: int, dpmm_max_components: int,
                 dpmm_concentration: float, seed: int) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    Run base clusterer n_runs times with random seeds.

    k-means: random k drawn from [k_min, k_max] each run (n_init=1 for diversity)
    dpmm:    BayesianGaussianMixture with DP prior — infers k automatically each run
    """
    rng        = np.random.default_rng(seed)
    all_labels = []
    desc       = f"Ensemble runs ({base_algorithm})"

    if base_algorithm == "kmeans":
        ks = rng.integers(k_min, k_max + 1, size=n_runs)
        for k in tqdm(ks, desc=desc, unit="run"):
            run_seed = int(rng.integers(0, 2**31))
            labels   = KMeans(n_clusters=int(k), n_init=1,
                              random_state=run_seed).fit_predict(X_norm)
            all_labels.append(labels)

    elif base_algorithm == "dpmm":
        for _ in tqdm(range(n_runs), desc=desc, unit="run"):
            run_seed = int(rng.integers(0, 2**31))
            labels   = BayesianGaussianMixture(
                n_components=dpmm_max_components,
                covariance_type="full",
                weight_concentration_prior_type="dirichlet_process",
                weight_concentration_prior=dpmm_concentration,
                random_state=run_seed,
                max_iter=200,
            ).fit_predict(X_norm)
            all_labels.append(labels)

    else:
        raise ValueError(f"Unknown base algorithm: {base_algorithm}")

    C = build_coassociation(all_labels, len(X_norm))
    return C, all_labels


def consensus_hdbscan(C: np.ndarray, min_cluster_size: int,
                      min_samples: int) -> np.ndarray:
    """Run HDBSCAN on the co-association distance matrix D = 1 - C."""
    D = np.clip(1.0 - C, 0.0, 1.0).astype(np.float64)
    np.fill_diagonal(D, 0.0)
    return HDBSCAN(metric="precomputed",
                   min_cluster_size=min_cluster_size,
                   min_samples=min_samples).fit_predict(D)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Ensemble clustering via co-association matrix + HDBSCAN.")
    # Paths
    ap.add_argument("--pca-root",     required=True)
    ap.add_argument("--output-root",  required=True)
    ap.add_argument("--outliers-csv", default=None,
                    help="Cluster only these outliers; omit to cluster all windows")
    ap.add_argument("--npz-root",     default=None,
                    help="Spectrogram .npz directory (needed for grids)")
    # Ensemble
    ap.add_argument("--base-algorithm", default="kmeans", choices=["kmeans", "dpmm"],
                    help="Base clusterer for ensemble generation (default: kmeans)")
    ap.add_argument("--n-runs",  type=int, default=100,
                    help="Number of base clusterings (default: 100)")
    ap.add_argument("--k-min",   type=int, default=2,
                    help="k-means: minimum k per run (default: 2)")
    ap.add_argument("--k-max",   type=int, default=10,
                    help="k-means: maximum k per run (default: 10)")
    ap.add_argument("--dpmm-max-components", type=int,   default=20,
                    help="DPMM: upper bound on clusters per run (default: 20)")
    ap.add_argument("--dpmm-concentration",  type=float, default=0.01,
                    help="DPMM: concentration α — lower = fewer clusters (default: 0.01)")
    ap.add_argument("--cluster-dims", type=int, default=10,
                    help="PCA dimensions used for clustering (default: 10)")
    ap.add_argument("--seed",    type=int, default=42)
    # HDBSCAN final step
    ap.add_argument("--min-cluster-size", type=int, default=5,
                    help="HDBSCAN min_cluster_size (default: 5)")
    ap.add_argument("--min-samples",      type=int, default=None,
                    help="HDBSCAN min_samples (default: same as min-cluster-size)")
    # All-windows extras
    ap.add_argument("--distance-metric", choices=["euclidean", "mahalanobis"],
                    default="euclidean")
    # Spectrogram
    ap.add_argument("--mel-start",   type=int,   default=None)
    ap.add_argument("--mel-end",     type=int,   default=None)
    ap.add_argument("--window-secs", type=float, default=5.0)
    ap.add_argument("--page-size", type=int, default=30,
                    help="Spectrograms per grid page — all windows saved across as many pages as needed (default: 30)")
    # Validation
    ap.add_argument("--validation-csv", default=None)
    ap.add_argument("--tolerance",      type=float, default=5.0)
    ap.add_argument("--no-plot", action="store_true", default=False)
    args = ap.parse_args()

    min_samples = args.min_samples or args.min_cluster_size
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    with open(output_root / "run_config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # ── Load PCA ─────────────────────────────────────────────────────────────
    pca_file = Path(args.pca_root) / "pca_results.npz"
    if not pca_file.exists():
        raise FileNotFoundError(f"pca_results.npz not found in {args.pca_root}")
    pca_data = np.load(pca_file, allow_pickle=True)
    X_pca    = pca_data["X_pca"]

    # ── Build DataFrame ───────────────────────────────────────────────────────
    if args.outliers_csv:
        df      = pd.read_csv(args.outliers_csv)
        if "Index" not in df.columns:
            raise ValueError("outliers.csv must have an 'Index' column.")
        indices = df["Index"].to_numpy(dtype=int)
        out_csv = "clusters.csv"
        mode    = "outliers"
        print(f"Mode: outliers  ({len(df)} windows from {args.outliers_csv})")
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
        out_csv = "clusters.csv"
        mode    = "all_windows"
        print(f"Mode: all windows  ({len(df)} windows)")

    n = len(df)

    # ── Feature matrix ────────────────────────────────────────────────────────
    dims   = min(args.cluster_dims, X_pca.shape[1])
    X_feat = X_pca[indices, :dims].astype(np.float32)
    X_norm = StandardScaler().fit_transform(X_feat)
    print(f"Feature matrix: {X_norm.shape}  (pca dims={dims})")

    # ── Ensemble generation ───────────────────────────────────────────────────
    if args.base_algorithm == "kmeans":
        gen_desc = f"k in [{args.k_min}, {args.k_max}], n_init=1"
    else:
        gen_desc = f"DPMM max_components={args.dpmm_max_components}, α={args.dpmm_concentration}"
    print(f"\nRunning {args.n_runs} {args.base_algorithm} base clusterings  ({gen_desc})...")
    C, all_labels = run_ensemble(
        X_norm, args.n_runs, args.base_algorithm,
        args.k_min, args.k_max,
        args.dpmm_max_components, args.dpmm_concentration,
        args.seed,
    )

    # Save co-association matrix
    np.save(output_root / "coassociation_matrix.npy", C)
    print(f"[save] coassociation_matrix.npy  (shape={C.shape})")

    # ── Consensus: HDBSCAN on D = 1 - C ──────────────────────────────────────
    print(f"\nRunning HDBSCAN on co-association distance matrix  "
          f"(min_cluster_size={args.min_cluster_size}, min_samples={min_samples})...")
    labels  = consensus_hdbscan(C, args.min_cluster_size, min_samples)
    k       = int(labels.max()) + 1
    n_noise = int((labels == -1).sum())

    print(f"\nFinal clustering:  {k} clusters,  {n_noise} noise points")
    for j in range(k):
        c = int((labels == j).sum())
        print(f"  Cluster {j}: {c:5d} windows  ({100*c/n:.1f}%)")
    if n_noise:
        print(f"  Noise:     {n_noise:5d} windows  ({100*n_noise/n:.1f}%)")

    # ── Metrics ───────────────────────────────────────────────────────────────
    metrics = compute_metrics(X_norm, labels, "ensemble-hdbscan")
    print(f"\nClustering metrics:")
    print(f"  Silhouette:        {metrics['silhouette']:.4f}  (higher = better)")
    print(f"  Davies-Bouldin:    {metrics['davies_bouldin']:.4f}  (lower = better)")
    print(f"  Calinski-Harabasz: {metrics['calinski_harabasz']:.1f}  (higher = better)")

    # ── Save CSVs ─────────────────────────────────────────────────────────────
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

    # ── Plots ─────────────────────────────────────────────────────────────────
    if not args.no_plot:
        labeled = labels != -1
        plot_coassociation_heatmap(C, labels, output_root / "coassociation_heatmap.png")
        plot_clusters(X_pca[indices, :2], labels, k, "ensemble-hdbscan",
                      output_root / "cluster_scatter.png", n_total=n)
        plot_cluster_sizes(labels, k, output_root / "cluster_sizes.png")
        plot_distance_boxplot(df, k, output_root / "distance_boxplot.png")
        if mode == "all_windows":
            plot_recorder_distribution(df, k, output_root / "recorder_distribution.png")
        if labeled.sum() > k > 1:
            plot_silhouette(X_norm, labels, k, output_root / "silhouette.png")

    # ── Validation recall ─────────────────────────────────────────────────────
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

    # ── Spectrogram grids ─────────────────────────────────────────────────────
    if not args.no_plot and args.npz_root:
        npz_root = Path(args.npz_root)
        if not npz_root.exists():
            print(f"[warn] --npz-root not found: {npz_root}")
        else:
            spec_cfg  = configs.get_specgram_config()
            wf_frames = max(1, round(args.window_secs * spec_cfg["sample_rate"] / spec_cfg["hop_length"]))
            all_labels_final = list(range(k)) + ([-1] if n_noise else [])
            print(f"\nSaving spectrogram grids...")
            for j in tqdm(all_labels_final, desc="Saving grids", unit="cluster"):
                label       = "noise" if j == -1 else f"cluster_{j}"
                cluster_dir = output_root / label
                cluster_dir.mkdir(exist_ok=True)
                save_cluster_grid(
                    df[df["cluster"] == j],
                    j, npz_root, wf_frames, spec_cfg,
                    cluster_dir / "spectrogram_grid.png",
                    mel_start=args.mel_start, mel_end=args.mel_end,
                    page_size=args.page_size,
                )

    print(f"\n[done] {output_root}")


if __name__ == "__main__":
    main()
