"""
Clustering of detected outlier windows.

Algorithms:  kmeans | gmm | agglomerative | hdbscan | optics
Feature modes: pca (default) | acoustic

Evaluation plots produced (all optional via --no-plot):
  cluster_scatter.png      — PC1 vs PC2 coloured by cluster
  silhouette.png           — per-sample silhouette bars (all algorithms)
  cluster_sizes.png        — bar chart of cluster sizes
  distance_boxplot.png     — PCA distance distribution per cluster
  elbow.png / elbow_compare.png — inertia vs k (k-means only)
  validation_recall.png    — narwhal call recall per cluster (needs --validation-csv)
  cluster_*/spectrogram_grid.png — spectrogram thumbnails per cluster

Usage:
    python analysis/cluster_outliers.py \
        --pca-root output/pipeline_results/pca \
        --outliers-csv output/pipeline_results/outliers/outliers.csv \
        --output-root output/pipeline_results/clusters \
        --algorithm kmeans --n-clusters 5 \
        --npz-root output/pipeline_results/npz --mel-start 9 --mel-end 61 \
        --validation-csv evaluation/validatedChristerCalls.csv
"""

import argparse
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
from tqdm import tqdm
from sklearn.cluster import KMeans, HDBSCAN, AgglomerativeClustering, OPTICS
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (silhouette_score, silhouette_samples,
                             davies_bouldin_score, calinski_harabasz_score)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import utilities.feature_utils as futils
import utilities.configs as configs

NEEDS_K   = {"kmeans", "gmm", "agglomerative"}
HAS_NOISE = {"hdbscan", "optics"}


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def acoustic_features(window: np.ndarray, low_band_bins: int = None) -> np.ndarray:
    """(n_bins, n_frames) → 10 scalar features."""
    eps = 1e-10
    n_bins       = window.shape[0]
    mean_per_bin = window.mean(axis=1)
    total_energy = mean_per_bin.sum() + eps
    frame_energy = window.mean(axis=0)

    energy    = float(total_energy / n_bins)
    aci       = float(np.abs(np.diff(window, axis=1)).sum() / (window.sum() + eps))
    centroid  = float(np.dot(np.arange(n_bins, dtype=float), mean_per_bin) / total_energy)
    flatness  = min(float(np.exp(np.mean(np.log(mean_per_bin + eps))) / (total_energy / n_bins + eps)), 1.0)
    occupancy = float((frame_energy > frame_energy.mean() * 0.5).mean())
    low_end   = low_band_bins or max(1, n_bins // 3)
    low_frac  = float(mean_per_bin[:low_end].sum() / total_energy)
    impulse   = float(window.max(axis=0).mean() / (frame_energy.mean() + eps))

    spectral_std        = float(mean_per_bin.std())
    temporal_smoothness = float(frame_energy.std() / (frame_energy.mean() + eps))
    top_n               = max(1, n_bins // 10)
    peak_concentration  = float(np.sort(mean_per_bin)[-top_n:].sum() / total_energy)

    return np.array([energy, aci, centroid, flatness, occupancy, low_frac, impulse,
                     spectral_std, temporal_smoothness, peak_concentration], dtype=np.float32)


def load_acoustic_features(df, npz_root, window_frames, spec_cfg, mel_start, mel_end):
    secs_per_frame = spec_cfg["hop_length"] / spec_cfg["sample_rate"]
    cache, rows = {}, []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Loading acoustic features", unit="window"):
        path = npz_root / row["File"]
        if path not in cache:
            try:
                cache[path], _ = futils.load_spectrogram(path, n_mels=spec_cfg.get("n_mels"))
            except Exception as e:
                print(f"  [warn] {path}: {e}")
                cache[path] = None
        S  = cache[path]
        ms = mel_start or 0
        me = mel_end or (S.shape[0] if S is not None else 0)
        t  = round(float(row["Start Time (s)"]) / secs_per_frame)
        try:
            if S is None:
                raise ValueError("not loaded")
            w = S[ms:me, t:t + window_frames]
            if w.shape[1] < window_frames:
                w = np.pad(w, ((0, 0), (0, window_frames - w.shape[1])))
            feat = acoustic_features(w)
        except Exception as e:
            print(f"  [warn] {row['File']} t={row['Start Time (s)']}: {e}")
            feat = np.zeros(10, dtype=np.float32)
        rows.append(feat)
    return np.stack(rows)


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def run_clustering(X_norm, args):
    algo = args.algorithm
    if algo == "kmeans":
        return KMeans(n_clusters=args.n_clusters, n_init=args.n_init,
                      random_state=args.seed).fit_predict(X_norm)
    elif algo == "gmm":
        return GaussianMixture(n_components=args.n_clusters,
                               random_state=args.seed).fit_predict(X_norm)
    elif algo == "agglomerative":
        return AgglomerativeClustering(n_clusters=args.n_clusters).fit_predict(X_norm)
    elif algo == "hdbscan":
        return HDBSCAN(min_cluster_size=args.min_cluster_size,
                       min_samples=args.min_samples or args.min_cluster_size).fit_predict(X_norm)
    elif algo == "optics":
        return OPTICS(min_samples=args.min_samples or 5).fit_predict(X_norm)


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def compute_metrics(X_norm, labels, algo):
    """Return dict of internal clustering metrics (skips inapplicable ones)."""
    labeled = labels != -1
    n_clusters = int(labels[labeled].max()) + 1 if labeled.any() else 0
    metrics = {"algorithm": algo, "k": n_clusters,
               "n_noise": int((labels == -1).sum())}

    if labeled.sum() > n_clusters > 1:
        metrics["silhouette"]        = float(silhouette_score(X_norm[labeled], labels[labeled]))
        metrics["davies_bouldin"]    = float(davies_bouldin_score(X_norm[labeled], labels[labeled]))
        metrics["calinski_harabasz"] = float(calinski_harabasz_score(X_norm[labeled], labels[labeled]))
    else:
        metrics["silhouette"] = metrics["davies_bouldin"] = metrics["calinski_harabasz"] = float("nan")

    return metrics


def compute_validation_recall(df, validation_csv, tolerance):
    """Per-cluster recall against validated narwhal call timestamps."""
    val     = pd.read_csv(validation_csv)
    n_total = len(val)
    records = []
    for j in sorted(df["cluster"].unique()):
        cdf     = df[df["cluster"] == j]
        covered = sum(
            ((cdf["File"] == v["file"]) & ((cdf["Start Time (s)"] - v["start_sec"]).abs() < tolerance)).any()
            for _, v in val.iterrows()
        )
        n_c = len(cdf)
        records.append({
            "cluster":   j,
            "size":      n_c,
            "covered":   covered,
            "recall":    covered / n_total if n_total else 0.0,
            "precision": covered / n_c     if n_c     else 0.0,
        })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_clusters(X_plot, labels, k, algo, save_path):
    colours    = cm.tab10(np.linspace(0, 1, k)) if k <= 10 else cm.tab20(np.linspace(0, 1, k))
    fig, ax    = plt.subplots(figsize=(9, 7))
    noise_mask = labels == -1
    if noise_mask.any():
        ax.scatter(X_plot[noise_mask, 0], X_plot[noise_mask, 1],
                   c="lightgrey", s=10, alpha=0.4, label=f"Noise (n={noise_mask.sum()})")
    for j in range(k):
        mask = labels == j
        ax.scatter(X_plot[mask, 0], X_plot[mask, 1], c=[colours[j]],
                   s=20, alpha=0.7, label=f"Cluster {j} (n={mask.sum()})")
        if mask.any():
            cx, cy = X_plot[mask, 0].mean(), X_plot[mask, 1].mean()
            ax.scatter(cx, cy, c=[colours[j]], s=200, marker="*",
                       edgecolors="black", linewidths=0.8, zorder=5)
            ax.annotate(str(j), (cx, cy), textcoords="offset points",
                        xytext=(6, 4), fontsize=9, fontweight="bold")
    ax.set_xlabel("PC1", fontsize=12)
    ax.set_ylabel("PC2", fontsize=12)
    ax.set_title(f"Clustering ({algo}, k={k})", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, markerscale=1.5)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_silhouette(X_norm, labels, k, save_path):
    """Per-sample silhouette bars grouped by cluster.
    Explains: how well each window fits its assigned cluster (1=perfect, 0=border, -1=wrong)."""
    labeled   = labels != -1
    sil_vals  = silhouette_samples(X_norm[labeled], labels[labeled])
    mean_sil  = sil_vals.mean()
    lab_clean = labels[labeled]
    colours   = cm.tab10(np.linspace(0, 1, k)) if k <= 10 else cm.tab20(np.linspace(0, 1, k))

    fig, ax = plt.subplots(figsize=(8, max(4, k * 1.2)))
    y, yticks, ylabels = 0, [], []
    for j in range(k):
        vals = np.sort(sil_vals[lab_clean == j])
        ax.barh(range(y, y + len(vals)), vals, height=1.0, color=colours[j], edgecolor="none")
        yticks.append(y + len(vals) / 2)
        ylabels.append(f"C{j}\n(n={len(vals)})")
        y += len(vals) + 2
    ax.axvline(mean_sil, color="red", linestyle="--", linewidth=1.5,
               label=f"Mean = {mean_sil:.3f}")
    ax.set_xlabel("Silhouette coefficient  (higher = better fit to cluster)", fontsize=11)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=9)
    ax.set_title(f"Silhouette plot  (k={k})", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] {save_path}")


def plot_cluster_sizes(labels, k, save_path):
    """Bar chart of how many windows are in each cluster."""
    clusters = list(range(k)) + ([-1] if (labels == -1).any() else [])
    sizes    = [int((labels == j).sum()) for j in clusters]
    xlabels  = [f"C{j}" if j >= 0 else "Noise" for j in clusters]
    colours  = (cm.tab10(np.linspace(0, 1, k)) if k <= 10 else cm.tab20(np.linspace(0, 1, k)))
    bar_colours = [colours[j] if j >= 0 else (0.8, 0.8, 0.8, 1.0) for j in clusters]

    fig, ax = plt.subplots(figsize=(max(6, len(clusters) * 1.2), 4))
    bars = ax.bar(xlabels, sizes, color=bar_colours, edgecolor="white", linewidth=0.5)
    ax.bar_label(bars, fontsize=9)
    ax.set_xlabel("Cluster", fontsize=12)
    ax.set_ylabel("Number of windows", fontsize=12)
    ax.set_title("Cluster sizes", fontsize=13, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] {save_path}")


def plot_distance_boxplot(df, k, save_path):
    """Boxplot of PCA outlier distance per cluster.
    Shows whether high-distance (most anomalous) windows concentrate in specific clusters."""
    clusters = list(range(k)) + ([-1] if (df["cluster"] == -1).any() else [])
    data     = [df.loc[df["cluster"] == j, "Distance"].values for j in clusters]
    xlabels  = [f"C{j}" if j >= 0 else "Noise" for j in clusters]

    fig, ax = plt.subplots(figsize=(max(6, len(clusters) * 1.2), 5))
    ax.boxplot(data, tick_labels=xlabels, patch_artist=True,
               boxprops=dict(facecolor="steelblue", alpha=0.6),
               medianprops=dict(color="red", linewidth=2))
    ax.set_xlabel("Cluster", fontsize=12)
    ax.set_ylabel("Distance from PCA centroid", fontsize=12)
    ax.set_title("Outlier distance distribution per cluster", fontsize=13, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] {save_path}")


def plot_validation_recall(val_df, k, tolerance, save_path):
    """Bar chart of validated narwhal call recall per cluster."""
    colours = cm.tab10(np.linspace(0, 1, k)) if k <= 10 else cm.tab20(np.linspace(0, 1, k))
    fig, ax = plt.subplots(figsize=(max(6, len(val_df) * 1.2), 4))
    for _, row in val_df.iterrows():
        j   = int(row["cluster"])
        col = "lightgrey" if j == -1 else colours[j]
        ax.bar(f"C{j}" if j >= 0 else "Noise", row["recall"], color=col, edgecolor="white")
    ax.set_xlabel("Cluster", fontsize=12)
    ax.set_ylabel("Recall", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Narwhal call recall per cluster  (tolerance={tolerance}s)",
                 fontsize=12, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] {save_path}")


def plot_elbow(X_norm, max_k, seed, n_init, save_path):
    """Inertia + silhouette vs k on one figure.
    Inertia: look for a bend (elbow). Silhouette: look for a peak. Both guide k choice."""
    ks        = list(range(2, max_k + 1))
    inertias, sil_scores = [], []
    for k in tqdm(ks, desc="Elbow search", unit="k"):
        km  = KMeans(n_clusters=k, n_init=n_init, random_state=seed)
        lbl = km.fit_predict(X_norm)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(X_norm, lbl))
        tqdm.write(f"  k={k}  inertia={km.inertia_:.1f}  silhouette={sil_scores[-1]:.3f}")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    ax1.plot(ks, inertias, marker="o", color="#2196F3", linewidth=2)
    ax1.set_ylabel("Inertia  (lower = more compact)", fontsize=11)
    ax1.set_title("Elbow + silhouette plot — choose k", fontsize=13, fontweight="bold")
    ax1.grid(True, alpha=0.3, linestyle="--")

    ax2.plot(ks, sil_scores, marker="o", color="#FF9800", linewidth=2)
    ax2.axhline(max(sil_scores), color="red", linestyle="--", alpha=0.5,
                label=f"Best k={ks[sil_scores.index(max(sil_scores))]}  ({max(sil_scores):.3f})")
    ax2.set_xlabel("Number of clusters k", fontsize=11)
    ax2.set_ylabel("Silhouette score  (higher = better)", fontsize=11)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, linestyle="--")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] {save_path}")


def plot_elbow_compare(X_pca, indices, dims_list, max_k, seed, n_init, save_path):
    ks     = list(range(2, max_k + 1))
    colors = cm.tab10(np.linspace(0, 0.9, len(dims_list)))
    fig, ax = plt.subplots(figsize=(10, 6))
    for dims, color in zip(dims_list, colors):
        d        = min(dims, X_pca.shape[1])
        X_norm   = StandardScaler().fit_transform(X_pca[indices, :d].astype(np.float32))
        inertias = [KMeans(n_clusters=k, n_init=n_init, random_state=seed).fit(X_norm).inertia_
                    for k in ks]
        ax.plot(ks, inertias, marker="o", linewidth=2, color=color, label=f"dims = {d}")
    ax.set_xlabel("k clusters", fontsize=12)
    ax.set_ylabel("Inertia  (lower = more compact clusters)", fontsize=12)
    ax.set_title("Elbow curves for varying PCA dimensions", fontsize=13, fontweight="bold")
    ax.legend(title="n PCA dims", framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_cluster_grid(cluster_df, cluster_id, npz_root, window_frames, spec_cfg,
                      save_path, mel_start=None, mel_end=None, page_size=100):
    n = len(cluster_df)
    if n == 0:
        return
    label      = "Noise" if cluster_id == -1 else f"Cluster {cluster_id}"
    rows_list  = list(cluster_df.iterrows())
    n_pages    = (n + page_size - 1) // page_size
    save_path  = Path(save_path)

    for page in range(n_pages):
        chunk  = rows_list[page * page_size:(page + 1) * page_size]
        n_cols = min(4, len(chunk))
        n_rows = (len(chunk) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
        axes = np.atleast_1d(np.array(axes)).flatten()

        for i, (_, row) in enumerate(chunk):
            ax, t = axes[i], float(row["Start Time (s)"])
            try:
                w = futils.get_window(npz_root / row["File"], t, window_frames,
                                      mel_start=mel_start, mel_end=mel_end, spec_cfg=spec_cfg)
                ax.imshow(w, aspect="auto", origin="lower", cmap="viridis")
                ax.set_title(f"{Path(row['File']).stem[:20]}\nt={t:.1f}s  d={row['Distance']:.2f}",
                             fontsize=8)
            except Exception as e:
                ax.set_title(f"Error: {e}", fontsize=7)
                ax.axis("off")

        for j in range(len(chunk), len(axes)):
            axes[j].axis("off")

        page_suffix = f"_{page + 1}" if n_pages > 1 else ""
        out = save_path.with_stem(save_path.stem + page_suffix)
        title = f"{label}  ({n} windows" + (f", page {page+1}/{n_pages})" if n_pages > 1 else ")")
        fig.suptitle(title, fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pca-root",     required=True)
    ap.add_argument("--outliers-csv", required=True)
    ap.add_argument("--output-root",  required=True)
    ap.add_argument("--npz-root",     default=None,
                    help="Needed for spectrogram grids and acoustic features.")
    ap.add_argument("--algorithm",    default="kmeans",
                    choices=["kmeans", "gmm", "agglomerative", "hdbscan", "optics"])
    ap.add_argument("--feature-mode", choices=["pca", "acoustic"], default="pca")
    # k-based
    ap.add_argument("--n-clusters",   type=int, default=5)
    ap.add_argument("--cluster-dims", type=int, default=10, help="PCA dims (pca mode).")
    ap.add_argument("--n-init",       type=int, default=10, help="k-means restarts.")
    ap.add_argument("--seed",         type=int, default=42)
    # density-based
    ap.add_argument("--min-cluster-size", type=int, default=10)
    ap.add_argument("--min-samples",      type=int, default=None,
                    help="Lower = less noise for hdbscan/optics.")
    # noise re-clustering
    ap.add_argument("--recluster-noise",        action="store_true", default=False)
    ap.add_argument("--noise-min-cluster-size", type=int, default=3)
    # spectrogram
    ap.add_argument("--mel-start",   type=int,   default=None)
    ap.add_argument("--mel-end",     type=int,   default=None)
    ap.add_argument("--window-secs", type=float, default=5.0)
    # evaluation
    ap.add_argument("--validation-csv", default=None,
                    help="Path to validatedChristerCalls.csv for per-cluster recall.")
    ap.add_argument("--tolerance",      type=float, default=5.0,
                    help="Time tolerance in seconds for validation matching (default: 5.0).")
    # elbow
    ap.add_argument("--elbow-max-k",         type=int,          default=None)
    ap.add_argument("--elbow-compare-dims",  type=int, nargs="+", default=None)
    ap.add_argument("--no-plot", action="store_true", default=False)
    args = ap.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    # Load PCA results
    pca_file = Path(args.pca_root) / "pca_results.npz"
    if not pca_file.exists():
        raise FileNotFoundError(f"pca_results.npz not found in {args.pca_root}")
    X_pca = np.load(pca_file, allow_pickle=True)["X_pca"]

    df = pd.read_csv(args.outliers_csv)
    if "Index" not in df.columns:
        raise ValueError("outliers.csv must have an 'Index' column.")
    n       = len(df)
    indices = df["Index"].to_numpy(dtype=int)

    if args.algorithm in NEEDS_K and n < args.n_clusters:
        raise ValueError(f"Fewer outliers ({n}) than clusters ({args.n_clusters}).")

    # Build feature matrix
    if args.feature_mode == "acoustic":
        if args.npz_root is None:
            raise ValueError("--npz-root is required for --feature-mode acoustic")
        npz_feat = Path(args.npz_root)
        if not npz_feat.exists():
            raise FileNotFoundError(f"--npz-root not found: {npz_feat}")
        spec_cfg = configs.get_specgram_config()
        wf       = max(1, round(args.window_secs * spec_cfg["sample_rate"] / spec_cfg["hop_length"]))
        print("Computing acoustic features...")
        X_feat       = load_acoustic_features(df, npz_feat, wf, spec_cfg, args.mel_start, args.mel_end)
        feature_desc = "acoustic(10)"
    else:
        dims         = min(args.cluster_dims, X_pca.shape[1])
        X_feat       = X_pca[indices, :dims].astype(np.float32)
        feature_desc = f"pca({dims})"

    X_norm = StandardScaler().fit_transform(X_feat)

    # Cluster
    labels  = run_clustering(X_norm, args)
    k       = int(labels.max()) + 1
    n_noise = int((labels == -1).sum())
    print(f"\nLoaded {n} outliers  |  features={feature_desc}  algorithm={args.algorithm}")
    if args.algorithm in HAS_NOISE:
        print(f"Found k={k}  noise={n_noise}")

    # Optional noise re-clustering
    if args.recluster_noise and n_noise > 0:
        if args.algorithm not in HAS_NOISE:
            print("[warn] --recluster-noise only applies to hdbscan/optics, skipping.")
        else:
            noise_mask   = labels == -1
            noise_labels = HDBSCAN(min_cluster_size=args.noise_min_cluster_size,
                                   min_samples=1).fit_predict(X_norm[noise_mask])
            n_new      = int(noise_labels.max()) + 1
            new_labels = labels.copy()
            for i, idx in enumerate(np.where(noise_mask)[0]):
                if noise_labels[i] != -1:
                    new_labels[idx] = k + noise_labels[i]
            labels  = new_labels
            k       = int(labels.max()) + 1
            n_noise = int((labels == -1).sum())
            print(f"Noise re-clustering → {n_new} sub-clusters, {n_noise} still-noise")

    # Print cluster breakdown
    for j in range(k):
        c = int((labels == j).sum())
        print(f"  Cluster {j}: {c} windows ({100*c/n:.1f}%)")
    if n_noise:
        print(f"  Noise:     {n_noise} windows ({100*n_noise/n:.1f}%)")

    # Compute and print metrics
    metrics = compute_metrics(X_norm, labels, args.algorithm)
    print(f"\n--- Clustering metrics ---")
    print(f"  Silhouette score:     {metrics['silhouette']:.4f}  "
          f"(range -1..1, higher = clusters are well-separated and compact)")
    print(f"  Davies-Bouldin index: {metrics['davies_bouldin']:.4f}  "
          f"(lower = clusters are compact relative to distance between them)")
    print(f"  Calinski-Harabasz:    {metrics['calinski_harabasz']:.1f}  "
          f"(higher = clusters are dense and well-separated)")

    # Save CSVs
    df["cluster"] = labels
    df.to_csv(output_root / "outliers_clustered.csv", index=False)
    print(f"\n[csv] {output_root / 'outliers_clustered.csv'}")

    summary = (df.groupby("cluster")
                 .agg(count=("cluster", "count"),
                      mean_distance=("Distance", "mean"),
                      max_distance=("Distance", "max"))
                 .reset_index())
    summary.to_csv(output_root / "cluster_summary.csv", index=False)
    print(f"[csv] {output_root / 'cluster_summary.csv'}")
    print(f"\n{summary.to_string(index=False)}")

    pd.DataFrame([metrics]).to_csv(output_root / "metrics.csv", index=False)
    print(f"[csv] {output_root / 'metrics.csv'}")

    # Plots
    if not args.no_plot:
        labeled = labels != -1
        plot_clusters(X_pca[indices, :2], labels, k, args.algorithm,
                      output_root / "cluster_scatter.png")
        plot_cluster_sizes(labels, k, output_root / "cluster_sizes.png")
        plot_distance_boxplot(df, k, output_root / "distance_boxplot.png")
        if labeled.sum() > k > 1:
            plot_silhouette(X_norm, labels, k, output_root / "silhouette.png")

    # Elbow (k-means only)
    if args.elbow_compare_dims:
        plot_elbow_compare(X_pca, indices, args.elbow_compare_dims,
                           args.elbow_max_k or 15, args.seed, args.n_init,
                           output_root / "elbow_compare.png")
    elif args.elbow_max_k:
        print(f"\nElbow search k=2..{args.elbow_max_k} (using same features as clustering)...")
        plot_elbow(X_norm, args.elbow_max_k, args.seed, args.n_init,
                   output_root / "elbow.png")

    # External validation
    if args.validation_csv:
        val_path = Path(args.validation_csv)
        if not val_path.exists():
            print(f"[warn] --validation-csv not found: {val_path}")
        else:
            print(f"\nValidation recall (tolerance={args.tolerance}s)...")
            val_df = compute_validation_recall(df, val_path, args.tolerance)
            val_df.to_csv(output_root / "validation_recall.csv", index=False)
            print(f"[csv] {output_root / 'validation_recall.csv'}")
            print(val_df.to_string(index=False))
            if not args.no_plot:
                plot_validation_recall(val_df, k, args.tolerance,
                                       output_root / "validation_recall.png")

    # Spectrogram grids
    if not args.no_plot and args.npz_root:
        npz_root = Path(args.npz_root)
        if not npz_root.exists():
            raise FileNotFoundError(f"--npz-root not found: {npz_root}")
        spec_cfg      = configs.get_specgram_config()
        window_frames = max(1, round(args.window_secs * spec_cfg["sample_rate"] / spec_cfg["hop_length"]))
        print(f"\nSaving spectrogram grids...")
        all_labels = list(range(k)) + ([-1] if n_noise else [])
        for j in tqdm(all_labels, desc="Saving spectrogram grids", unit="cluster"):
            label       = "noise" if j == -1 else f"cluster_{j}"
            cluster_dir = output_root / label
            cluster_dir.mkdir(exist_ok=True)
            save_cluster_grid(
                df[df["cluster"] == j].sort_values("Distance", ascending=False),
                j, npz_root, window_frames, spec_cfg,
                cluster_dir / "spectrogram_grid.png",
                mel_start=args.mel_start, mel_end=args.mel_end,
            )

    print(f"\n[done] {output_root}")


if __name__ == "__main__":
    main()
