"""
All matplotlib plot/save functions for clustering.
No algorithm logic here — import from clustering_core instead.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples

import utilities.feature_utils as futils


def _cluster_colours(k):
    return cm.tab10(np.linspace(0, 1, k)) if k <= 10 else cm.tab20(np.linspace(0, 1, k))


def plot_clusters(X_plot, labels, k, algo, save_path, n_total=None):
    colours    = _cluster_colours(k)
    fig, ax    = plt.subplots(figsize=(9, 7))
    noise_mask = labels == -1
    if noise_mask.any():
        ax.scatter(X_plot[noise_mask, 0], X_plot[noise_mask, 1],
                   c="lightgrey", s=8, alpha=0.3, label=f"Noise (n={noise_mask.sum()})")
    for j in range(k):
        mask = labels == j
        ax.scatter(X_plot[mask, 0], X_plot[mask, 1], c=[colours[j]],
                   s=10, alpha=0.5, label=f"Cluster {j} (n={mask.sum()})")
        if mask.any():
            cx, cy = X_plot[mask, 0].mean(), X_plot[mask, 1].mean()
            ax.scatter(cx, cy, c=[colours[j]], s=200, marker="*",
                       edgecolors="black", linewidths=0.8, zorder=5)
            ax.annotate(str(j), (cx, cy), textcoords="offset points",
                        xytext=(6, 4), fontsize=9, fontweight="bold")
    n_str = f", n={n_total}" if n_total is not None else ""
    ax.set_xlabel("PC1", fontsize=12)
    ax.set_ylabel("PC2", fontsize=12)
    ax.set_title(f"Clustering  ({algo}, k={k}{n_str})", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, markerscale=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] {save_path}")


def plot_cluster_sizes(labels, k, save_path):
    clusters    = list(range(k)) + ([-1] if (labels == -1).any() else [])
    sizes       = [int((labels == j).sum()) for j in clusters]
    xlabels     = [f"C{j}" if j >= 0 else "Noise" for j in clusters]
    colours     = _cluster_colours(k)
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


def plot_silhouette(X_norm, labels, k, save_path):
    labeled   = labels != -1
    sil_vals  = silhouette_samples(X_norm[labeled], labels[labeled])
    mean_sil  = sil_vals.mean()
    lab_clean = labels[labeled]
    colours   = _cluster_colours(k)

    fig, ax = plt.subplots(figsize=(8, max(4, k * 1.2)))
    y, yticks, ylabels = 0, [], []
    for j in range(k):
        vals = np.sort(sil_vals[lab_clean == j])
        ax.barh(range(y, y + len(vals)), vals, height=1.0,
                color=colours[j], edgecolor="none")
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


def plot_elbow(X_norm, max_k, seed, n_init, save_path):
    ks                   = list(range(2, max_k + 1))
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
    ax1.set_title("Elbow + silhouette — choose k", fontsize=13, fontweight="bold")
    ax1.grid(True, alpha=0.3, linestyle="--")
    ax2.plot(ks, sil_scores, marker="o", color="#FF9800", linewidth=2)
    best_k = ks[sil_scores.index(max(sil_scores))]
    ax2.axhline(max(sil_scores), color="red", linestyle="--", alpha=0.5,
                label=f"Best k={best_k}  ({max(sil_scores):.3f})")
    ax2.set_xlabel("Number of clusters k", fontsize=11)
    ax2.set_ylabel("Silhouette score  (higher = better)", fontsize=11)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] {save_path}")


def save_cluster_grid(cluster_df, cluster_id, npz_root, window_frames, spec_cfg,
                      save_path, mel_start=None, mel_end=None, page_size=30, n_cols=None):
    """Save all windows in a cluster as paginated spectrogram grids.

    Single page  → spectrogram_grid.png
    Multiple pages → spectrogram_grid_1.png, spectrogram_grid_2.png, …
    """
    n = len(cluster_df)
    if n == 0:
        return

    save_path  = Path(save_path)
    sorted_df  = cluster_df
    n_pages    = max(1, (n + page_size - 1) // page_size)
    label      = "Noise" if cluster_id == -1 else f"Cluster {cluster_id}"

    for page in range(n_pages):
        page_df = sorted_df.iloc[page * page_size : (page + 1) * page_size]
        n_show  = len(page_df)
        n_cols  = min(n_cols if n_cols else 6, n_show)
        n_rows  = (n_show + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
        axes = np.atleast_1d(np.array(axes)).flatten()

        for i, (_, row) in enumerate(page_df.iterrows()):
            ax, t = axes[i], float(row["Start Time (s)"])
            try:
                w = futils.get_window(npz_root / row["File"], t, window_frames,
                                      mel_start=mel_start, mel_end=mel_end, spec_cfg=spec_cfg)
                ax.imshow(w, aspect="auto", origin="lower", cmap="viridis")
                ax.set_title(f"{Path(row['File']).stem[:18]}\nt={t:.1f}s",
                             fontsize=8)
            except Exception as e:
                ax.set_title(f"Error: {e}", fontsize=7)
                ax.axis("off")
            ax.set_xticks([])
            ax.set_yticks([])

        for j in range(n_show, len(axes)):
            axes[j].axis("off")

        if n_pages == 1:
            title    = f"{label}  ({n} windows)"
            out_path = save_path
        else:
            title    = f"{label}  (page {page + 1}/{n_pages}  ·  {n} windows total)"
            out_path = save_path.parent / f"{save_path.stem}_{page + 1}{save_path.suffix}"

        fig.suptitle(title, fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

