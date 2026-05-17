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


def plot_distance_boxplot(df, k, save_path):
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


def plot_recorder_distribution(df, k, save_path):
    df = df.copy()
    df["recorder"] = df["File"].apply(lambda f: Path(f).stem.split(".")[0])
    recorders = sorted(df["recorder"].unique())
    clusters  = sorted(df["cluster"].unique())
    xlabels   = ["Noise" if c == -1 else f"C{c}" for c in clusters]
    totals    = {c: len(df[df["cluster"] == c]) for c in clusters}
    fracs     = {r: [df[(df["cluster"] == c) & (df["recorder"] == r)].shape[0] / max(totals[c], 1)
                     for c in clusters]
                 for r in recorders}

    fig, ax = plt.subplots(figsize=(max(5, len(clusters) * 1.2), 4))
    bottom  = [0.0] * len(clusters)
    colours = plt.cm.tab10.colors
    for i, r in enumerate(recorders):
        ax.bar(xlabels, fracs[r], bottom=bottom, label=r, color=colours[i % len(colours)])
        bottom = [b + f for b, f in zip(bottom, fracs[r])]
    ax.axhline(0.5, color="black", linewidth=0.7, linestyle="--", alpha=0.4)
    ax.set_ylabel("Fraction of windows", fontsize=12)
    ax.set_xlabel("Cluster", fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_title("Recorder distribution per cluster", fontsize=13, fontweight="bold")
    ax.legend(title="Recorder")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] {save_path}")


def plot_validation_recall(val_df, k, tolerance, save_path):
    colours = _cluster_colours(k)
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
                      save_path, mel_start=None, mel_end=None, max_samples=20):
    n = len(cluster_df)
    if n == 0:
        return
    sample_df = cluster_df.sort_values("Distance", ascending=False).head(max_samples)
    n_show    = len(sample_df)
    n_cols    = min(4, n_show)
    n_rows    = (n_show + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    axes = np.atleast_1d(np.array(axes)).flatten()

    for i, (_, row) in enumerate(sample_df.iterrows()):
        ax, t = axes[i], float(row["Start Time (s)"])
        try:
            w = futils.get_window(npz_root / row["File"], t, window_frames,
                                  mel_start=mel_start, mel_end=mel_end, spec_cfg=spec_cfg)
            ax.imshow(w, aspect="auto", origin="lower", cmap="viridis")
            ax.set_title(f"{Path(row['File']).stem[:18]}\nt={t:.1f}s  d={row['Distance']:.2f}",
                         fontsize=8)
        except Exception as e:
            ax.set_title(f"Error: {e}", fontsize=7)
            ax.axis("off")
        ax.set_xticks([])
        ax.set_yticks([])

    for j in range(n_show, len(axes)):
        axes[j].axis("off")

    label  = "Noise" if cluster_id == -1 else f"Cluster {cluster_id}"
    suffix = f"  (showing {n_show}/{n})" if n > max_samples else f"  ({n} windows)"
    fig.suptitle(f"{label}{suffix}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(Path(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
