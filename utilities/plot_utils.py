import matplotlib.pyplot as plt
import numpy as np 
from pathlib import Path 
import torch
import torchaudio as ta

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import utilities.feature_utils as futils
import utilities.configs as configs


def plot_pca_projection(X_pca, evr, window_meta, output_path):
    """
    Plot PCA projection of windows colored by source file.

    Args:
        X_pca:      (n_windows, n_components) PCA-projected data
        evr:        (n_components,) explained variance ratio
        window_meta: List of dicts with 'file' key for each window
        output_path: Path to save the plot PNG
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    files = [meta["file"] for meta in window_meta]
    unique_files = sorted(set(files))

    cmap = plt.get_cmap("tab20", len(unique_files)) # alternatively "viridis"
    for idx, fname in enumerate(unique_files):
        mask = [f == fname for f in files] # Boolean mask to find windows from this file
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], s=5, alpha=0.6, color=cmap(idx), label=fname)

    if len(unique_files) <= 20:  # Only show legend if not too many files
        ax.legend(markerscale=2, fontsize=6, loc="best")

    ax.set_xlabel(f"PC1 ({evr[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({evr[1]*100:.1f}%)")
    ax.set_title("PCA Projection of 5-sec Windows of Log-Mel Features")

    output_path = Path(output_path)
    fig.savefig(output_path)
    plt.close(fig)


def plot_pca_projection_3d(X_pca, evr, window_meta, output_path):
    """3-D scatter of the first three components, coloured by source file."""
    if X_pca.shape[1] < 3:
        return

    files        = [meta["file"] for meta in window_meta]
    unique_files = sorted(set(files))
    cmap         = plt.get_cmap("tab20", len(unique_files))

    fig = plt.figure(figsize=(9, 7))
    ax  = fig.add_subplot(111, projection="3d")

    for idx, fname in enumerate(unique_files):
        mask = [f == fname for f in files]
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], X_pca[mask, 2],
                   s=5, alpha=0.6, color=cmap(idx), label=fname)

    if len(unique_files) <= 20:
        ax.legend(markerscale=2, fontsize=6, loc="best")

    ax.set_xlabel(f"PC1 ({evr[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({evr[1]*100:.1f}%)")
    ax.set_zlabel(f"PC3 ({evr[2]*100:.1f}%)")
    ax.set_title("PCA Projection — first 3 components")

    fig.savefig(Path(output_path))
    plt.close(fig)


