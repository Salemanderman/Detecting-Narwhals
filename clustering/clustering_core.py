"""
Shared clustering algorithms, feature extraction, and evaluation metrics.
No matplotlib — pure computation only.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.cluster import KMeans, HDBSCAN, AgglomerativeClustering, OPTICS
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import utilities.feature_utils as futils

NEEDS_K   = {"kmeans", "gmm", "agglomerative"}
HAS_NOISE = {"hdbscan", "optics"}


def compute_distances(X_pca: np.ndarray, metric: str = "euclidean") -> np.ndarray:
    """Distance of every window from the centroid in PCA space."""
    mean = X_pca.mean(axis=0)
    if metric == "euclidean":
        return np.sqrt(np.sum((X_pca - mean) ** 2, axis=1))
    inv_cov = np.linalg.pinv(np.cov(X_pca, rowvar=False))
    diffs   = X_pca - mean
    return np.array([
        np.sqrt(d @ inv_cov @ d)
        for d in tqdm(diffs, desc="Computing Mahalanobis distances", unit="window")
    ])


def acoustic_features(window: np.ndarray, low_band_bins: int = None) -> np.ndarray:
    """(n_bins, n_frames) spectrogram window → 10 scalar features."""
    eps          = 1e-10
    n_bins       = window.shape[0]
    mean_per_bin = window.mean(axis=1)
    total_energy = mean_per_bin.sum() + eps
    frame_energy = window.mean(axis=0)

    energy    = float(total_energy / n_bins)
    aci       = float(np.abs(np.diff(window, axis=1)).sum() / (window.sum() + eps))
    centroid  = float(np.dot(np.arange(n_bins, dtype=float), mean_per_bin) / total_energy)
    flatness  = min(float(np.exp(np.mean(np.log(mean_per_bin + eps))) /
                          (total_energy / n_bins + eps)), 1.0)
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
    """Load spectrogram windows and compute acoustic features for every row in df."""
    secs_per_frame = spec_cfg["hop_length"] / spec_cfg["sample_rate"]
    cache, rows = {}, []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Acoustic features", unit="window"):
        path = npz_root / row["File"]
        if path not in cache:
            try:
                cache[path], _ = futils.load_spectrogram(path, n_mels=None)
            except Exception as e:
                print(f"  [warn] {path}: {e}")
                cache[path] = None
        S  = cache[path]
        ms = mel_start or 0
        me = mel_end or (S.shape[0] if S is not None else 0)
        t  = round(float(row["Start Time (s)"]) / secs_per_frame)
        try:
            if S is None:
                raise ValueError("spectrogram not loaded")
            w = S[ms:me, t:t + window_frames]
            if w.shape[1] < window_frames:
                w = np.pad(w, ((0, 0), (0, window_frames - w.shape[1])))
            feat = acoustic_features(w)
        except Exception as e:
            print(f"  [warn] {row['File']} t={row['Start Time (s)']}: {e}")
            feat = np.zeros(10, dtype=np.float32)
        rows.append(feat)
    return np.stack(rows)


def run_clustering(X_norm: np.ndarray, args) -> np.ndarray:
    """Run the selected clustering algorithm and return an integer label array."""
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
    raise ValueError(f"Unknown algorithm: {algo}")


def compute_metrics(X_norm: np.ndarray, labels: np.ndarray, algo: str) -> dict:
    """Silhouette, Davies-Bouldin, and Calinski-Harabasz internal metrics."""
    labeled    = labels != -1
    n_clusters = int(labels[labeled].max()) + 1 if labeled.any() else 0
    metrics    = {"algorithm": algo, "k": n_clusters,
                  "n_noise": int((labels == -1).sum())}
    if labeled.sum() > n_clusters > 1:
        metrics["silhouette"]        = float(silhouette_score(X_norm[labeled], labels[labeled]))
        metrics["davies_bouldin"]    = float(davies_bouldin_score(X_norm[labeled], labels[labeled]))
        metrics["calinski_harabasz"] = float(calinski_harabasz_score(X_norm[labeled], labels[labeled]))
    else:
        metrics["silhouette"] = metrics["davies_bouldin"] = metrics["calinski_harabasz"] = float("nan")
    return metrics


def compute_validation_recall(df: pd.DataFrame, validation_csv, tolerance: float) -> pd.DataFrame:
    """Per-cluster recall against validated narwhal call timestamps."""
    val     = pd.read_csv(validation_csv)
    n_total = len(val)
    records = []
    for j in sorted(df["cluster"].unique()):
        cdf     = df[df["cluster"] == j]
        covered = sum(
            ((cdf["File"] == v["file"]) &
             ((cdf["Start Time (s)"] - v["start_sec"]).abs() < tolerance)).any()
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
