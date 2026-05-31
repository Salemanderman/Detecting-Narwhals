"""
Shared clustering algorithms, feature extraction, and evaluation metrics.
No matplotlib — pure computation only.
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.fft import dct
from sklearn.cluster import KMeans, HDBSCAN
from sklearn.mixture import BayesianGaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

NEEDS_K = {"kmeans"}


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


def acoustic_features_extended(window: np.ndarray, low_band_bins: int = None) -> np.ndarray:
    """(n_bins, n_frames) spectrogram window → 31 acoustic features."""
    eps          = 1e-10
    n_bins, n_frames = window.shape
    mean_per_bin = window.mean(axis=1)
    total_energy = mean_per_bin.sum() + eps
    frame_energy = window.mean(axis=0)
    bins         = np.arange(n_bins, dtype=float)

    # spectral shape
    energy       = float(total_energy / n_bins)
    log_energy   = float(np.log(total_energy + eps))
    aci          = float(np.abs(np.diff(window, axis=1)).sum() / (window.sum() + eps))
    centroid     = float(np.dot(bins, mean_per_bin) / total_energy)
    bandwidth    = float(np.sqrt(np.dot((bins - centroid) ** 2, mean_per_bin) / total_energy))
    cumsum       = np.cumsum(mean_per_bin)
    rolloff_85   = float(np.searchsorted(cumsum, 0.85 * cumsum[-1]))
    bw_90        = float(np.searchsorted(cumsum, 0.90 * cumsum[-1]) -
                         np.searchsorted(cumsum, 0.10 * cumsum[-1]))
    flatness     = min(float(np.exp(np.mean(np.log(mean_per_bin + eps))) /
                             (total_energy / n_bins + eps)), 1.0)
    p            = mean_per_bin / total_energy
    spectral_ent = float(-np.sum(p * np.log2(p + eps)))
    norm_bins    = bins - centroid
    spec_skew    = float(np.dot(norm_bins ** 3, mean_per_bin) /
                         (total_energy * (bandwidth + eps) ** 3))
    spec_kurt    = float(np.dot(norm_bins ** 4, mean_per_bin) /
                         (total_energy * (bandwidth + eps) ** 4))
    peak_bin     = float(np.argmax(mean_per_bin))
    top_n        = max(1, n_bins // 10)
    peak_conc    = float(np.sort(mean_per_bin)[-top_n:].sum() / total_energy)
    spectral_std = float(mean_per_bin.std())

    # band fractions
    low_end  = low_band_bins or max(1, n_bins // 3)
    mid_end  = 2 * low_end
    low_frac = float(mean_per_bin[:low_end].sum() / total_energy)
    mid_frac = float(mean_per_bin[low_end:mid_end].sum() / total_energy)
    high_frac = float(mean_per_bin[mid_end:].sum() / total_energy)

    # spectral contrast and flux
    n_c           = max(1, n_bins // 10)
    sorted_bins   = np.sort(mean_per_bin)
    spec_contrast = float(sorted_bins[-n_c:].mean() / (sorted_bins[:n_c].mean() + eps))
    spec_flux     = float(np.mean(np.abs(np.diff(window, axis=1)).sum(axis=0)))

    # temporal features
    occupancy    = float((frame_energy > frame_energy.mean() * 0.5).mean())
    impulse      = float(window.max(axis=0).mean() / (frame_energy.mean() + eps))
    rms_energy   = float(np.sqrt((frame_energy ** 2).mean()))
    crest_factor = float(frame_energy.max() / (rms_energy + eps))
    temp_smooth  = float(frame_energy.std() / (frame_energy.mean() + eps))
    fe_norm      = frame_energy / (frame_energy.sum() + eps)
    temporal_ent = float(-np.sum(fe_norm * np.log2(fe_norm + eps)))
    fe_std       = frame_energy.std() + eps
    temp_skew    = float(np.mean(((frame_energy - frame_energy.mean()) / fe_std) ** 3))
    temp_kurt    = float(np.mean(((frame_energy - frame_energy.mean()) / fe_std) ** 4))
    rise_time    = float(np.argmax(frame_energy) / (n_frames - 1 + eps))
    n_peaks      = float(np.sum(np.diff(np.sign(np.diff(mean_per_bin))) < 0))
    temp_iqr     = float(np.percentile(frame_energy, 75) - np.percentile(frame_energy, 25))

    return np.array([
        energy, log_energy, aci, centroid, bandwidth, rolloff_85, bw_90, flatness,
        spectral_ent, spec_skew, spec_kurt, peak_bin, peak_conc, spectral_std,
        low_frac, mid_frac, high_frac, spec_contrast, spec_flux,
        occupancy, impulse, rms_energy, crest_factor, temp_smooth,
        temporal_ent, temp_skew, temp_kurt, rise_time, n_peaks, temp_iqr,
        float(np.sum(frame_energy > frame_energy.mean())),
    ], dtype=np.float32)


def mfcc_features(window: np.ndarray, n_mfcc: int = 20) -> np.ndarray:
    """(n_bins, n_frames) log-mel spectrogram window → 2*n_mfcc MFCC features (mean + std).

    Applies DCT along the frequency axis to each frame, keeps the first n_mfcc
    coefficients, then summarises across time with mean and std.
    """
    
    # window is already log-mel; DCT along frequency axis → (n_mfcc, n_frames)
    mfccs = dct(window, axis=0, norm="ortho")[:n_mfcc, :]
    return np.concatenate([mfccs.mean(axis=1), mfccs.std(axis=1)]).astype(np.float32)


def run_clustering(X_norm: np.ndarray, args) -> np.ndarray:
    """Run the selected clustering algorithm and return an integer label array."""
    algo = args.algorithm
    if algo == "kmeans":
        return KMeans(n_clusters=args.n_clusters, n_init=10,
                      random_state=args.seed).fit_predict(X_norm)
    elif algo == "hdbscan":
        return HDBSCAN(min_cluster_size=args.min_cluster_size,
                       min_samples=args.min_samples or args.min_cluster_size).fit_predict(X_norm)
    elif algo == "dpmm":
        max_k       = getattr(args, "dpmm_max_components", 20)
        alpha       = getattr(args, "dpmm_concentration",  0.01)
        return BayesianGaussianMixture(
            n_components=max_k,
            covariance_type="full",
            weight_concentration_prior_type="dirichlet_process",
            weight_concentration_prior=alpha,
            random_state=args.seed,
            max_iter=200,
        ).fit_predict(X_norm)
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
