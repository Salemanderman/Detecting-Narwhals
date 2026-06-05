"""
Shared clustering algorithms, feature extraction, and evaluation metrics.
No matplotlib — pure computation only.
"""

import numpy as np
import pandas as pd
from scipy.fft import dct
from sklearn.cluster import KMeans, HDBSCAN
from sklearn.mixture import BayesianGaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

NEEDS_K = {"kmeans"}




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
