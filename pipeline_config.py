"""
Default configuration for run_outlier_pipeline.py
"""

from pathlib import Path


def get_pipeline_config():
    """
    Returns default configuration for the outlier detection pipeline.

    Returns:
        dict: Configuration dictionary with all pipeline parameters
    """
    config = dict(
        # Input/output paths
        audio_root="data/subsetWithValidatedCalls",
        output_root="output/pipeline_results",

        # Shared parameters
        window_secs=5.0,
        stride_secs=None,
        mel_start=9,
        mel_end=None,

        # PCA parameters
        n_components=20,
        pca_method="mean_std",

        # Outlier detection parameters
        distance_metric="mahalanobis",
        threshold_std=3.0,

        # Skip flags
        skip_extraction=False,  # skip npz file extraction (use existing .npz files in output)
        skip_pca=False,  # skip PCA (use existing pca_results.npz in output)
        no_plot=False,  # skip plotting outliers (only save CSV)
        no_audio_clips=False,  # skip saving audio clips for outliers (only save CSV)

        # Subset flag (for testing)
        subset_len=0,  # only process first n audio files
    )

    return config

