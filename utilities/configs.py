import torch
from pathlib import Path

# Configurations for mel-spectrogram feature generation shared
# across the project.
def get_specgram_config():

    configs = dict(
        sample_rate=64_000,
        n_fft=1024,
        win_length=1024,
        hop_length=512,
        window_fn=torch.hann_window,
        resample_rate=None,
        n_mels=128, # n_mels=128, # From 256.
        power=2.0,
        center=True, # Default.
        pad_mode="reflect", # Default.
        f_min=0,
        f_max=32_000, # 32 kHz is the max frequency for 64 kHz sample rate.
    )

    return configs



def get_pipeline_config():
    """
    Returns default configuration for the outlier detection pipeline.

    Returns:
        dict: Configuration dictionary with all pipeline parameters
    """
    config = dict(
        # Input/output paths
        audio_root=str(Path("data") / "subsetWithValidatedCalls"), # should work with any operating system
        npz_root=None, # if None, uses output_root/npz as default
        output_root=str(Path("output") / "pipeline_results"),  # should work with any operating system

        # Shared parameters
        window_secs=5.0,
        stride_secs=5.0,
        mel_start=11,
        mel_end=None,   # None = all bins
        n_mels=128,

        # Reduction parameters (shared)
        reduction="pca",        # "pca" or "umap"
        n_components=20,
        # PCA-specific
        pca_method="mean_std",  # "mean_std" | "full_window" | "mfcc"
        # UMAP-specific
        feature_mode="mfcc",    # "mean_std" | "mfcc" | "extended_acoustic" | "passthrough"
        umap_n_neighbors=15,
        umap_min_dist=0.1,

        # Outlier detection parameters
        distance_metric="mahalanobis",
        threshold_percentile=95.0,

        # Skip flags
        skip_extraction=False,  # skip npz file extraction (use existing .npz files in output)
        skip_pca=False,  # skip PCA (use existing pca_results.npz in output)
        no_plot=False,  # skip plotting outliers (only save CSV)
        no_audio_clips=True,   # skip saving audio clips for outliers (only save CSV)

        # Subset flag (for testing)
        subset_len=0,  # only process first n audio files

        # Audio crop (must match between extraction and outlier detection)
        audio_crop_start_secs=5,  # seconds cut from start of each recording
    )

    return config
