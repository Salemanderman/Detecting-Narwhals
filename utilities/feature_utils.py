import numpy as np
import torch.nn.functional as F
from pathlib import Path

import utilities.configs as configs

def load_spectrogram(npz_path: Path, n_mels: int = None, key: str = "feature"):
    """Load one NPZ file and return the full (n_mels, T) float32 array and audio sr."""
    with np.load(npz_path, allow_pickle=True, mmap_mode="r") as f: # mmap_mode="r" not needed since it loads 1 file at a time.
        if key not in f.files:
            raise KeyError(f"{npz_path} does not contain key '{key}'. Keys available are: {f.files}")
        spectrogram = np.squeeze(f[key]).astype(np.float32)  # (1,128,T) -> (128,T)
        sr = int(f["sr"]) if "sr" in f.files else 64_000

    if spectrogram.ndim != 2:
        raise ValueError(f"{npz_path}: expected 2-D array after squeeze, got {spectrogram.shape}")

    # Auto-detect n_mels if not provided
    if n_mels is None:
        n_mels = spectrogram.shape[0]

    # Ensure shape is (n_mels, T).
    if spectrogram.shape[0] == n_mels:
        pass
    elif spectrogram.shape[1] == n_mels:
        spectrogram = spectrogram.T
    else:
        raise ValueError(f"{npz_path}: neither dim matches n_mels={n_mels}. Shape is: {spectrogram.shape}")

    return spectrogram, sr

def windows_from_spectrogram(S: np.ndarray, window_frames: int, stride_frames: int,
                              mel_start: int = 0, mel_end: int = None):
    """returns (start_frame, window_array) smaller window from full spectrogram S"""
    n_mels, T = S.shape
    if mel_end is None:
        mel_end = n_mels
    start = 0
    while start + window_frames <= T:
        yield start, S[mel_start:mel_end, start : start + window_frames]
        start += stride_frames


def get_window(npz_path: Path, start_sec: float, window_frames: int,
               mel_start: int = None, mel_end: int = None, spec_cfg: dict = None):
    """Extract a single window from a spectrogram at a specific time.

    Args:
        npz_path: Path to NPZ file containing spectrogram.
        start_sec: Start time in seconds.
        window_frames: Number of frames in the window.
        mel_start: First mel bin to include. Defaults to 0.
        mel_end: Last mel bin (exclusive). Defaults to all.
        spec_cfg: Spectrogram parameters. Defaults to configs.get_specgram_config().

    Returns:
        Window array of shape (mel_end - mel_start, window_frames).
    """
    if spec_cfg is None:
        spec_cfg = configs.get_specgram_config()

    secs_per_frame = spec_cfg["hop_length"] / spec_cfg["sample_rate"]
    start_frame = round(start_sec / secs_per_frame)

    S, sr = load_spectrogram(npz_path, n_mels=spec_cfg["n_mels"])
    n_mels, T = S.shape

    if mel_start is None:
        mel_start = 0
    if mel_end is None:
        mel_end = n_mels

    if start_frame + window_frames > T:
        raise ValueError(f"Window exceeds spectrogram length: start_frame={start_frame}, "
                         f"window_frames={window_frames}, T={T}")

    return S[mel_start:mel_end, start_frame:start_frame + window_frames]