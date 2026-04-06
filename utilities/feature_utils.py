import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

import utilities.configs as configs

# Constants
SECS_ROUNDING_DECIMALS = 3

# # Loads .npz feature files and return feature matrix S = (T, B).
# def load_features_TxB(files, n_mels=128, key="feature"):
#     rows = []
#     for fp in files:
#         with np.load(fp, mmap_mode="r") as s:
#             if key not in s.files:
#                 return None

#             S = np.squeeze(s[key]) # Drops channel axis: (1, 128, T) -> (128, T).

#             if S.ndim == 1:
#                 raise ValueError(f"{fp} has 1D feature {S.shape}")

#             # Put time on axis 0, mel on axis 1
#             if S.shape[0] == n_mels:
#                 S = S.T
#             elif S.shape[1] == n_mels:
#                 pass
#             else:
#                 raise ValueError(f"{fp}: unexpected shape {S.shape}")

#             rows.append(S)

#     S = np.vstack(rows)
#     ids = [fp.name for fp in files]

#     return S, ids

# def load_feature_BxT(filepath, n_mels, key="feature"):
#     with np.load(filepath, mmap_mode="r") as s:
#         if key not in s.files:
#             return None

#         S = np.squeeze(s[key])  # Drops channel axis: (1, 128, T) -> (128, T).

#         if S.ndim == 1:
#             raise ValueError(f"{filepath} has 1D feature {S.shape}")

#         # Shape to (B, T).
#         if S.shape[0] == n_mels:
#             return S 
#         elif S.shape[1] == n_mels:
#             return S.T
#         else:
#             raise ValueError(f"{filepath}: unexpected shape {S.shape}")
        
# def find_min_T(files, n_mels, key="feature"):
#     Ts = []
#     valid_files = []
#     for fp in files:
#         S = load_feature_BxT(fp, n_mels, key)
#         if S is None:
#             continue 
#         Ts.append(S.shape[1])
#         valid_files.append(fp)
#     if not Ts:
#         raise FileNotFoundError("No valid feature files found.")
#     min_T = int(min(Ts))

#     return min_T, valid_files

# def crop_to_T(S, T_target, mode="left"):
#     B, T = S.shape
#     if T == T_target:
#         return S
#     elif T < T_target:
#         raise ValueError(f"Cannot crop to larger T_target {T_target} from T {T}")

#     # Crop the sequence.
#     if mode == "left":
#         start = 0
#     elif mode == "right":
#         start = T - T_target
#     elif mode == "center":
#         start = (T - T_target) // 2
#     else:
#         raise ValueError(f"Unknown crop mode {mode}")

#     # Crop. 
#     return S[:, start:start + T_target]

# def build_feature_matrix_TxB(files, n_mels, key="feature", crop_mode="left"):
#     min_T, valid_files = find_min_T(files, n_mels, key)

#     X_time_major = []
#     # X_mel_major = []
#     ids = []

#     for fp in valid_files:
#         S = load_feature_BxT(fp, n_mels, key)
#         if S is None:
#             continue 
#         S_cropped = crop_to_T(S, min_T, crop_mode)
#         X_time_major.append(S_cropped.T)  # (T, B)
#         # X_mel_major.append(S_cropped)      # (B, T)
#         ids.append(fp.name)

#     X_time_major = np.stack(X_time_major, axis=0) # (N, T, B)
#     # X_mel_major = np.stack(X_mel_major, axis=0) # (N, B, T)

#     result = {
#         "min_T": min_T,
#         "X_time_major": X_time_major,
#         "ids": ids,
#         # "X_mel_major": X_mel_major,
#     }

#     return result 

# def downsample_time_avgpool_from_db(S_db, T_target, *, ref=1.0, floor_db=None):
#     """Downsample log-mel spectrogram S_db = (M, T) to T_target time frames using average pooling.

#     Since E(log) != log(E), we need to convert back to linear spectrogram before downsampling.

#     Args:
#         S_db: Log-mel spectrogram (M, T).
#         T_target: Target number of time frames.
#         ref: Reference value for dB conversion.
#         floor_db: Floor value in dB to avoid -inf.

#     Returns:
#         Downsampled log-mel spectrogram (M, T_target).
#     """
#     x = torch.as_tensor(S_db, dtype=torch.float32)

#     unbatched = (x.dim() == 2)
#     if unbatched:
#         x = x.unsqueeze(0)  # Shape: (1, M, T)

#     if floor_db is not None:
#         x = torch.maximum(x, torch.tensor(floor_db, dtype=x.dtype, device=x.device))

#     # dB to linear power.
#     P = ref * torch.pow(10.0, x / 10.0) # (B, M, T).

#     # Average pool downsampling over time.
#     P_ds = F.adaptive_avg_pool1d(P, T_target) # (B, M, T_target).
    
#     # Safeguard against log(0).
#     eps = 1e-10
#     P_ds = torch.clamp(P_ds, min=eps)

#     # Linear power to dB.
#     X_db = 10.0 * torch.log10(P_ds / ref)  # (B, M, T_target).

#     if floor_db is not None:
#         X_db = torch.maximum(X_db, torch.tensor(floor_db, dtype=X_db.dtype, device=X_db.device))

#     if unbatched:
#         X_db = X_db.squeeze(0)  # Shape: (M, T_target).

#     return X_db

# new functions
def load_spectrogram(npz_path: Path, n_mels: int = 128, key: str = "feature"):
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