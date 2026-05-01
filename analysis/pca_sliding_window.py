"""
Computes PCA on a sliding window across the full log-mel spectrogram.
Computes PCA with n_components components and saves the results.
NB: Only plots the 2 most significant components, even though there are more.
Use for example:
    python analysis/pca_sliding_window.py \
        --npz-root  inputDataDictionary/Aug_6229 \
        --output-root analysis/pca_output \
        --window-secs 5 \
        --stride-secs 2.5 \
        --mel-start 9 --mel-end 61 \
        --n-components 50

Or for a single file:
    python analysis/pca_sliding_window.py \
        --npz-root  inputDataDictionary/Aug_6229 \
        --output-root analysis/pca_output \
        --single-file 6229.220828162000.npz \
        --window-secs 5 \
        --stride-secs 2.5 \
        --mel-start 9 --mel-end 61 \
        --n-components 50
"""

import argparse
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import utilities.configs as configs
import utilities.feature_utils as futils
import utilities.plot_utils as putils


# Helper functions:

###### Maybe try with using the full window frames as features instead of summarising with mean+std?
###### Pros: More detailed info, PCA can find patterns across time frames.
###### Cons: Much higher dimensionality (n_mels * window_frames), risk of overfitting, more memory/computation needed for PCA.
###### Make comparison for the two approaches: run PCA with mean+std features vs run PCA with full window frames (flattened) and see which gives more meaningful components or better variance explained for the same number of components.
def window_feature(window: np.ndarray) -> np.ndarray:
    """
    Summarise a (n_mels, window_frames) window as mean+std -> (2*n_mels,) vector.
    This is to capture both the average of the spectrogram frequencies and the std without keeping all frames.
    So instead of (Frequencies X Time) vector, we get a single vector for mean and std across time, 
    which is easier to run PCA on and still captures the overall information in the spectroram window.
    """
    mu  = window.mean(axis=1)   # (n_mels,) vector of mean across time frames
    sig = window.std(axis=1)    # (n_mels,) vector of std across time frames
    return np.concatenate([mu, sig], axis=0) # (2*n_mels,) vector

def window_feature_full(window: np.ndarray) -> np.ndarray:
    """
    Flatten the full window (n_mels, window_frames) into a single vector (n_mels * window_frames,).
    This keeps all the information but results in a much higher dimensional feature space.
    PCA can then find patterns across both frequency and time dimensions, but it may require more components to capture the same variance.
    """
    return window.flatten()  # (n_mels * window_frames,) vector

def _aci_along_axis(window: np.ndarray, axis: int) -> np.ndarray:
    """ACI along one axis: sum(|diffs|) / sum(values) for each slice perpendicular to axis."""
    diffs  = np.abs(np.diff(window, axis=axis)).sum(axis=axis)
    totals = window.sum(axis=axis)
    return np.where(totals > 0, diffs / totals, 0.0)

def window_feature_ACI(window: np.ndarray) -> np.ndarray:
    """
    ACI per frequency bin (complexity over time for each mel bin) -> (n_mels,) vector.
    Captures temporal variation normalised by loudness — robust to overall volume shifts.
    """
    return _aci_along_axis(window, axis=1)  # diff over time axis

def window_feature_ACI_time(window: np.ndarray) -> np.ndarray:
    """
    ACI per time step (spectral complexity at each frame) -> (window_frames,) vector.
    Captures how spectrally complex each instant is — high during calls, low during silence.
    Note: dimensionality equals window_frames (~625 for 5s), which is larger than n_mels.
    """
    return _aci_along_axis(window, axis=0)  # diff over frequency axis

def window_feature_ACI_both(window: np.ndarray) -> np.ndarray:
    """
    Concatenation of ACI over frequency bins and ACI over time steps.
    -> (n_mels + window_frames,) vector.
    """
    return np.concatenate([window_feature_ACI(window), window_feature_ACI_time(window)])

def numpy_pca(X: np.ndarray, n_components: int):
    """
    PCA via economy SVD on zero-centred X (N, D).
    Returns:
        X_pca:     (N, n_components) projected data
        components:(n_components, D) principal axes (unit vectors)
        mean:      (D,) column means used for centring
        explained_variance_ratio: (n_components,)
    """
    mean = X.mean(axis=0)
    Xcentered = X - mean # centred

    # SVD on the centred matrix; economy form keeps at most min(N,D) singular values.
    U, s, Vt = np.linalg.svd(Xcentered, full_matrices=False)

    # Variance explained by each component.
    var_total = (s ** 2).sum()
    evr = (s ** 2) / var_total if var_total > 0 else np.zeros_like(s)

    k = min(n_components, len(s))
    components = Vt[:k]                  # (k, D)
    X_pca = Xcentered @ components.T       # (N, k)

    return X_pca, components, mean, evr[:k]




# Main function:
def main():
    ap = argparse.ArgumentParser(description="Sliding-window PCA over NPZ log-mel files.")
    ap.add_argument("--npz-root", required=True, help="Directory containing .npz feature files.")
    ap.add_argument("--output-root", required=True, help="Where to write PCA results.")
    ap.add_argument("--window-secs", type=float, default=5.0, help="Window length in seconds (default: 5).")
    ap.add_argument("--stride-secs", type=float, default=None, help="Stride between consecutive windows in seconds. (Default: non-overlapping).")
    ap.add_argument("--mel-start", type=int,   default=0, help="First mel bin to include (default: 0).")
    ap.add_argument("--mel-end", type=int, default=None,  help="Last mel bin to include (exclusive). If not provided, uses all bins (default: all).")
    ap.add_argument("--n-components", type=int, default=50, help="Number of PCA components to keep.")
    ap.add_argument("--n-mels", type=int, default=None, help="Total mel bins in the NPZ files (default: auto-detect from first file).")
    ap.add_argument("--feature-key", default="feature", help="Key inside NPZ files (default: 'feature').")
    ap.add_argument("--single-file", default=None, help="Process only a specific file. Provide name of that file.")
    ap.add_argument("--no-plot", action="store_true", help="Skip saving plots.")
    ap.add_argument("--pca-method", choices=["mean_std", "full_window", "ACI", "ACI_time", "ACI_both"], default="mean_std", help="Feature type for PCA.")
    args = ap.parse_args()

    npz_root = Path(args.npz_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)


    window_secs = args.window_secs
    stride_secs = args.stride_secs if args.stride_secs is not None else window_secs
    mel_start = args.mel_start
    mel_end = args.mel_end

    # Derive time-frame counts from spectrogram config.
    spec_config = configs.get_specgram_config()
    secs_per_frame = spec_config["hop_length"] / spec_config["sample_rate"]

    window_frames = max(1, round(window_secs / secs_per_frame))
    stride_frames = max(1, round(stride_secs / secs_per_frame))

    # Collect features from all windows from all files.
    npz_files = sorted(npz_root.glob("*.npz"))

    if args.single_file:
        npz_files = [p for p in npz_files if p.name == args.single_file] # only when single file is specified

    if not npz_files:
        print(f"ERROR, No NPZ files found under {npz_root}")
        sys.exit(1)

    print(f"NPZ root: {npz_root}")
    print(f"Output root: {output_root}")
    print(f"Window: {window_secs:.2f} s, {window_frames} frames")
    print(f"Stride: {stride_secs:.2f} s, {stride_frames} frames")
    print(f"n_components: {args.n_components}")
    print(f"Using features for pca: {args.pca_method}")
    print(f"\nFiles found: {len(npz_files)}")

    # Auto-detect n_mels from first file if not specified
    n_mels = args.n_mels
    if n_mels is None:
        try:
            S, _ = futils.load_spectrogram(npz_files[0], n_mels=None, key=args.feature_key)
            n_mels = S.shape[0]
            print(f"Auto-detected {n_mels} frequency bins from first file")
        except Exception as e:
            print(f"ERROR: Could not auto-detect n_mels from {npz_files[0]}: {e}")
            sys.exit(1)

    # Set mel_end to all bins if not specified
    if mel_end is None:
        mel_end = n_mels

    print(f"Frequency bins: [{mel_start}, {mel_end}) ({mel_end - mel_start} bins)")

    feature_rows = []  # each row: (2*(mel_end-mel_start),) vector
    window_meta  = []  # (filename, start_frame, start_sec) per window

    for i, npz_path in enumerate(npz_files): #enumerate for progress tracking
        try:
            S, sr = futils.load_spectrogram(npz_path, n_mels=n_mels, key=args.feature_key)
        except Exception as e:
            print(f"  [skip] {npz_path.name}: {e}")
            continue

        n_windows = 0
        for start_frame, win in futils.windows_from_spectrogram(S, window_frames, stride_frames,
                                                          mel_start=mel_start, mel_end=mel_end):
            if args.pca_method == "mean_std":
                feat = window_feature(win)
            elif args.pca_method == "ACI":
                feat = window_feature_ACI(win)
            elif args.pca_method == "ACI_time":
                feat = window_feature_ACI_time(win)
            elif args.pca_method == "ACI_both":
                feat = window_feature_ACI_both(win)
            else:
                feat = window_feature_full(win)
            feature_rows.append(feat)
            window_meta.append({
                "file": npz_path.name,
                "start_frame": int(start_frame),
                "start_sec": round(start_frame * secs_per_frame, 3),
            })
            n_windows += 1
        if (i+1) % 10 == 0:
            print(f"  {i+1}/{len(npz_files)} processed")
        # print(f"{npz_path.name}: T={S.shape[1]} frames, {n_windows} windows")

    if not feature_rows:
        print("[error] No windows extracted. Check window/stride settings vs recording length.")
        sys.exit(1)

    X = np.stack(feature_rows, axis=0)
    print(f"Feature matrix: {X.shape}  (windows × features)")

    # Z-score normalise each feature column before PCA.
    
    norm_mean = X.mean(axis=0)
    norm_std  = X.std(axis=0)
    norm_std  = np.where(norm_std > 0, norm_std, 1.0)  # keep silent bins at 0 rather than NaN
    X = (X - norm_mean) / norm_std

    # PCA with numpy
    n_components = min(args.n_components, X.shape[0], X.shape[1])
    print(f"Running PCA")

    X_pca, components, pca_mean, evr = numpy_pca(X, n_components)

    print(f"Variance explained: "
          f"PC1={evr[0]*100:.1f}% "
          f"PC2={evr[1]*100:.1f}% ")

    # Save PCA results with all metadata in one file
    out_npz = output_root / "pca_results.npz"

    # Extract window metadata arrays
    files_arr = np.array([window["file"] for window in window_meta], dtype=object)
    starts_arr = np.array([window["start_frame"] for window in window_meta], dtype=np.int64)
    secs_arr = np.array([window["start_sec"] for window in window_meta], dtype=np.float32)

    np.savez_compressed(
        out_npz,
        # PCA results
        X_pca=X_pca,
        components=components,
        pca_mean=pca_mean,
        evr=evr,
        # Window metadata (aligned with X_pca rows)
        window_files=files_arr,
        window_start_frames=starts_arr,
        window_start_secs=secs_arr,
        # Config parameters
        window_secs=window_secs,
        stride_secs=stride_secs,
        mel_start=mel_start,
        mel_end=mel_end,
        n_components=int(n_components),
        # Normalisation parameters (applied before PCA)
        norm_mean=norm_mean,
        norm_std=norm_std,
    )

    print(f"\nSaved to {out_npz}")

    # Save plots of pca

    if not args.no_plot:
        plot_path = output_root / "pca_plot.png"
        if len(npz_files) == 1:
            putils.plot_pca_projection_single(X_pca, evr, window_meta, plot_path)
        else:
            putils.plot_pca_projection(X_pca, evr, window_meta, plot_path)

        print(f"Saved to {plot_path}")

if __name__ == "__main__":
    main()
