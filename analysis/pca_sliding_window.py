"""
Use for example:
    python analysis/pca_sliding_window.py \
        --input-root  inputDataDictionary/Aug_6229 \
        --output-root analysis/pca_output \
        --window-secs 5 \
        --stride-secs 2.5 \
        --mel-start 9 --mel-end 61 \
        --n-components 50

Or for a single file:
    python analysis/pca_sliding_window.py \
        --input-root  inputDataDictionary/Aug_6229 \
        --output-root analysis/pca_output \
        --single-file 6229.220828162000.npz \
        --window-secs 5 \
        --stride-secs 2.5 \
        --mel-start 9 --mel-end 61 \
        --n-components 50
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from sklearn.decomposition import PCA
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

def sklearn_pca(X: np.ndarray, n_components: int):
    """
    PCA using sklearn on X (N, D).
    Returns:
        X_pca:     (N, n_components) projected data
        components:(n_components, D) principal axes (unit vectors)
        mean:      (D,) column means used for centring
        explained_variance_ratio: (n_components,)
    """
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    components = pca.components_  # (n_components, D)
    mean = pca.mean_              # (D,)
    evr = pca.explained_variance_ratio_  # (n_components,)

    return X_pca, components, mean, evr


# Main function:
def main():
    ap = argparse.ArgumentParser(description="Sliding-window PCA over NPZ log-mel files.")
    ap.add_argument("--input-root", required=True, help="Directory containing .npz feature files.")
    ap.add_argument("--output-root", required=True, help="Where to write PCA results.")
    ap.add_argument("--window-secs", type=float, default=5.0, help="Window length in seconds (default: 5).")
    ap.add_argument("--stride-secs", type=float, default=None, help="Stride between consecutive windows in seconds. (Default: non-overlapping).")
    ap.add_argument("--mel-start", type=int,   default=0, help="First mel bin to include (default: 0).")
    ap.add_argument("--mel-end", type=int, default=128,  help="Last mel bin to include. If 61 then 60 is the last (default: 128).")
    ap.add_argument("--n-components", type=int, default=50, help="Number of PCA components to keep.")
    ap.add_argument("--n-mels", type=int, default=128, help="Total mel bins in the NPZ files (default: 128).")
    ap.add_argument("--feature-key", default="feature", help="Key inside NPZ files (default: 'feature').")
    ap.add_argument("--single-file", default=None, help="Process only a specific file. Provide name of that file.")
    ap.add_argument("--no-plot", action="store_true", help="Skip saving plots.")
    args = ap.parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    window_secs = args.window_secs
    stride_secs = args.stride_secs if args.stride_secs is not None else window_secs
    n_mels = args.n_mels
    mel_start = args.mel_start
    mel_end = args.mel_end

    # Derive time-frame counts from spectrogram config.
    spec_config = configs.get_specgram_config()
    hop_length = spec_config["hop_length"]      # 512 samples
    audio_sr = spec_config["sample_rate"]     # 64 000 Hz
    secs_per_frame = hop_length / audio_sr    # 

    window_frames = max(1, round(window_secs / secs_per_frame))
    stride_frames = max(1, round(stride_secs / secs_per_frame))

    print(f"Input root: {input_root}")
    print(f"Output root: {output_root}")
    print(f"Window: {window_secs:.2f} s, {window_frames} frames")
    print(f"Stride: {stride_secs:.2f} s, {stride_frames} frames")
    print(f"Mel bins: [{mel_start}, {mel_end})  ({mel_end - mel_start} bins)")
    print(f"n_components: {args.n_components}")

    # Collect features from all windows from all files.
    npz_files = sorted(input_root.rglob("*.npz"))
    # Skip combined array files that don't hold single-recording spectrograms.
    npz_files = [p for p in npz_files if args.feature_key in np.load(p, allow_pickle=True).files]

    if args.single_file:
        npz_files = [p for p in npz_files if p.name == args.single_file] # only when single file is specified

    if not npz_files:
        print(f"ERROR, No NPZ files with key '{args.feature_key}' found under {input_root}")
        sys.exit(1)

    print(f"\nFiles found: {len(npz_files)}")

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
            feat = window_feature(win)
            feature_rows.append(feat)
            window_meta.append({
                "file":        npz_path.name,
                "start_frame": int(start_frame),
                "start_sec":   round(start_frame * secs_per_frame, 3),
            })
            n_windows += 1
        if (i+1) % 10 == 0:
            print(f"  {i+1}/{len(npz_files)} processed")
        # print(f"  {npz_path.name}: T={S.shape[1]} frames, {n_windows} windows")

    if not feature_rows:
        print("[error] No windows extracted. Check window/stride settings vs recording length.")
        sys.exit(1)

    X = np.stack(feature_rows, axis=0)
    print(f"\nFeature matrix: {X.shape}  (windows × features)")

    # PCA with numpy
    n_components = min(args.n_components, X.shape[0], X.shape[1])
    print(f"Running PCA  ({n_components} components)...")

    X_pca, components, pca_mean, evr = numpy_pca(X, n_components)

    cumulative_evr = np.cumsum(evr)
    print(f"Variance explained: "
          f"PC1={evr[0]*100:.1f}%  "
          f"PC1+PC2={cumulative_evr[1]*100:.1f}%  "
          f"Top-{n_components}={cumulative_evr[-1]*100:.1f}%")

    # Save PCA results and metadata
    out_npz = output_root / "pca_results.npz"
    np.savez_compressed(
        out_npz,
        X_pca = X_pca,
        components = components,
        pca_mean = pca_mean,
        evr = evr,
        cumulative_evr = cumulative_evr,
        X_raw = X,
    )

    meta = {
        "created":       datetime.now().isoformat(),
        "input_root":    str(input_root),
        "window_secs":   window_secs,
        "stride_secs":   stride_secs,
        "window_frames": window_frames,
        "stride_frames": stride_frames,
        "mel_start":     mel_start,
        "mel_end":       mel_end,
        "n_mels":        n_mels,
        "n_components":  int(n_components),
        "n_windows":     int(X.shape[0]),
        "feature_dim":   int(X.shape[1]),
        "evr_top10":     [float(v) for v in evr[:10]],
        "files":         [p.name for p in npz_files],
    }
    
    with open(output_root / "pca_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Save window metadata (file + time offset per row).
    meta_npz = output_root / "window_index.npz"
    files_arr = np.array([m["file"] for m in window_meta], dtype=object)
    starts_arr = np.array([m["start_frame"] for m in window_meta], dtype=np.int64)
    secs_arr = np.array([m["start_sec"] for m in window_meta], dtype=np.float32)
    np.savez_compressed(meta_npz, files=files_arr, start_frames=starts_arr, start_secs=secs_arr)

    print(f"\nSaved -> {out_npz}")
    print(f"Saved -> {meta_npz}")
    print(f"Saved -> {output_root / 'pca_meta.json'}")

    # Save plots of pca

    if not args.no_plot:
        plot_path = output_root / "pca_plot.png"
        if len(npz_files) == 1:
            putils.plot_pca_projection_single(X_pca, evr, window_meta, plot_path)
        else:
            putils.plot_pca_projection(X_pca, evr, window_meta, plot_path)
        print(f"Saved -> {plot_path}")

    print("\n[done]")


if __name__ == "__main__":
    main()
