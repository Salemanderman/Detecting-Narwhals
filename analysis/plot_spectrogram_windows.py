"""
Sliding-window spectrogram plots over log-mel NPZ feature files.

Each NPZ file holds a long recording as a log-mel spectrogram (1, n_mels, T).
A sliding window walks along the time axis; each window is plotted as a
log-mel spectrogram image.  One grid PNG is saved per input file.

Example usage:
    python analysis/plot_spectrogram_windows.py \
        --npz-root  testRunExtractionNoref/Aug_6229 \
        --output-root analysis/spectrogram_windows \
        --window-secs 5 \
        --stride-secs 5 \
        --mel-start 9 --mel-end 61 \
        --max-per-file 20 \
        --cols 4

Or for a single file:
    python analysis/plot_spectrogram_windows.py \
        --npz-root  testRunExtractionNoref/Aug_6229 \
        --output-root analysis/spectrogram_windows \
        --single-file recording_123.npz \
        --window-secs 5 \
        --stride-secs 5 \
        --mel-start 9 --mel-end 61 \
        --max-per-file 20 \
        --cols 4
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import utilities.configs as configs
import utilities.feature_utils as futils
from utilities.plot_utils import *


def plot_window_grid(
    windows,              # list of (start_sec, np.ndarray shape (n_bins, window_frames))
    title: str,
    secs_per_frame: float,
    cols: int,
    output_path: Path,
    mel_start: int = 0,
    cmap: str = "viridis",
):
    """Save a grid of spectrogram subplots, one cell per window."""
    n = len(windows)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(
        rows, cols,
        figsize=(cols * 3, rows * 2.2),
        squeeze=False,
    )
    fig.suptitle(title, fontsize=10, y=1.01)

    for idx, (start_frame, win) in enumerate(windows):
        r, c = divmod(idx, cols)
        ax = axes[r][c]

        n_bins = win.shape[0]
        start_sec = start_frame * secs_per_frame
        end_sec = start_sec + win.shape[1] * secs_per_frame
        im = ax.imshow(
            win,
            origin="lower",
            aspect="auto",
            cmap=cmap,
            interpolation="nearest",
        )
        ax.set_title(f"{start_sec:.1f}-{end_sec:.1f} s", fontsize=7)
        ax.set_xlabel("Frame", fontsize=6)
        ax.set_ylabel("Mel bin", fontsize=6)

        # Show original mel bin indices on y-axis.
        tick_positions = np.linspace(0, n_bins - 1, min(5, n_bins))
        tick_labels    = [str(int(round(mel_start + p))) for p in tick_positions]
        ax.set_yticks(tick_positions)
        ax.set_yticklabels(tick_labels)

        ax.tick_params(labelsize=5)
        plt.colorbar(im, ax=ax, pad=0.02, fraction=0.04).ax.tick_params(labelsize=5)

    # Hide unused cells.
    for idx in range(n, rows * cols):
        r, c = divmod(idx, cols)
        axes[r][c].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)



def main():
    ap = argparse.ArgumentParser(description="Sliding-window spectrogram plots from NPZ files.")
    ap.add_argument("--npz-root",   required=True, help="Directory tree containing .npz feature files.")
    ap.add_argument("--output-root",  required=True, help="Where to write plot images.")
    ap.add_argument("--window-secs",  type=float, default=5.0,  help="Window length in seconds (default: 5).")
    ap.add_argument("--stride-secs",  type=float, default=None,
                    help="Stride between windows in seconds. Defaults to window-secs (non-overlapping).")
    ap.add_argument("--mel-start",    type=int,   default=0,    help="First mel bin to include (default: 0).")
    ap.add_argument("--mel-end",      type=int,   default=None,  help="Last mel bin (exclusive) to include (default: 128).")
    ap.add_argument("--cols",         type=int,   default=2,    help="Grid columns per plot (default: 4).")
    ap.add_argument("--n-mels",       type=int,   default=None, help="Total mel bins in the NPZ files. If None, auto-detect from spectrogram shape.")
    ap.add_argument("--feature-key",  default="feature",        help="Key inside NPZ files (default: 'feature').")
    ap.add_argument("--single-file",  default=None,             help="Process only a specific file (by name).")
    ap.add_argument("--cmap",         default="viridis",         help="Matplotlib colormap (default: inferno).")
    args = ap.parse_args()

    npz_root  = Path(args.npz_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    window_secs = args.window_secs
    stride_secs = args.stride_secs if args.stride_secs is not None else window_secs
    n_mels      = args.n_mels
    mel_start   = args.mel_start
    mel_end     = args.mel_end

    # Derive time-frame counts from spectrogram config.
    spec_cfg       = configs.get_specgram_config()
    hop_length     = spec_cfg["hop_length"]      # 512 samples
    audio_sr       = spec_cfg["sample_rate"]     # 64 000 Hz
    secs_per_frame = hop_length / audio_sr       # ~0.008 s / frame

    window_frames = max(1, round(window_secs / secs_per_frame))
    stride_frames = max(1, round(stride_secs / secs_per_frame))

    print(f"NPZ root:    {npz_root}")
    print(f"Output root:   {output_root}")
    print(f"Window:        {window_secs:.2f} s  ->  {window_frames} frames")
    print(f"Stride:        {stride_secs:.2f} s  ->  {stride_frames} frames")
    print(f"Mel bins:      [start: {mel_start}, end: {mel_end})  ")
    print(f"Grid columns:  {args.cols}")

    # Load all NPZ files
    npz_files = sorted(npz_root.rglob("*.npz"))
    npz_files = [p for p in npz_files if args.feature_key in np.load(p, allow_pickle=True).files]

    if args.single_file:
        npz_files = [p for p in npz_files if p.name == args.single_file]

    if not npz_files:
        print(f"[error] No NPZ files with key '{args.feature_key}' found under {npz_root}")
        sys.exit(1)

    print(f"\nFiles found: {len(npz_files)}")


    for npz_path in npz_files:
        try:
            S, sr = futils.load_spectrogram(npz_path, n_mels=n_mels, key=args.feature_key)
        except Exception as e:
            print(f"  [skip] {npz_path.name}: {e}")
            continue

        # Collect all windows for this file.
        all_windows = [
            (start_frame, win)
            for start_frame, win in futils.windows_from_spectrogram(S, window_frames, stride_frames,
                                                              mel_start=mel_start, mel_end=mel_end)
        ]

        if not all_windows:
            print(f"  [skip] {npz_path.name}: recording too short for one window "
                  f"(T={S.shape[1]} frames, need {window_frames})")
            continue

        # Optionally limit how many windows to plot.

        out_path = output_root / (npz_path.stem + "_windows.png")
        plot_window_grid(
            windows = all_windows,
            title = f"{npz_path.name} " f"({window_secs:.1f} s windows, mel start: {mel_start}, end: {mel_end-1 if mel_end is not None else 'None'} )",
            secs_per_frame = secs_per_frame,
            cols           = args.cols,
            output_path    = out_path,
            mel_start      = mel_start,
            cmap           = args.cmap,
        )
        print(f"  {npz_path.name}: {len(all_windows)} windows total, "
              f"plotted {len(all_windows)}  ->  {out_path.name}")

    print("\n[done]")


if __name__ == "__main__":
    main()
