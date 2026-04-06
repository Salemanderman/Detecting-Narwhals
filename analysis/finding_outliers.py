"""
Finding Outliers in PCA Plot

This script loads the PCA results and identifies outliers based on distance
from the center of the PCA space.

Use for example:
    python analysis/finding_outliers.py \
        --pca-root subsetWithValidatedCalls/pca_output \
        --npz-root subsetWithValidatedCalls/npzFiles \
        --distance-metric mahalanobis \
        --threshold-std 3 \
        --plots-root analysis/outlier_plots \
        --save-csv \
        --mel-start 9 \
        --mel-end 61

Or with euclidean distance and saving results:
    python analysis/finding_outliers.py \
        --pca-root subsetWithValidatedCalls/pca_output \
        --npz-root subsetWithValidatedCalls/npzFiles \
        --distance-metric euclidean \
        --threshold-std 2.5 \
        --save-csv \
        --mel-start 9 \
        --mel-end 61
"""

import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import sys
from scipy.io import wavfile

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from utilities import feature_utils as futils
from utilities import configs


def compute_distances(X_pca: np.ndarray, metric: str = "mahalanobis"):
    """
    Compute distances from the mean in PCA space.

    Args:
        X_pca: PCA-transformed data (N, n_components)
        metric: "euclidean" or "mahalanobis"

    Returns:
        distances: array of distances for each point
        mean_pca: mean of the PCA data
    """
    mean_pca = X_pca.mean(axis=0)

    if metric == "euclidean":
        distances = np.sqrt(np.sum((X_pca - mean_pca) ** 2, axis=1))
    elif metric == "mahalanobis":
        cov_pca = np.cov(X_pca, rowvar=False)
        inv_cov = np.linalg.inv(cov_pca)
        distances = np.array([
            np.sqrt((X_pca[i] - mean_pca).T @ inv_cov @ (X_pca[i] - mean_pca))
            for i in range(X_pca.shape[0])
        ])
    else:
        raise ValueError(f"Unknown metric: {metric}. Use 'euclidean' or 'mahalanobis'.")

    return distances, mean_pca


def find_outliers(distances: np.ndarray, threshold_std: float = 3.0):
    """
    Find outliers based on distance threshold.

    Args:
        distances: array of distances
        threshold_std: number of standard deviations for threshold

    Returns:
        outlier_mask: boolean mask of outliers
        threshold: the computed threshold value
    """
    threshold = distances.mean() + threshold_std * distances.std()
    outlier_mask = distances > threshold
    return outlier_mask, threshold


def plot_pca_with_outliers(X_pca, evr, mean_pca, outlier_mask, save_path=None):
    """Plot PCA projection with outliers highlighted."""
    assert X_pca.shape[1] >= 2, "minimum 2 PCA components for plotting"
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot normal points
    normal_mask = ~outlier_mask
    ax.scatter(
        X_pca[normal_mask, 0],
        X_pca[normal_mask, 1],
        c="green",
        alpha=0.5,
        s=30,
        label=f"Normal ({normal_mask.sum()} points)",
    )

    # Plot outliers
    ax.scatter(
        X_pca[outlier_mask, 0],
        X_pca[outlier_mask, 1],
        c="red",
        alpha=0.8,
        s=50,
        label=f"Outliers ({outlier_mask.sum()} points)",
    )

    # Plot center
    ax.scatter(
        mean_pca[0],
        mean_pca[1],
        c="green",
        s=300,
        marker="*",
        label="Mean",
        zorder=5,
        edgecolors="black",
        linewidths=2,
    )

    ax.set_xlabel(f"PC1 ({evr[0]*100:.1f}% variance)", fontsize=12)
    ax.set_ylabel(f"PC2 ({evr[1]*100:.1f}% variance)", fontsize=12)
    ax.set_title("PCA Projection with Outliers marked", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved PCA plot to: {save_path}")
    else:
        plt.show()


def save_outlier_spectrogram(row, npz_root, window_frames, spec_cfg, save_path,
                             mel_start=None, mel_end=None):
    """Save a single outlier spectrogram for a single window in a single file to file."""
    npz_path = npz_root / row["File"]
    start_sec = row["Start Time (s)"]

    window = futils.get_window(npz_path, start_sec, window_frames,
                               mel_start=mel_start, mel_end=mel_end, spec_cfg=spec_cfg)

    n_bins = window.shape[0]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.imshow(window, aspect="auto", origin="lower", cmap="viridis")
    ax.set_title(f"{row['File']}\nt={start_sec:.2f}s, distance={row['Distance']:.2f}", fontsize=10)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Mel bin")

    #adjust y-axis ticks to show mel frequencies
    tick_positions = np.linspace(0, n_bins - 1, min(5, n_bins))
    tick_labels = [str(int(round(mel_start + p))) for p in tick_positions]
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_file_outliers_grid(filename, file_outliers_df, npz_root, window_frames, spec_cfg,
                            save_path, mel_start=None, mel_end=None):
    """
    Save a grid plot of all outliers from a single source file.

    Args:
        filename: The source .npz filename
        file_outliers_df: DataFrame subset containing only outliers from this file
        npz_root: Root directory for .npz files
        window_frames: Number of frames per window
        spec_cfg: Spectrogram config dict
        save_path: Path to save the output image
        mel_start: First mel bin (default: 0)
        mel_end: Last mel bin exclusive (default: all)
    """
    n_outliers = len(file_outliers_df)
    if n_outliers == 0:
        return

    npz_path = npz_root / filename

    # Grid layout
    n_cols = min(4, n_outliers)
    n_rows = (n_outliers + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    if n_outliers == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes).flatten()

    for i, (_, row) in enumerate(file_outliers_df.iterrows()):
        start_sec = row["Start Time (s)"]
        try:
            window = futils.get_window(npz_path, start_sec, window_frames,
                                       mel_start=mel_start, mel_end=mel_end, spec_cfg=spec_cfg)
            ax = axes[i]
            ax.imshow(window, aspect="auto", origin="lower", cmap="viridis")
            ax.set_title(f"t={start_sec:.2f}s, d={row['Distance']:.2f}", fontsize=9)
            ax.set_xlabel("Frame", fontsize=8)
            ax.set_ylabel("Mel bin", fontsize=8)

            # Y-axis ticks for mel bins
            if mel_start is not None:
                n_bins = window.shape[0]
                tick_positions = np.linspace(0, n_bins - 1, min(4, n_bins))
                tick_labels = [str(int(round(mel_start + p))) for p in tick_positions]
                ax.set_yticks(tick_positions)
                ax.set_yticklabels(tick_labels, fontsize=7)
        except Exception as e:
            axes[i].set_title(f"Error: {e}", fontsize=8)
            axes[i].axis("off")

    # Hide unused subplots
    for j in range(n_outliers, len(axes)):
        axes[j].axis("off")

    file_stem = Path(filename).stem
    fig.suptitle(f"Outliers in {file_stem} ({n_outliers} windows)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_audio_clip(audio_path, start_sec, window_sec, save_path):
    """
    Extract and save an audio clip from a wav file.

    Args:
        audio_path: Path to source .wav file
        start_sec: Start time in seconds
        window_sec: Window length in seconds
        save_path: Path to save the output .wav file
    """
    sr, audio = wavfile.read(audio_path)
    start_sample = int((start_sec + 5) * sr) # + 5 since first 5 seconds are cut off in npz files
    end_sample = int(((start_sec + 5) + window_sec) * sr)

    # Clamp to valid range
    start_sample = max(0, start_sample)
    end_sample = min(len(audio), end_sample)

    clip = audio[start_sample:end_sample]
    wavfile.write(save_path, sr, clip)


def main():
    ap = argparse.ArgumentParser(description="Find outliers in PCA results based on distance from center.")
    ap.add_argument("--pca-root", required=True, help="Directory containing pca_results.npz from pca_sliding_window.py.")
    ap.add_argument("--npz-root", required=True, help="Directory containing the original .npz spectrogram files.")
    ap.add_argument("--distance-metric", choices=["euclidean", "mahalanobis"], default="mahalanobis",
                    help="Distance metric for outlier detection (default: mahalanobis).")
    ap.add_argument("--threshold-std", type=float, default=3.0,
                    help="Number of std deviations for outlier threshold (default: 3.0).")
    ap.add_argument("--window-secs", type=float, default=5.0,
                    help="Window length in seconds for spectrogram plots (default: 5.0).")
    ap.add_argument("--mel-start", type=int, default=None,
                    help="First mel bin to include in spectrogram plots (default: 0).")
    ap.add_argument("--mel-end", type=int, default=None,
                    help="Last mel bin (exclusive) to include in spectrogram plots (default: all).")
    ap.add_argument("--save-csv", action="store_true",
                    help="Save outlier table to CSV file.")
    ap.add_argument("--plots-root", type=str, required=True,
                    help="Directory to save plots. If not provided, plots will be shown instead of saved")
    ap.add_argument("--audio-root", type=str, default=None,
                    help="Directory containing raw .wav files. If provided, audio clips will be saved for each outlier.")
    ap.add_argument("--no-plot", action="store_true",
                    help="Skip all plotting.")
    args = ap.parse_args()

    pca_output_root = Path(args.pca_root)
    npz_root = Path(args.npz_root)
    plot_output_root = Path(args.plots_root)
    plot_output_root.mkdir(parents=True, exist_ok=True)

    print(f"PCA output: {pca_output_root}")
    print(f"Spectrogram root: {npz_root}")
    print(f"Distance metric: {args.distance_metric}")
    print(f"Threshold: {args.threshold_std} std deviations")

    # List files in PCA output directory
    print(f"\nFiles found at {pca_output_root}:")
    for f in pca_output_root.glob("*"):
        print(f"  - {f.name}")

    # Load PCA results
    pca_data = np.load(pca_output_root / "pca_results.npz", allow_pickle=True)
    X_pca = pca_data["X_pca"]
    evr = pca_data["evr"]
    files = pca_data["window_files"]
    start_frames = pca_data["window_start_frames"]
    start_secs = pca_data["window_start_secs"]

    print(f"\nX_pca shape: {X_pca.shape}")
    print(f"{len(files)} windows")

    # Compute distances
    distances, mean_pca = compute_distances(X_pca, metric=args.distance_metric)

    print(f"\n{args.distance_metric.capitalize()} distance stats:")
    print(f"  Mean: {distances.mean():.3f}, Std: {distances.std():.3f}")
    print(f"  Min: {distances.min():.3f}, Max: {distances.max():.3f}")

    # Find outliers
    outlier_mask, threshold = find_outliers(distances, threshold_std=args.threshold_std)
    outlier_indices = np.asarray(outlier_mask).nonzero()[0]

    print(f"\nThreshold: {threshold:.3f}")
    print(f"Number of outliers: {len(outlier_indices)} out of {len(distances)} windows")

    # Create outlier dataframe
    outlier_info = []
    for idx in outlier_indices:
        outlier_info.append({
            "Index": idx,
            "File": files[idx],
            "Start Frame": int(start_frames[idx]),
            "Start Time (s)": float(start_secs[idx]),
            "Distance": distances[idx],
            "PC1": X_pca[idx, 0],
            "PC2": X_pca[idx, 1] if X_pca.shape[1] > 1 else 0,
        })

    outlier_df = pd.DataFrame(outlier_info)
    if not outlier_df.empty:
        outlier_df = outlier_df.sort_values(["File", "Start Time (s)"], ascending=[True, True])

        # Group outliers by source file
        outliers_by_file = outlier_df.groupby("File").size().sort_values(ascending=False)

    # Save CSV if requested
    if args.save_csv and not outlier_df.empty:
        csv_path = plot_output_root / "outliers.csv"
        outlier_df.to_csv(csv_path, index=False)
        print(f"\nSaved outlier table to: {csv_path}")

        # Write summary report to text file
        report_path = plot_output_root / "outliers_report.txt"
        with open(report_path, "w") as f:
            f.write("OUTLIER DETECTION REPORT\n\n\n")
            f.write(f"Distance metric: {args.distance_metric}\n")
            f.write(f"Threshold: {threshold:.3f} ({args.threshold_std} std deviations)\n")
            f.write(f"Number of outliers: {len(outlier_indices)} out of {len(distances)} windows\n\n")
            f.write(f"{args.distance_metric} distance stats:\n")
            f.write(f"  Mean: {distances.mean():.3f}, Std: {distances.std():.3f}\n")
            f.write(f"  Min: {distances.min():.3f}, Max: {distances.max():.3f}\n\n")
            f.write("OUTLIERS PER FILE\n")
            f.write("-" * 40 + "\n")
            f.write(outliers_by_file.to_string() + "\n\n")
            f.write("OUTLIER DETAILS\n")
            f.write("-" * 40 + "\n")
            f.write(outlier_df.to_string() + "\n")
        print(f"Saved report to: {report_path}")

    # Plotting
    if not args.no_plot:
        # Settings for window extraction
        spec_cfg = configs.get_specgram_config()
        secs_per_frame = spec_cfg["hop_length"] / spec_cfg["sample_rate"]
        window_frames = max(1, round(args.window_secs / secs_per_frame))

        pca_plot_path = plot_output_root / "outliers_pca_plot.png"
        plot_pca_with_outliers(X_pca, evr, mean_pca, outlier_mask, save_path=pca_plot_path)

        if outlier_df.empty:
            print("\nNo outliers found, skipping spectrogram plotting.")
        else:
            outliers_root = plot_output_root / "outliers"
            outliers_root.mkdir(parents=True, exist_ok=True)

            # Group outliers by source file and save one grid plot per file
            grouped = outlier_df.groupby("File")
            print(f"\nSaving outlier grids for {len(grouped)} files to {outliers_root}")

            for filename, file_outliers in grouped:
                file_stem = Path(filename).stem

                # Create directory for this file
                file_dir = outliers_root / file_stem
                file_dir.mkdir(parents=True, exist_ok=True)

                # Save grid plot in the file's directory
                save_path = file_dir / f"{file_stem}_grid.png"
                try:
                    save_file_outliers_grid(
                        filename, file_outliers, npz_root, window_frames, spec_cfg,
                        save_path, mel_start=args.mel_start, mel_end=args.mel_end
                    )
                except Exception as e:
                    print(f"  Error saving {file_stem}: {e}")

            print(f"  Saved {len(grouped)} grid plots")

    # Save audio clips if audio_root is provided
    if args.audio_root and not outlier_df.empty:
        audio_root = Path(args.audio_root)
        outliers_root = plot_output_root / "outliers"
        outliers_root.mkdir(parents=True, exist_ok=True)

        # Group outliers by source file
        grouped = outlier_df.groupby("File")
        print(f"\nSaving audio clips for {len(grouped)} files to {outliers_root}")

        total_saved = 0
        for filename, file_outliers in grouped:
            file_stem = Path(filename).stem

            # Create directory for this file (will already exist if plots were made)
            file_dir = outliers_root / file_stem
            file_dir.mkdir(parents=True, exist_ok=True)

            # Convert .npz filename to .wav
            wav_filename = Path(filename).stem + ".wav"
            audio_path = audio_root / wav_filename

            if not audio_path.exists():
                print(f"  Warning: Audio file not found: {audio_path}")
                continue

            # Save audio clip for each outlier from this file
            for _, row in file_outliers.iterrows():
                start_sec = row["Start Time (s)"]
                clip_filename = f"{file_stem}_t{start_sec:.2f}s.wav"
                save_path = file_dir / clip_filename

                try:
                    save_audio_clip(audio_path, start_sec, args.window_secs, save_path)
                    total_saved += 1
                except Exception as e:
                    print(f"  Error saving {clip_filename}: {e}")

        print(f"  Saved {total_saved} audio clips")

    print("\nDone.")


if __name__ == "__main__":
    main()
