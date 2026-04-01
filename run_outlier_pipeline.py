"""
Full pipeline: audio extraction -> PCA -> outlier detection

This script runs the complete outlier detection pipeline:
1. Extract mel spectrograms from raw audio files
2. Run PCA on sliding windows
3. Find and visualize outliers

Example usage:
    python run_outlier_pipeline.py \
        --audio-root data/subsetWithValidatedCalls \
        --output-root output/pipeline_results \
        --window-secs 5 \
        --mel-start 9 --mel-end 61 \
        --n-components 20 \
        --pca-method mean_std \
        --distance-metric mahalanobis \
        --threshold-std 3 


To skip steps that have already been run:
    python run_outlier_pipeline.py \
        --audio-root data/subsetWithValidatedCalls \
        --output-root output/pipeline_results \
        --skip-extraction \
        --skip-pca \
        --threshold-std 3
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description="Run full outlier detection pipeline: extraction -> PCA -> outliers")

    # Input/output paths
    ap.add_argument("--audio-root", required=True, help="Directory containing raw .wav audio files.")
    ap.add_argument("--output-root", required=True, help="Base output directory for all pipeline results.")

    # Shared parameters
    ap.add_argument("--window-secs", type=float, default=5.0, help="Window length in seconds (default: 5.0).")
    ap.add_argument("--stride-secs", type=float, default=None, help="Stride between windows in seconds (default: non-overlapping).")
    ap.add_argument("--mel-start", type=int, default=None, help="First mel bin to include (default: 0).")
    ap.add_argument("--mel-end", type=int, default=None, help="Last mel bin exclusive (default: all).")

    # PCA parameters
    ap.add_argument("--n-components", type=int, default=10, help="Number of PCA components (default: 10).")
    ap.add_argument("--pca-method", choices=["mean_std", "full_window"], default="mean_std", help="Feature type for PCA (default: mean_std).")

    # Outlier detection parameters
    ap.add_argument("--distance-metric", choices=["euclidean", "mahalanobis"], default="mahalanobis", help="Distance metric for outlier detection (default: mahalanobis).")
    ap.add_argument("--threshold-std", type=float, default=3.0, help="Number of std deviations for outlier threshold (default: 3.0).")

    # Skip flags
    ap.add_argument("--skip-extraction", action="store_true", help="Skip spectrogram extraction (use existing .npz files).")
    ap.add_argument("--skip-pca", action="store_true", help="Skip PCA (use existing pca_results.npz).")
    ap.add_argument("--no-plot", action="store_true", help="Skip plotting in all steps.")
    ap.add_argument("--no-audio-clips", action="store_true", help="Skip saving audio clips for outliers.")

    # Subset flag
    ap.add_argument("--subset-len", type=int, default=0, help="Limit to first N audio files (for testing).")

    args = ap.parse_args()

    # Setup paths
    audio_root = Path(args.audio_root)
    output_root = Path(args.output_root)
    npz_root = output_root / "npz"
    pca_root = output_root / "pca"
    outliers_root = output_root / "outliers"

    output_root.mkdir(parents=True, exist_ok=True)

    print(f"Pipeline Configuration:")
    print(f"  Audio root:   {audio_root}")
    print(f"  Output root:  {output_root}")
    print(f"  Window:       {args.window_secs}s")
    print(f"  Mel bins:     {args.mel_start or 0} to {args.mel_end or 'all'}")
    print(f"  PCA components: {args.n_components}")
    print(f"  Outlier threshold: {args.threshold_std} std ({args.distance_metric})")

    # Step 1: Extract spectrograms
    if not args.skip_extraction:
        cmd = [
            sys.executable, "preprocessing/run_extraction_noref.py",
            "--input-root", str(audio_root),
            "--output-root", str(npz_root),
        ]
        if args.subset_len > 0:
            cmd.extend(["--subset-len", str(args.subset_len)])


        print(f"\n\n\n\n{'='*60}")
        print(f"\nSTEP: Spectrogram Extraction to NPZ files")
        print(f"{'='*60}")
        print(f"\nRunning: {' '.join(cmd)}\n")

        result = subprocess.run(cmd, cwd=Path(__file__).parent)
        if result.returncode != 0:
            print(f"\nError: Spectrogram Extraction failed with return code {result.returncode}")
            sys.exit(result.returncode)

        print(f"\n[Done] Completed: Spectrogram Extraction")
    else:
        print("\nSkipping spectrogram extraction (--skip-extraction)")

    # Step 2: Run PCA
    if not args.skip_pca:
        cmd = [
            sys.executable, "analysis/pca_sliding_window.py",
            "--input-root", str(npz_root),
            "--output-root", str(pca_root),
            "--window-secs", str(args.window_secs),
            "--n-components", str(args.n_components),
            "--pca-method", args.pca_method,
        ]
        if args.stride_secs is not None:
            cmd.extend(["--stride-secs", str(args.stride_secs)])
        if args.mel_start is not None:
            cmd.extend(["--mel-start", str(args.mel_start)])
        if args.mel_end is not None:
            cmd.extend(["--mel-end", str(args.mel_end)])
        if args.no_plot:
            cmd.append("--no-plot")

        print(f"\n\n\n\n{'='*60}")
        print(f"\nSTEP: PCA Analysis")
        print(f"{'='*60}")
        print(f"\nRunning: {' '.join(cmd)}\n")

        result = subprocess.run(cmd, cwd=Path(__file__).parent)
        if result.returncode != 0:
            print(f"\nError: PCA Analysis failed with return code {result.returncode}")
            sys.exit(result.returncode)

        print(f"\n[Done] Completed: PCA Analysis")
    else:
        print("\nSkipping PCA (--skip-pca)")

    # Step 3: Find outliers
    cmd = [
        sys.executable, "analysis/finding_outliers.py",
        "--pca-root", str(pca_root),
        "--npz-root", str(npz_root),
        "--plots-root", str(outliers_root),
        "--window-secs", str(args.window_secs),
        "--distance-metric", args.distance_metric,
        "--threshold-std", str(args.threshold_std),
        "--save-csv",
    ]
    if args.mel_start is not None:
        cmd.extend(["--mel-start", str(args.mel_start)])
    if args.mel_end is not None:
        cmd.extend(["--mel-end", str(args.mel_end)])
    if not args.no_audio_clips:
        cmd.extend(["--audio-root", str(audio_root)])
    if args.no_plot:
        cmd.append("--no-plot")


    print(f"\n\n\n\n{'='*60}")
    print(f"\nSTEP: Outlier Detection")
    print(f"{'='*60}")
    print(f"\nRunning: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    if result.returncode != 0:
        print(f"\nError: Outlier Detection failed with return code {result.returncode}")
        sys.exit(result.returncode)

    print(f"\n[Done] Completed: Outlier Detection")


    # Summary
    print(f"\n\n\n\n{'='*60}")
    print("PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"Results saved to: {output_root}")
    print(f"  - Spectrograms: {npz_root}")
    print(f"  - PCA results:  {pca_root}")
    print(f"  - Outliers:     {outliers_root}")


if __name__ == "__main__":
    main()
