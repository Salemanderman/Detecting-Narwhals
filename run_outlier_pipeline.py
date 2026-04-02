"""
Full pipeline: audio extraction -> PCA -> outlier detection

This script runs the complete outlier detection pipeline:
1. Extract mel spectrograms from raw audio files
2. Run PCA on sliding windows
3. Find and visualize outliers

Example usage:
    # Run with default config (from pipeline_config.py):
    python run_outlier_pipeline.py

    # Override specific parameters:
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
        --skip-extraction \
        --skip-pca \
        --threshold-std 3
"""

import argparse
import subprocess
import sys
from pathlib import Path

try:
    import pipeline_config as config
except ImportError:
    config = None


def main():
    if config:
        cfg = config.get_pipeline_config()
    else:
        cfg = {
            'audio_root': None,
            'output_root': None,
            'window_secs': 5.0,
            'stride_secs': None,
            'mel_start': None,
            'mel_end': None,
            'n_components': 10,
            'pca_method': 'mean_std',
            'distance_metric': 'mahalanobis',
            'threshold_std': 3.0,
            'skip_extraction': False,
            'skip_pca': False,
            'no_plot': False,
            'no_audio_clips': False,
            'subset_len': 0,
        }

    ap = argparse.ArgumentParser(description="Run full outlier detection pipeline: extraction -> PCA -> outliers")
    ap.add_argument("--audio-root", default=cfg['audio_root'], required=config is None, help="Directory containing raw .wav audio files.")
    ap.add_argument("--output-root", default=cfg['output_root'], required=config is None, help="Base output directory for all pipeline results.")
    ap.add_argument("--window-secs", type=float, default=cfg['window_secs'], help=f"Window length in seconds (default: {cfg['window_secs']}).")
    ap.add_argument("--stride-secs", type=float, default=cfg['stride_secs'], help="Stride between windows in seconds (default: non-overlapping).")
    ap.add_argument("--mel-start", type=int, default=cfg['mel_start'], help=f"First mel bin to include (default: {cfg['mel_start'] or 0}).")
    ap.add_argument("--mel-end", type=int, default=cfg['mel_end'], help=f"Last mel bin exclusive (default: {cfg['mel_end'] or 'all'}).")
    ap.add_argument("--n-components", type=int, default=cfg['n_components'], help=f"Number of PCA components (default: {cfg['n_components']}).")
    ap.add_argument("--pca-method", choices=["mean_std", "full_window"], default=cfg['pca_method'], help=f"Feature type for PCA (default: {cfg['pca_method']}).")
    ap.add_argument("--distance-metric", choices=["euclidean", "mahalanobis"], default=cfg['distance_metric'], help=f"Distance metric for outlier detection (default: {cfg['distance_metric']}).")
    ap.add_argument("--threshold-std", type=float, default=cfg['threshold_std'], help=f"Number of std deviations for outlier threshold (default: {cfg['threshold_std']}).")
    ap.add_argument("--skip-extraction", action="store_true", default=cfg['skip_extraction'], help="Skip spectrogram extraction (use existing .npz files).")
    ap.add_argument("--skip-pca", action="store_true", default=cfg['skip_pca'], help="Skip PCA (use existing pca_results.npz).")
    ap.add_argument("--no-plot", action="store_true", default=cfg['no_plot'], help="Skip plotting in all steps.")
    ap.add_argument("--no-audio-clips", action="store_true", default=cfg['no_audio_clips'], help="Skip saving audio clips for outliers.")
    ap.add_argument("--subset-len", type=int, default=cfg['subset_len'], help=f"Limit to first N audio files (for testing) (default: {cfg['subset_len']}).")

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
