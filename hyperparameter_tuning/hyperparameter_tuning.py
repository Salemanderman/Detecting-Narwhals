'''
Hyperparameter tuning for outlier detection

Example usage:
    python hyperparameter_tuning/hyperparameter_tuning.py \
        --audio-root data/subsetWithValidatedCalls \
        --npz-root subsetWithValidatedCalls/npzFiles \
        --validation-csv evaluation/validatedChristerCalls.csv \
        --output-root output/tuning_results_test
'''

import argparse
import subprocess
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from itertools import product


PARAM_GRID = {
    'n_components':[21, 34], 
    'mel_start': [9],
    'mel_end': [None],
    'window_secs': [5], #[2.0, 5.0, 6.5],
    'threshold_std':[3], #[3.5, 4.5, 5.5],
    'distance_metric': ['mahalanobis'],
    'pca_method': ['mean_std'],
}


def run_pipeline(config, audio_root, npz_root, output_root, skip_extraction=False):
    """Run the outlier detection pipeline with given configuration."""
    cmd = [
        sys.executable, "run_outlier_pipeline.py",
        "--audio-root", str(audio_root),
        "--npz-root", str(npz_root),
        "--output-root", str(output_root),
        "--window-secs", str(config['window_secs']),
        "--n-components", str(config['n_components']),
        "--pca-method", config['pca_method'],
        "--distance-metric", config['distance_metric'],
        "--threshold-std", str(config['threshold_std']),
        "--no-audio-clips",
    ]


    if skip_extraction:
        cmd.append("--skip-extraction")
    if config['mel_start'] is not None:
        cmd.extend(["--mel-start", str(config['mel_start'])])
    if config['mel_end'] is not None:
        cmd.extend(["--mel-end", str(config['mel_end'])])


    root_dir = Path(__file__).parent.parent # Start from project directory (Detecting-Narwhals/)
    result = subprocess.run(cmd, cwd=root_dir, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  Error: {result.stderr}")

    return result.returncode

def evaluate_performance(outliers_csv, validation_csv, tolerance=5.0):
    """Evaluate outlier detection performance against validation calls."""
    try:
        outliers_df = pd.read_csv(outliers_csv)
        validation_df = pd.read_csv(validation_csv)
    except Exception as e:
        print(f"  Error loading CSVs: {e}")
        return {'recall': 0.0, 'precision': 0.0, 'f1': 0.0, 'n_outliers': 0, 'n_matched': 0, 'n_validation': 0}

    n_matched = 0
    for row in validation_df.itertuples(index=False):
        val_file = row.file
        val_time = row.start_sec
        time_match = np.abs(outliers_df["Start Time (s)"] - val_time) < tolerance
        file_match = outliers_df["File"] == val_file
        matches = outliers_df[time_match & file_match]

        if not matches.empty:
            n_matched += 1

    n_validation = len(validation_df)
    n_outliers = len(outliers_df)

    recall = n_matched / n_validation if n_validation > 0 else 0.0
    precision = n_matched / n_outliers if n_outliers > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'recall': recall,
        'precision': precision,
        'f1': f1,
        'n_outliers': n_outliers,
        'n_matched': n_matched,
        'n_validation': n_validation,
    }


def grid_search(audio_root, npz_root, validation_csv, output_root, tolerance=5.0, skip_extraction=False):
    """Perform grid search over pipeline hyperparameters."""
    results = []

    parameter_names = list(PARAM_GRID.keys()) # ['n_components', 'mel_start', 'mel_end',...]]
    parameter_values = [PARAM_GRID[param_name] for param_name in parameter_names] # [[16, 32, 52], [0, 9, 15], [61, None], ...]
    param_combinations = list(product(*parameter_values)) # [(16, 0, 61, 2.0, 3.5, 'mahalanobis', 'mean_std'), (32, 0, 61, 2.0, 3.5, 'mahalanobis', 'mean_std'), ...]

    total_runs = len(param_combinations)
    print(f"\nTotal configurations to test: {total_runs}")

    for run_num, pipeline_combo in enumerate(param_combinations, 1):
        config = dict(zip(parameter_names, pipeline_combo))

        config_parts = [] # Directory with configuration details
        for key, value in config.items():
            config_parts.append(f"{key}={value}")
        run_output = output_root / f"trial_{run_num:04d}"
        config_str = ", ".join(config_parts)

        # Run pipeline
        print(f"\n\n\n[{run_num}/{total_runs}]\n {config_str}")
        returncode = run_pipeline(config, audio_root, npz_root, run_output, skip_extraction=skip_extraction)

        # Evaluate
        if returncode == 0:
            outliers_csv = run_output / "outliers" / "outliers.csv"
            performance = evaluate_performance(outliers_csv, validation_csv, tolerance)
        else:
            print(f"  Pipeline failed, skipping evaluation for this configuration.")
            performance = {'recall': 0.0, 'precision': 0.0, 'f1': 0.0, 'n_outliers': 0, 'n_matched': 0, 'n_validation': 0}

        result = {
            'run_num': run_num,
            **config, # unpack all config parameters with the double star operator
            **performance, # unpack performance metrics with the double star operator
            'output_path': str(run_output),
        }
        results.append(result)

        print(f"Results: Recall={performance['recall']}, "
              f"Precision={performance['precision']}, "
              f"F1={performance['f1']}")
        print(f"Matched: {performance['n_matched']}/{performance['n_validation']} validation calls")
        print(f"Total outliers: {performance['n_outliers']}")

    return pd.DataFrame(results)


def main():
    ap = argparse.ArgumentParser(description="Hyperparameter tuning for outlier detection pipeline")
    ap.add_argument("--audio-root", required=True, help="Directory containing audio files")
    ap.add_argument("--npz-root", default=None, help="Directory for storing/loading .npz files")
    ap.add_argument("--validation-csv", required=True, help="Path to validation CSV with ground truth")
    ap.add_argument("--output-root", required=True, help="Root directory for tuning results")
    ap.add_argument("--tolerance", type=float, default=5.0, help="Time tolerance for matching validation calls (default: 5.0s)")
    ap.add_argument("--skip-extraction", action='store_true', help="Skip feature extraction step (use existing .npz files)")
    args = ap.parse_args()

    audio_root = Path(args.audio_root)
    npz_root = Path(args.npz_root) if args.npz_root else Path(args.output_root) / "npz"
    validation_csv = Path(args.validation_csv)
    output_root = Path(args.output_root)

    if not audio_root.exists():
        raise FileNotFoundError(f"Audio root not found: {audio_root}")
    if not validation_csv.exists():
        raise FileNotFoundError(f"Validation CSV not found: {validation_csv}")

    output_root.mkdir(parents=True, exist_ok=True)

    config_info = {
        'audio_root': str(audio_root),
        'npz_root': str(npz_root),
        'validation_csv': str(validation_csv),
        'tolerance': args.tolerance,
        'param_grid': PARAM_GRID,
    }

    with open(output_root / "tuning_config.json", 'w') as f:
        json.dump(config_info, f, indent=2)

    print(f"\nHyperparameter Tuning")
    print(f"Audio: {audio_root}")
    print(f"Validation: {validation_csv}")
    print(f"Output: {output_root}\n")

    # Run grid search
    results_df = grid_search(audio_root, npz_root, validation_csv, output_root, tolerance=args.tolerance)

    # Save results
    results_csv = output_root / "tuning_results.csv"
    results_df.to_csv(results_csv, index=False)

    # Summary
    print(f"\n=================SUMMARY=================\n")
    print(f"Results saved to: {results_csv}")
    print(f"Best configuration by F1:")
    best_row = results_df.sort_values(by='f1', ascending=False).iloc[0]
    for key in PARAM_GRID.keys():
        print(f"  {key}: {best_row[key]}")

if __name__ == "__main__":
    main()
