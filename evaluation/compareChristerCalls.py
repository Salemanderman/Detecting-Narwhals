"""
Check if outliers found match Christer's validation calls.

Example usage:
    python evaluation/compareChristerCalls.py 
        --outliers-csv output/pipeline_results/outliers/outliers.csv 
        --validation-csv evaluation/validatedChristerCalls.csv 
        --tolerance 5.0 
        --output evaluation/comparison_report.txt
"""

import argparse

import numpy as np
import pandas as pd


def find_matched_validations(outliers_df, validation_df, tolerance_sec):
    """Find validation calls that match detected outliers within tolerance."""
    matched = []
    unmatched = []

    for _, val_row in validation_df.iterrows():
        val_file = val_row["file"]
        val_time = val_row["start_sec"]

        time_match = np.abs(outliers_df["Start Time (s)"] - val_time) < tolerance_sec
        file_match = outliers_df["File"] == val_file
        matches = outliers_df[time_match & file_match]

        if not matches.empty:
            matched.append({
                "file": val_file,
                "validation_time": val_time,
                "outlier_times": matches["Start Time (s)"].tolist()
            })
        else:
            unmatched.append({
                "file": val_file,
                "validation_time": val_time
            })

    return matched, unmatched


def main():
    ap = argparse.ArgumentParser(description="Check if outliers found match Christer's validation calls")
    ap.add_argument("--outliers-csv", required=True, help="Path to outliers.csv from pipeline results")
    ap.add_argument("--validation-csv", required=True, help="Path to validation CSV with validated call timestamps")
    ap.add_argument("--tolerance", type=float, default=5.0, help="Time tolerance in seconds (default: 5.0)")
    ap.add_argument("--output", type=str, default=None, help="Optional path to save report")
    args = ap.parse_args()

    outliers_df = pd.read_csv(args.outliers_csv)
    validation_df = pd.read_csv(args.validation_csv)

    print(f"Loaded {len(outliers_df)} outliers, {len(validation_df)} validation calls")
    print(f"Tolerance: {args.tolerance}s\n")

    matched, unmatched = find_matched_validations(outliers_df, validation_df, args.tolerance)

    # Statistics
    total = len(validation_df)
    n_matched = len(matched)
    recall = n_matched / total if total > 0 else 0
    precision = n_matched / len(outliers_df) if len(outliers_df) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0


    print("Results;")
    # Build report
    results = [
        "RESULTS FROM COMPARISON",
        "_" * 50,
        f"n validation calls:     {total}",
        f"n matched as outliers:  {n_matched}",
        f"n missed:               {len(unmatched)}",
        f"Recall:                 {recall:.1%}",
        f"Precision:              {precision:.1%}",
        f"F1 Score:              {f1_score:.1%}",
        "",
        "List of matched:",
    ]
    for m in matched:
        results.append(f"  file: {m['file']}, time: {m['validation_time']}s")

    results.append("\nList of missed validationcalls:")
    for u in unmatched:
        results.append(f"  file: {u['file']}, time: {u['validation_time']}s")

    report = "\n".join(results)
    print(report)

    if args.output:
        with open(args.output, "w") as f:
            f.write(report)
        print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()

