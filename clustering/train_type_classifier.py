"""
Train a Random Forest classifier on type-annotated windows.

Loads evaluation/type_annotations.csv, recomputes MFCC features from NPZ
files, trains a RandomForestClassifier with balanced class weights, and saves
the model + metadata with joblib.

Usage:
    python analysis/train_type_classifier.py \
        --npz-root output/mixedDataset/melBins/towsey/npz \
        --output-path evaluation/type_classifier.joblib

    # Multiple NPZ roots when annotations span several datasets:
    python analysis/train_type_classifier.py \
        --npz-root output/mixedDataset/melBins/towsey/npz \
                   output/GOODClusteringWithTowsey/subsetWithValidatedCalls/npz \
        --output-path evaluation/type_classifier.joblib
"""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "clustering"))

import utilities.configs as configs
import utilities.feature_utils as futils
from clustering_core import mfcc_features


def build_npz_index(npz_roots: list[Path]) -> dict[str, Path]:
    """Map filename → full path across all provided roots."""
    index = {}
    for root in npz_roots:
        for p in root.glob("*.npz"):
            index[p.name] = p
    return index


def extract_features(annotations: pd.DataFrame, npz_index: dict,
                     mel_start: int, mel_end: int, n_mfcc: int) -> tuple:
    spec_cfg     = configs.get_specgram_config()
    window_secs  = 5.0
    window_frames = round(window_secs * spec_cfg["sample_rate"] / spec_cfg["hop_length"])

    X, y, skipped = [], [], 0
    for _, row in annotations.iterrows():
        npz_path = npz_index.get(row["File"])
        if npz_path is None:
            skipped += 1
            continue
        try:
            win = futils.get_window(npz_path, float(row["Start Time (s)"]),
                                    window_frames, mel_start, mel_end, spec_cfg)
            X.append(mfcc_features(win, n_mfcc=n_mfcc))
            y.append(row["type"])
        except Exception:
            skipped += 1

    if skipped:
        print(f"  [warn] {skipped} windows skipped (file not found or too short)")
    return np.array(X, dtype=np.float32), np.array(y)


def main():
    ap = argparse.ArgumentParser(description="Train a type classifier on annotated windows.")
    ap.add_argument("--annotations-csv", default=str(ROOT / "evaluation" / "type_annotations.csv"),
                    help="Type annotations CSV (default: evaluation/type_annotations.csv)")
    ap.add_argument("--npz-root", nargs="+", required=True,
                    help="One or more directories containing .npz spectrogram files")
    ap.add_argument("--output-path", default=str(ROOT / "evaluation" / "type_classifier.joblib"),
                    help="Where to save the trained model (default: evaluation/type_classifier.joblib)")
    ap.add_argument("--mel-start", type=int, default=11)
    ap.add_argument("--mel-end",   type=int, default=128)
    ap.add_argument("--n-mfcc",    type=int, default=20)
    ap.add_argument("--n-estimators", type=int, default=300)
    args = ap.parse_args()

    ann_path = Path(args.annotations_csv)
    if not ann_path.exists():
        print(f"[error] Annotations CSV not found: {ann_path}")
        sys.exit(1)

    annotations = pd.read_csv(ann_path)
    print(f"Loaded {len(annotations)} annotations from {ann_path}")
    print(f"  Type counts:\n{annotations['type'].value_counts().to_string()}")

    # Drop unknowns and types with < 5 examples
    counts = annotations["type"].value_counts()
    rare   = counts[counts < 5].index.tolist()
    if rare:
        print(f"  [warn] Dropping rare types (< 5 examples): {rare}")
        annotations = annotations[~annotations["type"].isin(rare)]
    annotations = annotations[annotations["type"] != "unknown"]

    if len(annotations) < 10:
        print("[error] Not enough labelled windows to train. Label more clusters first.")
        sys.exit(1)

    npz_index = build_npz_index([Path(r) for r in args.npz_root])
    print(f"\nIndexed {len(npz_index)} NPZ files across {len(args.npz_root)} root(s)")

    print(f"\nExtracting MFCC features  (mel_start={args.mel_start}, mel_end={args.mel_end}, n_mfcc={args.n_mfcc})...")
    X, y = extract_features(annotations, npz_index, args.mel_start, args.mel_end, args.n_mfcc)
    print(f"  Feature matrix: {X.shape}  ({len(np.unique(y))} classes)")

    clf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    print(f"\nCross-validation (5-fold)...")
    min_class_count = int(np.unique(y, return_counts=True)[1].min())
    cv = min(5, min_class_count)
    scores = cross_val_score(clf, X, y, cv=cv, scoring="balanced_accuracy")
    print(f"  Balanced accuracy: {scores.mean():.3f} ± {scores.std():.3f}")

    clf.fit(X, y)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model":      clf,
        "classes":    list(clf.classes_),
        "mel_start":  args.mel_start,
        "mel_end":    args.mel_end,
        "n_mfcc":     args.n_mfcc,
        "window_secs": 5.0,
        "n_train":    len(X),
    }
    joblib.dump(payload, output_path)
    print(f"\nModel saved → {output_path}")
    print(f"  Classes: {list(clf.classes_)}")
    print(f"  Feature importances (top 5): {np.argsort(clf.feature_importances_)[::-1][:5].tolist()}")


if __name__ == "__main__":
    main()
