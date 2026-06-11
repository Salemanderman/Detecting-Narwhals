"""
Train a Random Forest classifier on type-annotated windows.

Loads evaluation/type_annotations.csv, recomputes MFCC features from NPZ
files, trains a RandomForestClassifier with balanced class weights, and saves
the model + metadata with joblib.

Usage:
    python clustering/train_type_classifier.py \
        --npz-root output/mixedDataset/melBins/towsey/npz \
        --output-root evaluation/
"""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "clustering"))

import utilities.configs as configs
import utilities.feature_utils as futils
from clustering_core import mfcc_features


def build_npz_index(npz_root: Path) -> dict[str, Path]:
    """Map filename → full path."""
    return {p.name: p for p in npz_root.glob("*.npz")}


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
    ap.add_argument("--npz-root", required=True,
                    help="Directory containing .npz spectrogram files")
    ap.add_argument("--output-root", default=str(ROOT / "evaluation"),
                    help="Directory to save the trained model (default: evaluation/)")
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

    annotations = annotations[annotations["type"] != "unknown"]
    if len(annotations) == 0:
        print("[error] No labelled annotations found or all are unknown.")
        sys.exit(1)

    # Drop types with fewer than CV_FOLDS examples: stratified k-fold needs at
    # least k per class, and a singleton class makes cross_val_score fail.
    CV_FOLDS = 5
    counts = annotations["type"].value_counts()
    rare   = counts[counts < CV_FOLDS].index.tolist()
    if rare:
        print(f"  [warn] Dropping rare types (< {CV_FOLDS} examples): {rare}")
        annotations = annotations[~annotations["type"].isin(rare)]
        if len(annotations) == 0:
            print(f"[error] No types left with >= {CV_FOLDS} examples. Label more clusters.")
            sys.exit(1)

    npz_index = build_npz_index(Path(args.npz_root))
    print(f"\nIndexed {len(npz_index)} NPZ files")

    print(f"\nExtracting MFCC features  (mel_start={args.mel_start}, mel_end={args.mel_end}, n_mfcc={args.n_mfcc})...")
    X, y = extract_features(annotations, npz_index, args.mel_start, args.mel_end, args.n_mfcc)
    print(f"  Feature matrix: {X.shape}  ({len(np.unique(y))} classes)")

    clf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    print(f"\nCross-validation ({CV_FOLDS}-fold)...")
    # Secondary guard: feature extraction may skip rows (missing/short files), so a
    # class can shrink below CV_FOLDS even after the rare-type filter above.
    min_class_count = int(np.unique(y, return_counts=True)[1].min())
    cv = min(CV_FOLDS, min_class_count)
    scores = cross_val_score(clf, X, y, cv=cv, scoring="balanced_accuracy")
    print(f"  Balanced accuracy: {scores.mean():.3f} ± {scores.std():.3f}")

    clf.fit(X, y)

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / "type_classifier.joblib"
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
    print(f"\nModel saved to {output_path}")
    print(f"  Classes: {list(clf.classes_)}")


if __name__ == "__main__":
    main()
