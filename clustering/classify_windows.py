"""
Run the trained type classifier on a clusters.csv (or any window list CSV).

Adds predicted_type and type_confidence columns, prints a per-cluster summary
flagging uncertain or mixed-prediction clusters, and saves an annotated CSV.

Usage:
    python clustering/classify_windows.py \
        --windows-csv output/mixedDataset/melBins/towsey/iterative_mfcc/pass_5/clusters/clusters.csv \
        --npz-root    output/mixedDataset/melBins/towsey/npz \
        --output-root output/mixedDataset/melBins/towsey/iterative_mfcc/pass_5/clusters
"""

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "clustering"))

import utilities.configs as configs
import utilities.feature_utils as futils
from clustering_core import mfcc_features


def build_npz_index(npz_root: Path) -> dict[str, Path]:
    return {p.name: p for p in npz_root.glob("*.npz")}


def predict_windows(df: pd.DataFrame, npz_index: dict, payload: dict) -> pd.DataFrame:
    """Add predicted_type and type_confidence columns to df. Returns a copy."""
    spec_cfg       = configs.get_specgram_config()
    secs_per_frame = spec_cfg["hop_length"] / spec_cfg["sample_rate"]
    window_frames  = round(payload["window_secs"] * spec_cfg["sample_rate"] / spec_cfg["hop_length"])
    mel_start      = payload["mel_start"]
    mel_end        = payload["mel_end"]
    n_mfcc         = payload["n_mfcc"]
    model          = payload["model"]

    pred_types = ["unknown"] * len(df)
    pred_confs = [0.0]       * len(df)

    # Group by file so each NPZ is loaded once instead of once per window.
    for npz_name, group in tqdm(df.groupby("File"), desc="Classifying", unit="file"):
        npz_path = npz_index.get(npz_name)
        if npz_path is None:
            continue
        try:
            S, _ = futils.load_spectrogram(npz_path, n_mels=None)
            n_bins, T = S.shape
            mel_e  = mel_end if mel_end else n_bins
            S_crop = S[mel_start:mel_e, :]

            for idx, row in group.iterrows():
                start_frame = round(float(row["Start Time (s)"]) / secs_per_frame)
                end_frame   = min(start_frame + window_frames, T)
                win = S_crop[:, start_frame:end_frame]
                if win.shape[1] < window_frames:
                    continue
                feat = mfcc_features(win, n_mfcc=n_mfcc).reshape(1, -1)
                pred_types[idx] = model.predict(feat)[0]
                pred_confs[idx] = float(model.predict_proba(feat).max())
        except Exception as e:
            tqdm.write(f"  [warn] {npz_name}: {e}")

    out = df.copy()
    out["predicted_type"]  = pred_types
    out["type_confidence"] = pred_confs
    return out


def cluster_summary(df: pd.DataFrame, confidence_threshold: float):
    """Print a per-cluster summary if a cluster column exists, otherwise per-type."""
    print(f"\nPrediction summary (confidence threshold: {confidence_threshold:.0%})")

    if "cluster" in df.columns:
        for cid in sorted(df["cluster"].unique()):
            sub  = df[df["cluster"] == cid]
            n    = len(sub)
            name = "Noise" if cid == -1 else f"Cluster {cid}"

            type_counts = sub["predicted_type"].value_counts()
            top_type    = type_counts.index[0]
            top_frac    = type_counts.iloc[0] / n
            mean_conf   = sub["type_confidence"].mean()

            flags = []
            if top_frac < 0.7:
                flags.append("MIXED")
            if mean_conf < confidence_threshold:
                flags.append("UNCERTAIN")
            flag_str = "  " + " + ".join(flags) if flags else ""

            dist = "  ".join(f"{t}:{c}" for t, c in type_counts.items())
            print(f"  {name:12s}  n={n:4d}  {top_type:12s} ({top_frac:.0%})  conf={mean_conf:.2f}{flag_str}")
            if len(type_counts) > 1:
                print(f"              distribution: {dist}")
    else:
        for t, n in df["predicted_type"].value_counts().items():
            sub       = df[df["predicted_type"] == t]
            mean_conf = sub["type_confidence"].mean()
            low_frac  = (sub["type_confidence"] < confidence_threshold).mean()
            flag = "  UNCERTAIN" if mean_conf < confidence_threshold else ""
            print(f"  {t:14s}  n={n:5d}  ({n/len(df):.0%})  conf={mean_conf:.2f}{flag}")

    n_low = (df["type_confidence"] < confidence_threshold).sum()
    n_unk = (df["predicted_type"] == "unknown").sum()
    print(f"\n  {n_low} windows below confidence threshold  |  {n_unk} windows unpredictable (file not found)")


def main():
    ap = argparse.ArgumentParser(description="Classify windows by acoustic type using a trained model.")
    ap.add_argument("--windows-csv",   required=True,
                    help="CSV with File + Start Time (s) columns (e.g. clusters.csv)")
    ap.add_argument("--npz-root",      required=True,
                    help="Directory containing .npz spectrogram files")
    ap.add_argument("--model-path",    default=str(ROOT / "evaluation" / "type_classifier.joblib"),
                    help="Trained model from train_type_classifier.py (default: evaluation/type_classifier.joblib)")
    ap.add_argument("--output-root",   default=None,
                    help="Directory to save the annotated CSV (default: alongside the input CSV)")
    ap.add_argument("--confidence-threshold", type=float, default=0.6,
                    help="Below this confidence, a window is flagged as uncertain (default: 0.6)")
    args = ap.parse_args()

    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"[error] Model not found: {model_path}")
        print("  Run clustering/train_type_classifier.py first.")
        sys.exit(1)

    payload = joblib.load(model_path)
    print(f"Loaded model: {model_path}")
    print(f"  Classes: {payload['classes']}  |  trained on {payload['n_train']} windows")
    print(f"  Features: mel_start={payload['mel_start']}  mel_end={payload['mel_end']}  n_mfcc={payload['n_mfcc']}")

    windows_csv = Path(args.windows_csv)
    df = pd.read_csv(windows_csv)
    print(f"\nLoaded {len(df)} windows from {windows_csv}")

    npz_index = build_npz_index(Path(args.npz_root))
    print(f"Indexed {len(npz_index)} NPZ files")

    print(f"\nRunning predictions...")
    df_out = predict_windows(df, npz_index, payload)

    cluster_summary(df_out, args.confidence_threshold)

    output_root = Path(args.output_root) if args.output_root else windows_csv.parent
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / (windows_csv.stem + "_classified.csv")
    df_out.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
