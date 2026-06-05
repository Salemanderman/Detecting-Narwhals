"""
UMAP embedding — drop-in replacement for pca_sliding_window.py.

Writes pca_results.npz with the same keys so cluster.py, finding_outliers.py,
and ensemble_cluster.py all work unchanged.

Feature modes:
  mean_std          — mean + std per mel bin
  mfcc              — 2*n_mfcc MFCC coefficients (mean + std)
  extended_acoustic — 31 acoustic features (spectral + temporal)
  passthrough       — no reduction, raw normalised features as-is

Iterative clustering: use --filter-csv to re-embed only a subset of windows
(e.g. pass the narwhal cluster CSV to re-run UMAP on just those windows).

Usage:
    python analysis/umap_embedding.py \
        --npz-root  output/mixedDataset/melBins/towsey/npz \
        --output-root output/mixedDataset/melBins/towsey/umap \
        --window-secs 5 --stride-secs 5 \
        --mel-start 11 --mel-end 128 \
        --n-components 10 \
        --feature-mode mfcc

    # Iterative pass — re-embed only a kept subset
    python analysis/umap_embedding.py \
        --npz-root  output/mixedDataset/melBins/towsey/npz \
        --output-root output/mixedDataset/melBins/towsey/umap_iter2 \
        --window-secs 5 --stride-secs 5 \
        --mel-start 11 --mel-end 128 \
        --n-components 10 \
        --feature-mode mfcc \
        --filter-csv output/iterative/pass_1/kept_windows.csv
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import umap

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import utilities.configs as configs
import utilities.feature_utils as futils
import utilities.plot_utils as putils
from clustering.clustering_core import mfcc_features


def _int_or_none(v):
    return None if str(v).lower() == "none" else int(v)


def window_feature(window: np.ndarray) -> np.ndarray:
    mu  = window.mean(axis=1)
    sig = window.std(axis=1)
    return np.concatenate([mu, sig])


def main():
    ap = argparse.ArgumentParser(description="UMAP embedding over NPZ log-mel files.")
    ap.add_argument("--npz-root",    required=True)
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--window-secs", type=float, default=5.0)
    ap.add_argument("--stride-secs", type=float, default=None)
    ap.add_argument("--mel-start",   type=int,   default=0)
    ap.add_argument("--mel-end",     type=_int_or_none, default=None)
    ap.add_argument("--n-components", type=int,  default=10)
    ap.add_argument("--n-neighbors",  type=int,  default=15)
    ap.add_argument("--min-dist",     type=float, default=0.1)
    ap.add_argument("--metric",       default="euclidean")
    ap.add_argument("--seed",         type=int,  default=42)
    ap.add_argument("--feature-key",  default="feature")
    ap.add_argument("--single-file",  default=None)
    ap.add_argument("--no-plot",      action="store_true")
    ap.add_argument("--n-mels",       type=_int_or_none, default=None)
    ap.add_argument("--n-mfcc",       type=int, default=20,
                    help="Number of MFCC coefficients (only used when --feature-mode mfcc, default: 20).")
    ap.add_argument("--feature-mode",
                    choices=["mean_std", "mfcc", "passthrough"],
                    default="mean_std")
    ap.add_argument("--filter-csv",   default=None,
                    help="Only embed windows listed in this CSV (File + Start Time (s)).")
    args = ap.parse_args()

    npz_root    = Path(args.npz_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    with open(output_root / "run_config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    stride_secs    = args.stride_secs or args.window_secs
    spec_config    = configs.get_specgram_config()
    secs_per_frame = spec_config["hop_length"] / spec_config["sample_rate"]
    window_frames  = max(1, round(args.window_secs / secs_per_frame))
    stride_frames  = max(1, round(stride_secs / secs_per_frame))

    npz_files = sorted(npz_root.glob("*.npz"))
    if args.single_file:
        npz_files = [p for p in npz_files if p.name == args.single_file]
    if not npz_files:
        print(f"ERROR: no NPZ files found under {npz_root}")
        sys.exit(1)

    n_mels = args.n_mels
    if n_mels is None:
        S, _ = futils.load_spectrogram(npz_files[0], n_mels=None, key=args.feature_key)
        n_mels = S.shape[0]
        print(f"Auto-detected {n_mels} frequency bins from first file")

    mel_end = args.mel_end or n_mels

    filter_set = None
    if args.filter_csv:
        fdf = pd.read_csv(args.filter_csv)
        filter_set = set(zip(fdf["File"], fdf["Start Time (s)"].round(3)))
        print(f"Filter CSV: {args.filter_csv}  ({len(filter_set)} windows)")

    print(f"NPZ root:     {npz_root}  ({len(npz_files)} files)")
    print(f"Output root:  {output_root}")
    print(f"Window: {args.window_secs:.2f}s ({window_frames} frames)  "
          f"Stride: {stride_secs:.2f}s ({stride_frames} frames)")
    print(f"Mel bins: [{args.mel_start}, {mel_end})")
    print(f"Feature mode: {args.feature_mode}")
    print(f"UMAP: n_components={args.n_components}  n_neighbors={args.n_neighbors}  "
          f"min_dist={args.min_dist}  metric={args.metric}  seed={args.seed}")

    feature_rows, window_meta = [], []

    for npz_path in tqdm(npz_files, desc="Extracting windows", unit="file"):
        try:
            S, _ = futils.load_spectrogram(npz_path, n_mels=n_mels, key=args.feature_key)
        except Exception as e:
            tqdm.write(f"  [skip] {npz_path.name}: {e}")
            continue
        for start_frame, win in futils.windows_from_spectrogram(
                S, window_frames, stride_frames,
                mel_start=args.mel_start, mel_end=mel_end):
            start_sec = round(start_frame * secs_per_frame, 3)
            if filter_set and (npz_path.name, start_sec) not in filter_set:
                continue
            elif args.feature_mode == "mfcc":
                feat = mfcc_features(win, n_mfcc=args.n_mfcc)
            elif args.feature_mode == "passthrough":
                feat = window_feature(win)
            else:
                feat = window_feature(win)
            feature_rows.append(feat)
            window_meta.append({
                "file": npz_path.name,
                "start_frame": int(start_frame),
                "start_sec": start_sec,
            })

    if not feature_rows:
        print("[error] No windows extracted.")
        sys.exit(1)

    X = np.stack(feature_rows, axis=0)
    print(f"\nFeature matrix: {X.shape}  (windows x features)")

    norm_mean = X.mean(axis=0)
    norm_std  = X.std(axis=0)
    norm_std  = np.where(norm_std > 0, norm_std, 1.0)
    X_norm    = (X - norm_mean) / norm_std

    if args.feature_mode == "passthrough":
        print("Skipping reduction — using normalised features directly.")
        X_umap = X_norm
    else:
        print("Fitting UMAP...")
        reducer = umap.UMAP(
            n_components=args.n_components,
            n_neighbors=args.n_neighbors,
            min_dist=args.min_dist,
            metric=args.metric,
            random_state=args.seed,
            verbose=False,
        )
        X_umap = reducer.fit_transform(X_norm)
    print(f"Embedding shape: {X_umap.shape}")

    files_arr  = np.array([w["file"]        for w in window_meta], dtype=object)
    starts_arr = np.array([w["start_frame"] for w in window_meta], dtype=np.int64)
    secs_arr   = np.array([w["start_sec"]   for w in window_meta], dtype=np.float32)

    out_npz = output_root / "pca_results.npz"
    np.savez_compressed(
        out_npz,
        X_pca=X_umap,
        evr=np.zeros(X_umap.shape[1], dtype=np.float32),
        window_files=files_arr,
        window_start_frames=starts_arr,
        window_start_secs=secs_arr,
        window_secs=args.window_secs,
        stride_secs=stride_secs,
        mel_start=args.mel_start,
        mel_end=mel_end,
        n_components=int(X_umap.shape[1]),
        norm_mean=norm_mean,
        norm_std=norm_std,
        reduction_method="umap" if args.feature_mode != "passthrough" else "passthrough",
        umap_n_neighbors=args.n_neighbors,
        umap_min_dist=args.min_dist,
        umap_metric=args.metric,
        feature_mode=args.feature_mode,
    )
    print(f"\nSaved to {out_npz}")

    if not args.no_plot:
        plot_path = output_root / "umap_plot.png"
        evr_zeros = np.zeros(X_umap.shape[1], dtype=np.float32)
        if len(npz_files) == 1:
            putils.plot_pca_projection_single(X_umap, evr_zeros, window_meta, plot_path)
            putils.plot_pca_projection_single_3d(X_umap, evr_zeros, window_meta,
                                                  output_root / "umap_plot_3d.png")
        else:
            putils.plot_pca_projection(X_umap, evr_zeros, window_meta, plot_path)
            putils.plot_pca_projection_3d(X_umap, evr_zeros, window_meta,
                                           output_root / "umap_plot_3d.png")
        print(f"Saved to {plot_path}")


if __name__ == "__main__":
    main()
