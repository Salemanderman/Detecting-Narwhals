"""
Save spectrogram grids grouped by predicted_type from a classified CSV.

Takes the output of classify_windows.py (predicted_type + type_confidence columns)
and saves paginated spectrogram grids per type so you can visually verify whether
the RF classifier is making sensible predictions.

Windows are sorted highest-confidence first within each type, so the most certain
predictions appear on the first page.

Usage:
    python analysis/plot_classified_grids.py \
        --classified-csv output/mixedDataset/melBins/towsey/clusters_dpmm_init/clusters_classified.csv \
        --npz-root        output/mixedDataset/melBins/towsey/npz \
        --output-root     output/mixedDataset/melBins/towsey/classified_grids
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "clustering"))

import utilities.configs as configs
import utilities.feature_utils as futils


def save_type_grid(type_df: pd.DataFrame, type_label: str, npz_root: Path,
                   output_dir: Path, window_frames: int, spec_cfg: dict,
                   mel_start: int, mel_end: int, page_size: int = 30):
    """Save paginated spectrogram grids for all windows of one predicted type."""
    n          = len(type_df)
    sorted_df  = type_df.sort_values("type_confidence", ascending=False)
    n_pages    = max(1, (n + page_size - 1) // page_size)
    output_dir.mkdir(parents=True, exist_ok=True)

    for page in range(n_pages):
        page_df = sorted_df.iloc[page * page_size : (page + 1) * page_size]
        n_show  = len(page_df)
        n_cols  = min(6, n_show)
        n_rows  = (n_show + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
        axes = np.atleast_1d(np.array(axes)).flatten()

        for i, (_, row) in enumerate(page_df.iterrows()):
            ax   = axes[i]
            t    = float(row["Start Time (s)"])
            conf = float(row["type_confidence"])
            try:
                w = futils.get_window(npz_root / row["File"], t, window_frames,
                                      mel_start=mel_start, mel_end=mel_end,
                                      spec_cfg=spec_cfg)
                ax.imshow(w, aspect="auto", origin="lower", cmap="viridis")
                ax.set_title(f"{Path(row['File']).stem[:18]}\nt={t:.1f}s  conf={conf:.2f}",
                             fontsize=8)
            except Exception as e:
                ax.set_title(f"Error: {e}", fontsize=7)
                ax.axis("off")
            ax.set_xticks([])
            ax.set_yticks([])

        for j in range(n_show, len(axes)):
            axes[j].axis("off")

        if n_pages == 1:
            title    = f"{type_label}  ({n} windows)"
            out_path = output_dir / "spectrogram_grid.png"
        else:
            title    = f"{type_label}  (page {page + 1}/{n_pages}  ·  {n} windows total)"
            out_path = output_dir / f"spectrogram_grid_{page + 1}.png"

        fig.suptitle(title, fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser(
        description="Save spectrogram grids per predicted type from a classified CSV.")
    ap.add_argument("--classified-csv", required=True,
                    help="Output of classify_windows.py (needs predicted_type + type_confidence columns)")
    ap.add_argument("--npz-root",       required=True,
                    help="Directory containing .npz spectrogram files")
    ap.add_argument("--output-root",    required=True,
                    help="Where to write per-type subdirectories with grids")
    ap.add_argument("--mel-start",      type=int,   default=11)
    ap.add_argument("--mel-end",        type=int,   default=128)
    ap.add_argument("--window-secs",    type=float, default=5.0)
    ap.add_argument("--page-size",      type=int,   default=30,
                    help="Windows per grid page (default: 30)")
    ap.add_argument("--min-confidence", type=float, default=0.0,
                    help="Only show windows at or above this confidence (default: 0.0 = all)")
    ap.add_argument("--types", nargs="*", default=None,
                    help="Which predicted types to plot (e.g. --types clicks tonal). "
                         "Omit to plot all types. Use --list-types to see what exists.")
    ap.add_argument("--list-types", action="store_true",
                    help="Print available types in the CSV and exit without plotting.")
    args = ap.parse_args()

    df = pd.read_csv(args.classified_csv)
    if "predicted_type" not in df.columns or "type_confidence" not in df.columns:
        print("[error] CSV must have predicted_type and type_confidence columns.")
        print("  Run clustering/classify_windows.py first.")
        sys.exit(1)

    if args.min_confidence > 0:
        low_conf = df["type_confidence"] < args.min_confidence
        df.loc[low_conf, "predicted_type"] = "uncertain"
        print(f"Windows below confidence {args.min_confidence:.0%} "
              f"→ 'uncertain'  ({low_conf.sum()} windows)")

    all_types = sorted(df["predicted_type"].unique())
    print(f"\nType distribution:")
    for t in all_types:
        sub       = df[df["predicted_type"] == t]
        mean_conf = sub["type_confidence"].mean()
        print(f"  {t:14s}  {len(sub):5d} windows  mean_conf={mean_conf:.2f}")

    if args.list_types:
        sys.exit(0)

    if args.types is not None:
        unknown = [t for t in args.types if t not in all_types]
        if unknown:
            print(f"\n[warn] requested types not found in CSV: {unknown}")
            print(f"  available: {all_types}")
        types = [t for t in args.types if t in all_types]
        if not types:
            print("[error] no valid types to plot after filtering.")
            sys.exit(1)
        print(f"\nPlotting {len(types)}/{len(all_types)} type(s): {types}")
    else:
        types = all_types

    spec_cfg      = configs.get_specgram_config()
    window_frames = round(args.window_secs * spec_cfg["sample_rate"] / spec_cfg["hop_length"])
    npz_root      = Path(args.npz_root)
    output_root   = Path(args.output_root)

    print(f"\nSaving grids to {output_root} ...")
    for t in tqdm(types, desc="Saving grids", unit="type"):
        sub      = df[df["predicted_type"] == t]
        out_dir  = output_root / t
        save_type_grid(sub, t, npz_root, out_dir, window_frames, spec_cfg,
                       args.mel_start, args.mel_end, args.page_size)
        tqdm.write(f"  [{t}] {len(sub)} windows → {out_dir}")

    print(f"\n[done] {output_root}")


if __name__ == "__main__":
    main()
