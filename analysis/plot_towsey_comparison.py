"""
Visual comparison of plain vs Towsey preprocessing.

Produces three figures:
  Figure 1 — spectrogram grid: N windows side by side (plain left, Towsey right)
  Figure 2 — monthly mean spectrum: one curve per month, plain vs Towsey panels
  Figure 3 — PCA scatter coloured by days since first recording (continuous scale)

Requires two pipeline runs on the same audio:
  Run A: python run_outlier_pipeline.py  (plain)
  Run B: python run_outlier_pipeline.py --towsey

Usage:
    python analysis/plot_towsey_comparison.py \
        --npz-root-plain  output/6230/plain/npz \
        --npz-root-towsey output/6230/towsey/npz \
        --pca-root-plain  output/6230/plain/pca \
        --pca-root-towsey output/6230/towsey/pca \
        --output-root     output/figures/towsey
"""

import argparse
from datetime import datetime
from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utilities import configs
import utilities.feature_utils as futils

MONTH_NAMES   = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                 7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
MONTH_COLORS  = ["#2171b5","#cb181d","#238b45","#d94801","#6a51a3","#636363","#8c6d31"]


def parse_datetime(filename: str):
    """'RECORDER.YYMMDDHHMMSS.npz' → datetime, or None."""
    parts = Path(filename).stem.split(".")
    if len(parts) < 2 or len(parts[1]) < 12:
        return None
    ts = parts[1]
    try:
        return datetime(2000 + int(ts[0:2]), int(ts[2:4]), int(ts[4:6]),
                        int(ts[6:8]), int(ts[8:10]), int(ts[10:12]))
    except ValueError:
        return None


def _save(fig, path):
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
        plt.close(fig)
    else:
        plt.show()


# ── Figure 1: spectrogram grid ────────────────────────────────────────────────

def plot_spectrogram_grid(npz_plain, npz_towsey, mel_start, mel_end,
                          window_secs, n_examples, output_path):
    spec_cfg      = configs.get_specgram_config()
    secs_per_frame = spec_cfg["hop_length"] / spec_cfg["sample_rate"]
    window_frames  = max(1, round(window_secs / secs_per_frame))

    files = sorted(npz_plain.glob("*.npz"))
    if not files:
        print(f"[warn] No .npz files in {npz_plain}")
        return

    # Sample evenly across sorted (= chronological) file list
    sample_idx = np.linspace(0, len(files) - 1, n_examples, dtype=int)
    sampled    = [files[i] for i in sample_idx]

    fig, axes = plt.subplots(n_examples, 2, figsize=(10, 3 * n_examples))
    if n_examples == 1:
        axes = axes[np.newaxis, :]

    for row, fpath in enumerate(sampled):
        fname  = fpath.name
        dt     = parse_datetime(fname)
        dt_str = dt.strftime("%d %b %Y") if dt else fname

        for col, root in enumerate((npz_plain, npz_towsey)):
            ax = axes[row, col]
            try:
                S, _ = futils.load_spectrogram(root / fname, n_mels=None)
                ms   = mel_start or 0
                me   = mel_end   or S.shape[0]
                mid  = S.shape[1] // 2
                w    = S[ms:me, mid:mid + window_frames]
                if w.shape[1] < window_frames:
                    w = np.pad(w, ((0, 0), (0, window_frames - w.shape[1])))
                ax.imshow(w, aspect="auto", origin="lower", cmap="viridis",
                          vmin=np.percentile(w, 5), vmax=np.percentile(w, 99),
                          interpolation="nearest")
            except Exception as e:
                ax.text(0.5, 0.5, str(e), ha="center", va="center",
                        fontsize=7, transform=ax.transAxes, wrap=True)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title("Plain" if col == 0 else "Towsey",
                             fontsize=12, fontweight="bold")
        axes[row, 0].set_ylabel(dt_str, fontsize=9, rotation=0,
                                labelpad=65, va="center")

    fig.suptitle("Spectrogram comparison — plain vs Towsey  (same window, same file)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, output_path)


# ── Figure 2: monthly mean spectrum ──────────────────────────────────────────

def plot_monthly_mean_spectrum(npz_plain, npz_towsey, mel_start, mel_end, output_path):

    def _monthly_means(root):
        monthly = {}
        for fpath in sorted(root.glob("*.npz")):
            dt = parse_datetime(fpath.name)
            if dt is None:
                continue
            try:
                S, _ = futils.load_spectrogram(fpath, n_mels=None)
                ms   = mel_start or 0
                me   = mel_end   or S.shape[0]
                monthly.setdefault(dt.month, []).append(S[ms:me].mean(axis=1))
            except Exception:
                pass
        return {m: np.stack(v).mean(axis=0) for m, v in monthly.items()}

    print("  Loading plain spectra...")
    means_plain  = _monthly_means(npz_plain)
    print("  Loading Towsey spectra...")
    means_towsey = _monthly_means(npz_towsey)

    months    = sorted(set(means_plain) | set(means_towsey))
    color_map = {m: MONTH_COLORS[i % len(MONTH_COLORS)] for i, m in enumerate(months)}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    for m in months:
        label = MONTH_NAMES.get(m, str(m))
        c     = color_map[m]
        if m in means_plain:
            ax1.plot(means_plain[m],  color=c, linewidth=1.8, label=label)
        if m in means_towsey:
            ax2.plot(means_towsey[m], color=c, linewidth=1.8, label=label)

    for ax, title in ((ax1, "Plain"), (ax2, "Towsey")):
        ax.set_xlabel("Frequency bin", fontsize=11)
        ax.set_ylabel("Mean power", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend(title="Month", fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Monthly mean spectrum — plain vs Towsey\n"
                 "(curves collapsing in Towsey panel = seasonal noise floor removed)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, output_path)


# ── Figure 3: PCA scatter coloured by days since first ───────────────────────

def plot_pca_time_colour(pca_root_plain, pca_root_towsey, output_path):

    def _load(pca_root):
        raw   = np.load(pca_root / "pca_results.npz", allow_pickle=True)
        X_pca = raw["X_pca"][:, :2]
        files = raw["window_files"]
        dts   = np.array([parse_datetime(str(f)) for f in files])
        valid = np.array([d is not None for d in dts])
        dts   = dts[valid]
        first = min(dts)
        days  = np.array([(d - first).total_seconds() / 86400.0 for d in dts])
        return X_pca[valid], days

    X_a, days_a = _load(pca_root_plain)
    X_b, days_b = _load(pca_root_towsey)

    vmin = min(days_a.min(), days_b.min())
    vmax = max(days_a.max(), days_b.max())
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.viridis

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    for ax, X, days, title in (
        (ax1, X_a, days_a, "Plain"),
        (ax2, X_b, days_b, "Towsey"),
    ):
        ax.scatter(X[:, 0], X[:, 1], c=days, cmap=cmap, norm=norm,
                   s=4, alpha=0.4, linewidths=0)
        ax.set_xlabel("PC1", fontsize=11)
        ax.set_ylabel("PC2", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.2)

    cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap),
                        ax=[ax1, ax2], orientation="vertical",
                        fraction=0.02, pad=0.02)
    cbar.set_label("Days since first recording", fontsize=11)

    fig.suptitle("PCA scatter coloured by recording date\n"
                 "(a gradient along PC1 in the plain panel = PCA capturing time, not biology)",
                 fontsize=13, fontweight="bold")
    _save(fig, output_path)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Compare plain vs Towsey preprocessing with three diagnostic figures.")
    ap.add_argument("--npz-root-plain",  required=True)
    ap.add_argument("--npz-root-towsey", required=True)
    ap.add_argument("--pca-root-plain",  default=None,
                    help="Required for the PCA scatter figure (Figure 3)")
    ap.add_argument("--pca-root-towsey", default=None,
                    help="Required for the PCA scatter figure (Figure 3)")
    ap.add_argument("--output-root",     default=None,
                    help="Save PNGs here; show interactively if omitted")
    ap.add_argument("--mel-start",       type=int,   default=None)
    ap.add_argument("--mel-end",         type=int,   default=None)
    ap.add_argument("--window-secs",     type=float, default=5.0)
    ap.add_argument("--n-examples",      type=int,   default=6,
                    help="Spectrogram pairs in Figure 1 (default: 6)")
    args = ap.parse_args()

    npz_plain  = Path(args.npz_root_plain)
    npz_towsey = Path(args.npz_root_towsey)
    out        = Path(args.output_root) if args.output_root else None
    if out:
        out.mkdir(parents=True, exist_ok=True)

    print("Figure 1: spectrogram grid...")
    plot_spectrogram_grid(
        npz_plain, npz_towsey,
        args.mel_start, args.mel_end, args.window_secs, args.n_examples,
        out / "spectrogram_grid.png" if out else None,
    )

    print("\nFigure 2: monthly mean spectrum...")
    plot_monthly_mean_spectrum(
        npz_plain, npz_towsey,
        args.mel_start, args.mel_end,
        out / "monthly_mean_spectrum.png" if out else None,
    )

    if args.pca_root_plain and args.pca_root_towsey:
        print("\nFigure 3: PCA scatter by date...")
        plot_pca_time_colour(
            Path(args.pca_root_plain), Path(args.pca_root_towsey),
            out / "pca_time_colour.png" if out else None,
        )
    else:
        print("\nSkipping Figure 3 (provide --pca-root-plain and --pca-root-towsey to enable)")

    print("\nDone.")


if __name__ == "__main__":
    main()
