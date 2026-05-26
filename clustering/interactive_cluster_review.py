"""
Interactive cluster review tool — iterative narwhal call clustering for biologists.

Shows each cluster's spectrogram grid, lets you label it, then offers to re-cluster
(UMAP re-embed + HDBSCAN) on the cleaned subset, or run final ensemble clustering.
All passes are saved under output-root/pass_N/ so nothing is lost.

Usage:
    python clustering/interactive_cluster_review.py \
        --clusters-dir output/umap_large/clusters \
        --npz-root     output/threshold_sweep/plain/npz \
        --output-root  output/iterative_review \
        --mel-start 11 --mel-end 128

Then pick labels in the terminal; images open in your system viewer.
After labeling all clusters, choose what to do next:
  r = re-cluster on kept windows
  t/l = tighter/looser clusters
  f = final ensemble clustering
  s = save filtered CSV and exit
  q = quit
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

LABEL_MAP = {
    'k': 'keep',
    'r': 'remove',
}
REMOVE_LABELS = {'remove'}


# ── Display helpers ───────────────────────────────────────────────────────────

def bar(n, total, width=20):
    filled = round(width * n / max(total, 1))
    return '█' * filled + '░' * (width - filled)


def open_image(path: Path):
    print(f"  [image: {path}]")
    if not path.exists():
        print("  [image not found — continuing without preview]")
        return
    try:
        if sys.platform == 'darwin':
            subprocess.Popen(['open', str(path)])
        elif sys.platform == 'win32':
            os.startfile(str(path))
        else:
            subprocess.Popen(['xdg-open', str(path)])
    except Exception as e:
        print(f"  [could not open image: {e}]")


def grid_paths(clusters_dir: Path, cluster_id: int) -> list[Path]:
    """Return all spectrogram grid pages for a cluster, sorted."""
    subdir = clusters_dir / ('noise' if cluster_id == -1 else f'cluster_{cluster_id}')
    single = subdir / 'spectrogram_grid.png'
    if single.exists():
        return [single]
    pages = sorted(subdir.glob('spectrogram_grid_*.png'))
    return pages


# ── Load clusters CSV ─────────────────────────────────────────────────────────

def load_clusters_csv(clusters_dir: Path) -> pd.DataFrame:
    p = clusters_dir / "clusters.csv"
    if not p.exists():
        raise FileNotFoundError(f"clusters.csv not found in {clusters_dir}")
    return pd.read_csv(p)


# ── Per-pass interactive review ───────────────────────────────────────────────

def review_pass(df: pd.DataFrame, clusters_dir: Path, pass_n: int) -> dict:
    """
    Walk through each cluster interactively. Returns {cluster_id: label}.
    Navigation: sequential by default, type a cluster number to jump, b=back, d=done.
    """
    cluster_ids = sorted(df['cluster'].unique())
    n_total     = len(df)

    print(f"\n{'═'*62}")
    print(f"  Pass {pass_n}  ·  {len(cluster_ids)} clusters  ·  {n_total} windows")
    print(f"{'═'*62}")

    # Overview table
    print()
    for c in cluster_ids:
        n    = len(df[df['cluster'] == c])
        name = "Noise   " if c == -1 else f"Cluster {c:3d}"
        print(f"  {name}: {n:5d} windows  {bar(n, n_total)}")

    print()
    print("  Labels : [k]eep  [r]emove  [?] help")
    print("  Nav    : Enter=next  b=back  d=done(skip rest)  <number>=jump  n/p=next/prev page")
    print()

    labels: dict[int, str] = {}
    idx = 0

    while idx < len(cluster_ids):
        c   = cluster_ids[idx]
        n   = len(df[df['cluster'] == c])
        pos = f"{idx + 1}/{len(cluster_ids)}"

        current = f"  [{labels[c]}]" if c in labels else ""
        name    = "Noise" if c == -1 else f"Cluster {c}"
        print(f"{'─'*62}")
        print(f"  [{pos}] {name} — {n} windows{current}")

        pages    = grid_paths(clusters_dir, c)
        page_idx = 0
        if pages:
            open_image(pages[0])
            if len(pages) > 1:
                print(f"  Page 1/{len(pages)} — n=next page  p=prev page")
        else:
            print("  [no spectrogram grid found]")

        while True:
            raw = input("  > ").strip().lower()

            if raw == 'n':
                if len(pages) > 1:
                    page_idx = (page_idx + 1) % len(pages)
                    open_image(pages[page_idx])
                    print(f"  Page {page_idx + 1}/{len(pages)}")
                else:
                    print("  Only one page.")
                continue

            elif raw == 'p':
                if len(pages) > 1:
                    page_idx = (page_idx - 1) % len(pages)
                    open_image(pages[page_idx])
                    print(f"  Page {page_idx + 1}/{len(pages)}")
                else:
                    print("  Only one page.")
                continue

            elif raw == '?':
                print("    k=keep  r=remove  Enter=skip/confirm current label")
                print("    b=back to previous  d=done (skip all remaining unlabeled)")
                print("    n=next page  p=prev page  <number>=jump to that cluster")

            elif raw == 'b':
                idx = max(0, idx - 1)
                break

            elif raw == 'd':
                for remaining in cluster_ids[idx:]:
                    if remaining not in labels:
                        labels[remaining] = 'skip'
                idx = len(cluster_ids)
                break

            elif raw == '':
                if c in labels:
                    print(f"    keeping: {labels[c]}")
                else:
                    labels[c] = 'skip'
                    print("    → skip")
                idx += 1
                break

            elif raw in LABEL_MAP:
                labels[c] = LABEL_MAP[raw]
                print(f"    → {LABEL_MAP[raw]}")
                idx += 1
                break

            elif raw.lstrip('-').isdigit():
                target = int(raw)
                if target in cluster_ids:
                    idx = cluster_ids.index(target)
                    break
                else:
                    print(f"    Cluster {target} not in this pass. Available: {cluster_ids}")

            else:
                print(f"    Unknown input '{raw}'. Type ? for help.")

    return labels


# ── Summary after review ──────────────────────────────────────────────────────

def show_summary(labels: dict, df: pd.DataFrame):
    cluster_ids = sorted(df['cluster'].unique())
    print(f"\n{'═'*62}")
    print("  Label summary:")
    for lbl in ['keep', 'skip', 'remove']:
        cs = [c for c in cluster_ids if labels.get(c) == lbl]
        if not cs:
            continue
        n_win = sum(int((df['cluster'] == c).sum()) for c in cs)
        ids   = ', '.join('Noise' if c == -1 else str(c) for c in cs)
        print(f"  {lbl:12s}: {len(cs):2d} cluster(s)  ({n_win:5d} windows)  clusters=[{ids}]")

    keep_ids = [c for c in cluster_ids if labels.get(c) not in REMOVE_LABELS]
    n_keep   = sum(int((df['cluster'] == c).sum()) for c in keep_ids)
    print(f"\n  Keeping {n_keep}/{len(df)} windows  ({len(df) - n_keep} removed)")


# ── Next action prompt ────────────────────────────────────────────────────────

def ask_action(min_cluster_size: int) -> tuple[str, int]:
    t = min_cluster_size + 3
    l = max(3, min_cluster_size - 3)
    print(f"\n{'═'*62}")
    print("  What next?")
    print(f"  [r] Re-cluster   UMAP re-embed + HDBSCAN  min_cluster_size={min_cluster_size}")
    print(f"  [t] Tighter      same, but min_cluster_size={t}")
    print(f"  [l] Looser       same, but min_cluster_size={l}")
    print(f"  [c] Custom       enter a specific min_cluster_size")
    print(f"  [f] Final        ensemble clustering on remaining windows")
    print(f"  [s] Save         write filtered windows CSV and exit")
    print(f"  [q] Quit         exit without saving")
    print(f"{'═'*62}")
    while True:
        raw = input("  > ").strip().lower()
        if raw == 'r':
            return 'recluster', min_cluster_size
        elif raw == 't':
            return 'recluster', t
        elif raw == 'l':
            return 'recluster', l
        elif raw == 'c':
            v = input(f"  min_cluster_size [{min_cluster_size}]: ").strip()
            try:
                return 'recluster', int(v) if v else min_cluster_size
            except ValueError:
                print("  Must be an integer.")
        elif raw == 'f':
            return 'final', min_cluster_size
        elif raw == 's':
            return 'save', min_cluster_size
        elif raw == 'q':
            return 'quit', min_cluster_size
        else:
            print("  Unknown. Use r/t/l/c/f/s/q.")


# ── Subprocess wrappers ───────────────────────────────────────────────────────

def _run(cmd: list[str]) -> bool:
    print(f"\n  $ {' '.join(str(x) for x in cmd)}\n")
    return subprocess.run(cmd, cwd=str(ROOT)).returncode == 0


def run_reduction(filter_csv: Path, out: Path, args, plot: bool = False) -> bool:
    """Re-embed the filtered windows using PCA or UMAP."""
    if args.reduction == 'pca':
        cmd = [
            sys.executable, str(ROOT / 'analysis' / 'pca_sliding_window.py'),
            '--npz-root',    str(args.npz_root),
            '--output-root', str(out),
            '--window-secs', str(args.window_secs),
            '--stride-secs', str(args.stride_secs),
            '--mel-start',   str(args.mel_start),
            '--n-components',str(args.n_components),
            '--pca-method',  args.pca_method,
            '--filter-csv',  str(filter_csv),
        ]
        if args.mel_end:
            cmd += ['--mel-end', str(args.mel_end)]
    else:
        cmd = [
            sys.executable, str(ROOT / 'analysis' / 'umap_experiment.py'),
            '--npz-root',    str(args.npz_root),
            '--output-root', str(out),
            '--window-secs', str(args.window_secs),
            '--stride-secs', str(args.stride_secs),
            '--mel-start',   str(args.mel_start),
            '--n-components',str(args.n_components),
            '--n-neighbors', str(args.n_neighbors),
            '--min-dist',    str(args.min_dist),
            '--feature-mode',args.feature_mode,
            '--filter-csv',  str(filter_csv),
        ]
        if args.mel_end:
            cmd += ['--mel-end', str(args.mel_end)]
    if not plot:
        cmd += ['--no-plot']
    return _run(cmd)


def run_hdbscan(pca_root: Path, cluster_out: Path,
                min_cluster_size: int, args) -> bool:
    cmd = [
        sys.executable, str(ROOT / 'clustering' / 'cluster.py'),
        '--pca-root',         str(pca_root),
        '--output-root',      str(cluster_out),
        '--algorithm',        'hdbscan',
        '--min-cluster-size', str(min_cluster_size),
        '--npz-root',         str(args.npz_root),
        '--mel-start',        str(args.mel_start),
    ]
    if args.mel_end:
        cmd += ['--mel-end', str(args.mel_end)]
    if args.min_samples:
        cmd += ['--min-samples', str(args.min_samples)]
    return _run(cmd)


def run_ensemble(pca_root: Path, ensemble_out: Path, args) -> bool:
    cmd = [
        sys.executable, str(ROOT / 'clustering' / 'ensemble_cluster.py'),
        '--pca-root',    str(pca_root),
        '--output-root', str(ensemble_out),
        '--npz-root',    str(args.npz_root),
        '--mel-start',   str(args.mel_start),
        '--n-runs',      '100',
        '--k-min',       '2',
        '--k-max',       '15',
        '--min-cluster-size', str(args.min_cluster_size),
    ]
    if args.mel_end:
        cmd += ['--mel-end', str(args.mel_end)]
    if args.min_samples:
        cmd += ['--min-samples', str(args.min_samples)]
    return _run(cmd)


# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Interactive iterative cluster review for narwhal call detection.")
    # Required paths
    ap.add_argument('--clusters-dir', required=True,
                    help="Directory from cluster.py (contains CSV + cluster_X/ grids)")
    ap.add_argument('--npz-root', required=True,
                    help="Spectrogram .npz directory (used for re-embedding and grids)")
    ap.add_argument('--output-root', required=True,
                    help="Where to write pass_1/, pass_2/, etc.")
    # Reduction method
    ap.add_argument('--reduction', default='pca', choices=['pca', 'umap'],
                    help="Dimensionality reduction for re-embedding (default: pca)")
    # Shared re-embedding params
    ap.add_argument('--window-secs',  type=float, default=5.0)
    ap.add_argument('--stride-secs',  type=float, default=5.0)
    ap.add_argument('--mel-start',    type=int,   default=11)
    ap.add_argument('--mel-end',      type=int,   default=None)
    ap.add_argument('--n-components', type=int,   default=10,
                    help="PCA/UMAP output dimensions (default: 10)")
    # PCA-specific
    ap.add_argument('--pca-method', default='mean_std', choices=['mean_std', 'full_window'],
                    help="Feature type for PCA (default: mean_std)")
    # UMAP-specific (ignored when --reduction pca)
    ap.add_argument('--n-neighbors',  type=int,   default=15)
    ap.add_argument('--min-dist',     type=float, default=0.1)
    ap.add_argument('--feature-mode', default='mean_std',
                    choices=['mean_std', 'acoustic', 'extended_acoustic', 'mfcc', 'passthrough'])
    # HDBSCAN params
    ap.add_argument('--min-cluster-size', type=int, default=8)
    ap.add_argument('--min-samples',      type=int, default=None)
    args = ap.parse_args()

    clusters_dir = Path(args.clusters_dir)
    output_root  = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    min_cluster_size     = args.min_cluster_size
    current_clusters_dir = clusters_dir
    pass_n               = 1

    print()
    print("  ╔══════════════════════════════════════════════════════════╗")
    print("  ║       Iterative Cluster Review — Narwhal Detection       ║")
    print("  ╚══════════════════════════════════════════════════════════╝")
    print(f"  Starting clusters : {clusters_dir}")
    print(f"  Output root       : {output_root}")
    print(f"  NPZ root          : {args.npz_root}")

    while True:
        # Load current clusters
        try:
            df = load_clusters_csv(current_clusters_dir)
        except FileNotFoundError as e:
            print(f"\n[error] {e}")
            sys.exit(1)

        print(f"\n  Loaded {len(df)} windows from {current_clusters_dir}")

        # Interactive review
        labels = review_pass(df, current_clusters_dir, pass_n)

        # Summary
        show_summary(labels, df)

        # Persist labels
        labels_file = output_root / f"pass_{pass_n}_labels.json"
        with open(labels_file, 'w') as f:
            json.dump({str(k): v for k, v in labels.items()}, f, indent=2)
        print(f"\n  [labels saved → {labels_file}]")

        # Decide action
        action, min_cluster_size = ask_action(min_cluster_size)

        if action == 'quit':
            print("\n  Exiting without further changes.")
            sys.exit(0)

        # Build filtered CSV for the kept windows
        keep_ids    = [c for c in df['cluster'].unique()
                       if labels.get(c) not in REMOVE_LABELS]
        filtered_df = df[df['cluster'].isin(keep_ids)].copy()
        pass_dir    = output_root / f"pass_{pass_n}"
        pass_dir.mkdir(exist_ok=True)
        filtered_csv = pass_dir / 'kept_windows.csv'
        filtered_df[['File', 'Start Time (s)']].to_csv(filtered_csv, index=False)
        print(f"\n  [{len(filtered_df)} windows kept → {filtered_csv}]")

        if action == 'save':
            print(f"\n  Done. Filtered windows CSV: {filtered_csv}")
            sys.exit(0)

        # Re-embed filtered windows with PCA or UMAP
        reduction_dir = pass_dir / args.reduction
        is_final      = action == 'final'
        print(f"\n  Re-embedding {len(filtered_df)} windows with {args.reduction.upper()}...")
        if not run_reduction(filtered_csv, reduction_dir, args, plot=is_final):
            print(f"[error] {args.reduction.upper()} step failed — check output above.")
            sys.exit(1)

        if is_final:
            ensemble_dir = pass_dir / 'ensemble'
            print(f"\n  Running final ensemble clustering (100 runs)...")
            if not run_ensemble(reduction_dir, ensemble_dir, args):
                print("[error] Ensemble clustering failed — check output above.")
                sys.exit(1)
            print(f"\n  Final clustering complete → {ensemble_dir}")
            print("  Entering review of final clusters...")
            current_clusters_dir = ensemble_dir
        else:
            # action == 'recluster'
            cluster_dir = pass_dir / 'clusters'
            print(f"\n  Clustering with HDBSCAN (min_cluster_size={min_cluster_size})...")
            if not run_hdbscan(reduction_dir, cluster_dir, min_cluster_size, args):
                print("[error] Clustering step failed — check output above.")
                sys.exit(1)
            print(f"\n  Pass {pass_n} done → {cluster_dir}")
            current_clusters_dir = cluster_dir

        pass_n += 1


if __name__ == "__main__":
    main()
