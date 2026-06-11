"""
Interactive iterative cluster review for narwhal call detection.

Usage:
    python clustering/interactive_cluster_review.py \
        --pca-root    output/mixedDataset/melBins/towsey/pca_mfcc \
        --npz-root    output/mixedDataset/melBins/towsey/npz \
        --output-root output/iterative_review \
        --strategy d --mel-start 11 --mel-end 128

Strategies (--strategy, also switchable per pass):
  h = UMAP + MFCC + HDBSCAN
  d = PCA  + MFCC + DPMM
  k = PCA  + MFCC + KMeans
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

# Named clustering strategies, shown in the ask_action() sub-prompt.
# Each entry sets the reduction method, feature mode, and clustering algorithm.
STRATEGIES = {
    'h': dict(reduction='umap',  feature_mode='mfcc', pca_method='mfcc', algorithm='hdbscan',
              desc='UMAP + MFCC + HDBSCAN'),
    'd': dict(reduction='pca',   feature_mode='mfcc', pca_method='mfcc', algorithm='dpmm',
              desc='PCA  + MFCC + DPMM'),
    'k': dict(reduction='pca',   feature_mode='mfcc', pca_method='mfcc', algorithm='kmeans',
              desc='PCA  + MFCC + KMeans'),
}


def open_image(path: Path):
    print(f"  [image: {path}]")
    if not path.exists():
        print("  [image not found, no preview]")
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


def load_clusters_csv(clusters_dir: Path) -> pd.DataFrame:
    p = clusters_dir / "clusters.csv"
    if not p.exists():
        raise FileNotFoundError(f"clusters.csv not found in {clusters_dir}")
    return pd.read_csv(p)


def review_pass(df: pd.DataFrame, clusters_dir: Path, pass_n: int,
                session_labels: set) -> tuple[dict, dict]:
    cluster_ids = sorted(df['cluster'].unique())
    n_total     = len(df)
    custom_keys = ' / '.join(sorted(session_labels)) if session_labels else ''

    print(f"\nPass {pass_n}: {len(cluster_ids)} clusters  {n_total} windows")
    for c in cluster_ids:
        n    = len(df[df['cluster'] == c])
        name = "Noise" if c == -1 else f"Cluster {c}"
        print(f"  {name}: {n} windows")

    print(f"\nLabels: [k]eep  [r]emove  or type a name then choose keep/remove")
    if custom_keys:
        print(f"  Labels used so far: {custom_keys}")
    print("Nav:    Enter=advance (keep)  b=back  d=done (keep rest)  <number>=jump  n/p=page  ?=help\n")

    labels: dict[int, str] = {}
    types:  dict[int, str] = {}
    idx = 0

    while idx < len(cluster_ids):
        c   = cluster_ids[idx]
        n   = len(df[df['cluster'] == c])
        pos = f"{idx + 1}/{len(cluster_ids)}"

        if c in labels:
            type_suffix = f": {types[c]}" if c in types else ""
            current = f"  [{labels[c]}{type_suffix}]"
        else:
            current = ""
        name = "Noise" if c == -1 else f"Cluster {c}"
        print(f"[{pos}] {name}: {n} windows{current}")
        pages    = grid_paths(clusters_dir, c)
        page_idx = 0
        if pages:
            open_image(pages[0])
            if len(pages) > 1:
                print(f"  Page 1/{len(pages)}: n=next page  p=prev page")
        else:
            print("  [no spectrogram grid found]")

        while True:
            raw = input("  > ").strip().lower()

            if raw == 'n':
                if len(pages) > 1:
                    page_idx = (page_idx + 1) % len(pages)
                    open_image(pages[page_idx])
                    print(f"  page {page_idx + 1}/{len(pages)}")
                else:
                    print("  only one page")
                continue

            elif raw == 'p':
                if len(pages) > 1:
                    page_idx = (page_idx - 1) % len(pages)
                    open_image(pages[page_idx])
                    print(f"  page {page_idx + 1}/{len(pages)}")
                else:
                    print("  only one page")
                continue

            elif raw == '?':
                print("  k=keep  r=remove  Enter=confirm/advance (default: keep)")
                print("  any other text = label cluster, then prompted for keep/remove")
                print("  b=back  d=done (keep rest)  n/p=page  <number>=jump")

            elif raw == 'b':
                idx = max(0, idx - 1)
                break

            elif raw == 'd':
                idx = len(cluster_ids)
                break

            elif raw == '':
                if c in labels:
                    type_suffix = f": {types[c]}" if c in types else ""
                    print(f"  {labels[c]}{type_suffix}")
                idx += 1
                break

            elif raw in LABEL_MAP:
                labels[c] = LABEL_MAP[raw]
                print(f"  {LABEL_MAP[raw]}")
                idx += 1
                break

            elif raw.lstrip('-').isdigit():
                target = int(raw)
                if target in cluster_ids:
                    idx = cluster_ids.index(target)
                    break
                else:
                    print(f"  cluster {target} not found. Available: {cluster_ids}")

            else:
                # treat as free-text label
                types[c] = raw
                session_labels.add(raw)
                kr = input(f"  labelled [{raw}]: keep or remove? ([k]/r): ").strip().lower()
                labels[c] = 'remove' if kr == 'r' else 'keep'
                print(f"  {labels[c]}  [{raw}]")
                idx += 1
                break

    return labels, types


def show_summary(labels: dict, types: dict, df: pd.DataFrame):
    cluster_ids = sorted(df['cluster'].unique())
    keep_ids   = [c for c in cluster_ids if labels.get(c) != 'remove']
    remove_ids = [c for c in cluster_ids if labels.get(c) == 'remove']
    n_keep     = sum(int((df['cluster'] == c).sum()) for c in keep_ids)

    print("\nLabel summary:")
    if remove_ids:
        n_rem = sum(int((df['cluster'] == c).sum()) for c in remove_ids)
        ids   = ', '.join('Noise' if c == -1 else str(c) for c in remove_ids)
        print(f"  remove: {len(remove_ids)} cluster(s), {n_rem} windows  [{ids}]")
    if keep_ids:
        ids = ', '.join(
            ('Noise' if c == -1 else str(c)) + (f"={types[c]}" if c in types else "")
            for c in keep_ids
        )
        print(f"  keep:   {len(keep_ids)} cluster(s), {n_keep} windows  [{ids}]")
    print(f"  {n_keep}/{len(df)} windows pass to next pass")


def save_type_annotations(df: pd.DataFrame, types: dict, csv_path: Path):
    """
    Merge typed-cluster annotations into the central annotations CSV.
    Only windows belonging to a typed cluster are written.
    Existing rows with the same (File, Start Time (s)) are overwritten.
    """
    typed_clusters = [c for c in df['cluster'].unique() if c in types]
    if not typed_clusters:
        return

    new_rows = df[df['cluster'].isin(typed_clusters)][['File', 'Start Time (s)', 'cluster']].copy()
    new_rows['type'] = new_rows['cluster'].map(types)
    new_rows = new_rows.drop(columns=['cluster'])

    if csv_path.exists():
        existing = pd.read_csv(csv_path)
        merged = pd.concat([existing, new_rows], ignore_index=True)
        merged = merged.drop_duplicates(subset=['File', 'Start Time (s)'], keep='last')
    else:
        merged = new_rows

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(csv_path, index=False)
    print(f"  [annotations saved to {csv_path}  ({len(new_rows)} new rows, {len(merged)} total)]")


def _ask_strategy(current_algorithm: str) -> str | None:
    current_key  = next((k for k, v in STRATEGIES.items() if v['algorithm'] == current_algorithm), '?')
    current_desc = STRATEGIES.get(current_key, {}).get('desc', current_algorithm)
    strat_lines  = '  '.join(f"[{k}] {v['desc']}" for k, v in STRATEGIES.items())
    print(f"  strategy (current: {current_desc})")
    print(f"  {strat_lines}  [Enter] keep current")
    while True:
        raw = input("  > ").strip().lower()
        if raw == '':
            return None
        if raw in STRATEGIES:
            return raw
        print(f"  use {'/'.join(STRATEGIES)} or Enter")


def ask_action(args) -> tuple[str, str | None]:
    """Prompt for next action. Mutates args in-place for algorithm-specific parameter changes."""
    if args.algorithm == 'hdbscan':
        param, label, current = 'min_cluster_size', 'mcs', args.min_cluster_size
    elif args.algorithm == 'kmeans':
        param, label, current = 'n_clusters', 'k', args.n_clusters
    else:  # dpmm
        param, label, current = 'dpmm_max_components', 'max_comp', args.dpmm_max_components

    t = current + 3
    l = max(2, current - 3)

    print(f"\nWhat next?  ({args.algorithm.upper()}  {label}={current})")
    print(f"  [r] re-cluster  [t] tighter ({label}={t})  [l] looser ({label}={l})  [c] custom {label}")
    print(f"  [s] save + exit  [q] quit")
    while True:
        raw = input("> ").strip().lower()
        if raw == 'r':
            return 'recluster', _ask_strategy(args.algorithm)
        elif raw == 't':
            setattr(args, param, t)
            return 'recluster', _ask_strategy(args.algorithm)
        elif raw == 'l':
            setattr(args, param, l)
            return 'recluster', _ask_strategy(args.algorithm)
        elif raw == 'c':
            v = input(f"{label} [{current}]: ").strip()
            try:
                setattr(args, param, int(v) if v else current)
            except ValueError:
                print("  must be an integer")
                continue
            return 'recluster', _ask_strategy(args.algorithm)
        elif raw == 's':
            return 'save', None
        elif raw == 'q':
            return 'quit', None
        else:
            print("  use r/t/l/c/s/q")


def _run(cmd: list[str]) -> bool:
    print(f"\n  $ {' '.join(str(x) for x in cmd)}\n")
    return subprocess.run(cmd, cwd=str(ROOT)).returncode == 0


def run_reduction(filter_csv: Path, out: Path, args, plot: bool = False) -> bool:
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
            sys.executable, str(ROOT / 'analysis' / 'umap_embedding.py'),
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


def run_cluster_step(pca_root: Path, cluster_out: Path,
                     min_cluster_size: int, args) -> bool:
    cmd = [
        sys.executable, str(ROOT / 'clustering' / 'cluster.py'),
        '--pca-root',         str(pca_root),
        '--output-root',      str(cluster_out),
        '--algorithm',        args.algorithm,
        '--min-cluster-size', str(min_cluster_size),
        '--npz-root',         str(args.npz_root),
        '--mel-start',        str(args.mel_start),
    ]
    if args.mel_end:
        cmd += ['--mel-end', str(args.mel_end)]
    if args.algorithm == 'dpmm':
        cmd += ['--dpmm-max-components', str(args.dpmm_max_components),
                '--dpmm-concentration',  str(args.dpmm_concentration)]
    if args.algorithm == 'kmeans':
        cmd += ['--n-clusters', str(args.n_clusters)]
    return _run(cmd)



def main():
    ap = argparse.ArgumentParser(
        description="Interactive iterative cluster review for narwhal call detection.")
    ap.add_argument('--pca-root', required=True,
                    help="Directory containing pca_results.npz. Initial clustering runs "
                         "automatically, then the interactive review begins.")
    ap.add_argument('--npz-root', required=True,
                    help="Spectrogram .npz directory (used for re-embedding and grids)")
    ap.add_argument('--output-root', required=True,
                    help="Where to write pass_1/, pass_2/, etc.")
    # Named strategy: sets reduction, algorithm, and feature mode
    strat_help = '  '.join(f"{k}={v['desc']}" for k, v in STRATEGIES.items())
    ap.add_argument('--strategy', default=None, choices=list(STRATEGIES),
                    help=f"Starting clustering strategy (overrides --reduction/--algorithm/--pca-method/--feature-mode). "
                         f"Switchable per pass in-session. Options: {strat_help}")
    # Reduction method (overridden by --strategy if given)
    ap.add_argument('--reduction', default='pca', choices=['pca', 'umap'],
                    help="Dimensionality reduction for re-embedding; overridden by --strategy (default: pca)")
    # Shared re-embedding params
    ap.add_argument('--window-secs',  type=float, default=5.0)
    ap.add_argument('--stride-secs',  type=float, default=5.0)
    ap.add_argument('--mel-start',    type=int,   default=11)
    ap.add_argument('--mel-end',      type=int,   default=None)
    ap.add_argument('--n-components', type=int,   default=20,
                    help="PCA/UMAP output dimensions (default: 20)")
    # PCA-specific (overridden by --strategy if given)
    ap.add_argument('--pca-method', default='mfcc', choices=['mean_std', 'full_window', 'mfcc'],
                    help="Feature type for PCA (default: mfcc)")
    # UMAP-specific (overridden by --strategy if given)
    ap.add_argument('--n-neighbors',  type=int,   default=15)
    ap.add_argument('--min-dist',     type=float, default=0.1)
    ap.add_argument('--feature-mode', default='mfcc',
                    choices=['mean_std', 'mfcc'],
                    help="Feature mode for UMAP embedding (default: mfcc)")
    # Clustering algorithm (overridden by --strategy if given)
    ap.add_argument('--algorithm', default='dpmm',
                    choices=['hdbscan', 'dpmm', 'kmeans'],
                    help="Clustering algorithm; overridden by --strategy (default: dpmm)")
    ap.add_argument('--n-clusters', type=int, default=10,
                    help="k for kmeans (default: 10)")
    # Type annotation output
    ap.add_argument('--type-annotations-csv',
                    default=None,
                    help="CSV for accumulated type labels (default: <output-root>/type_annotations.csv)")
    # HDBSCAN params
    ap.add_argument('--min-cluster-size', type=int, default=8)
    # DPMM params
    ap.add_argument('--dpmm-max-components', type=int,   default=20,
                    help="DPMM upper bound on components (default: 20)")
    ap.add_argument('--dpmm-concentration',  type=float, default=0.01,
                    help="DPMM concentration α (default: 0.01)")
    args = ap.parse_args()

    if args.strategy:
        s = STRATEGIES[args.strategy]
        args.reduction    = s['reduction']
        args.algorithm    = s['algorithm']
        args.feature_mode = s['feature_mode']
        args.pca_method   = s['pca_method']

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if args.type_annotations_csv is None:
        args.type_annotations_csv = str(output_root / 'type_annotations.csv')

    print(f"output: {output_root}")
    print(f"npz:    {args.npz_root}")

    init_clusters_dir = output_root / "clusters_init"
    print(f"\nRunning initial clustering ({args.algorithm}, mcs={args.min_cluster_size})...")
    if not run_cluster_step(Path(args.pca_root), init_clusters_dir, args.min_cluster_size, args):
        print("[error] Initial clustering failed.")
        sys.exit(1)
    current_clusters_dir = init_clusters_dir

    pass_n       = 1
    session_labels: set = set()

    while True:
        # Load current clusters
        try:
            df = load_clusters_csv(current_clusters_dir)
        except FileNotFoundError as e:
            print(f"\n[error] {e}")
            sys.exit(1)

        print(f"\nLoaded {len(df)} windows from {current_clusters_dir}")

        labels, types = review_pass(df, current_clusters_dir, pass_n, session_labels)

        # Summary
        show_summary(labels, types, df)

        # Persist labels + types
        labels_file = output_root / f"pass_{pass_n}_labels.json"
        with open(labels_file, 'w') as f:
            json.dump({
                'labels': {str(k): v for k, v in labels.items()},
                'types':  {str(k): v for k, v in types.items()},
            }, f, indent=2)
        print(f"  [labels saved to {labels_file}]")

        # Save type annotations to central CSV
        save_type_annotations(df, types, Path(args.type_annotations_csv))

        action, strategy_key = ask_action(args)
        if strategy_key:
            s = STRATEGIES[strategy_key]
            args.reduction    = s['reduction']
            args.algorithm    = s['algorithm']
            args.feature_mode = s['feature_mode']
            args.pca_method   = s['pca_method']
            print(f"  strategy: {s['desc']}")

        if action == 'quit':
            sys.exit(0)

        # Build filtered CSV for the kept windows
        keep_ids    = [c for c in df['cluster'].unique()
                       if labels.get(c) != 'remove']
        filtered_df = df[df['cluster'].isin(keep_ids)].copy()
        pass_dir    = output_root / f"pass_{pass_n}"
        pass_dir.mkdir(exist_ok=True)
        filtered_csv = pass_dir / 'kept_windows.csv'
        filtered_df[['File', 'Start Time (s)']].to_csv(filtered_csv, index=False)
        print(f"  [{len(filtered_df)} windows kept: {filtered_csv}]")

        if action == 'save':
            sys.exit(0)

        reduction_dir = pass_dir / args.reduction
        print(f"\nRe-embedding {len(filtered_df)} windows ({args.reduction}, {args.pca_method})...")
        if not run_reduction(filtered_csv, reduction_dir, args, plot=False):
            print(f"[error] {args.reduction.upper()} step failed")
            sys.exit(1)

        cluster_dir = pass_dir / 'clusters'
        print(f"\nClustering ({args.algorithm}, mcs={args.min_cluster_size})...")
        if not run_cluster_step(reduction_dir, cluster_dir, args.min_cluster_size, args):
            print("[error] Clustering step failed")
            sys.exit(1)
        current_clusters_dir = cluster_dir

        pass_n += 1


if __name__ == "__main__":
    main()
