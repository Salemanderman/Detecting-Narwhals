"""
Interactive cluster review tool — iterative narwhal call clustering for biologists.

Shows each cluster's spectrogram grid, lets you label it, then offers to re-cluster
(UMAP re-embed + HDBSCAN) on the cleaned subset, or run final ensemble clustering.
All passes are saved under output-root/pass_N/ so nothing is lost.

Usage:
    # Start with DPMM strategy (PCA + MFCC + DPMM):
    python clustering/interactive_cluster_review.py \
        --pca-root    output/mixedDataset/melBins/towsey/pca_mfcc \
        --npz-root    output/mixedDataset/melBins/towsey/npz \
        --output-root output/iterative_review \
        --strategy d \
        --mel-start 11 --mel-end 128

    # Start with HDBSCAN strategy (UMAP + MFCC + HDBSCAN):
    python clustering/interactive_cluster_review.py \
        --pca-root    output/mixedDataset/melBins/towsey/pca_mfcc \
        --npz-root    output/mixedDataset/melBins/towsey/npz \
        --output-root output/iterative_review \
        --strategy h

Strategies (--strategy / switchable per pass in-session):
  h = UMAP + MFCC + HDBSCAN
  d = PCA  + MFCC + DPMM
  k = PCA  + MFCC + KMeans

After labelling all clusters, choose what to do next:
  r/t/l/c = re-cluster (prompts for strategy)
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

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "clustering"))

LABEL_MAP = {
    'k': 'keep',
    'r': 'remove',
}
REMOVE_LABELS = {'remove'}

# Named clustering strategies — shown in the ask_action() sub-prompt.
# Each entry sets the reduction method, feature mode, and clustering algorithm.
STRATEGIES = {
    'h': dict(reduction='umap',  feature_mode='mfcc', pca_method='mfcc', algorithm='hdbscan',
              desc='UMAP + MFCC + HDBSCAN'),
    'd': dict(reduction='pca',   feature_mode='mfcc', pca_method='mfcc', algorithm='dpmm',
              desc='PCA  + MFCC + DPMM'),
    'k': dict(reduction='pca',   feature_mode='mfcc', pca_method='mfcc', algorithm='kmeans',
              desc='PCA  + MFCC + KMeans'),
}

# Recognised type keywords — typing any of these assigns the type and also determines
# whether the cluster is kept or removed.
# Extend freely; keys are what the user types, values are the canonical stored names.
TYPE_MAP = {
    'clicks':  'clicks',
    'tonal':   'tonal',
    'burst':   'burst-pulse',
    'noise':   'noise',
    'ice':     'ice-noise',
    'mixed':   'mixed',
}
# These types are tagged in the annotations CSV but removed from the iterative pipeline.
REMOVE_TYPES = {'noise', 'ice'}


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

def review_pass(df: pd.DataFrame, clusters_dir: Path, pass_n: int,
                cluster_preds: dict = None) -> tuple[dict, dict]:
    """
    Walk through each cluster interactively.
    Returns (labels, types) where:
      labels: {cluster_id: 'keep'|'remove'|'skip'}
      types:  {cluster_id: canonical_type_string}  — only for typed clusters
    cluster_preds: optional {cluster_id: {label, confidence, detail}} from classifier.
    Navigation: sequential by default, type a cluster number to jump, b=back, d=done.
    """
    cluster_preds = cluster_preds or {}
    cluster_ids = sorted(df['cluster'].unique())
    n_total     = len(df)
    type_keys   = '  /  '.join(TYPE_MAP.keys())

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
    print(f"  Labels : [k]eep  [r]emove  or a type name (keeps + tags): {type_keys}")
    print(f"  Nav    : Enter=next  b=back  d=done(skip rest)  <number>=jump  n/p=next/prev page  [?] help")
    print()

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
        print(f"{'─'*62}")
        print(f"  [{pos}] {name} — {n} windows{current}")
        if c in cluster_preds:
            p = cluster_preds[c]
            print(f"  Model → {p['label']}  conf={p['confidence']:.0%}  {p['detail']}")

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
                print(f"    type name = keep + tag: {type_keys}")
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
                    type_suffix = f": {types[c]}" if c in types else ""
                    print(f"    keeping: {labels[c]}{type_suffix}")
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

            elif raw in TYPE_MAP:
                action    = 'remove' if raw in REMOVE_TYPES else 'keep'
                labels[c] = action
                types[c]  = TYPE_MAP[raw]
                print(f"    → {action}  [type: {TYPE_MAP[raw]}]")
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

    return labels, types


# ── Summary after review ──────────────────────────────────────────────────────

def show_summary(labels: dict, types: dict, df: pd.DataFrame):
    cluster_ids = sorted(df['cluster'].unique())
    print(f"\n{'═'*62}")
    print("  Label summary:")
    for lbl in ['keep', 'skip', 'remove']:
        cs = [c for c in cluster_ids if labels.get(c) == lbl]
        if not cs:
            continue
        n_win = sum(int((df['cluster'] == c).sum()) for c in cs)
        ids   = ', '.join(
            ('Noise' if c == -1 else str(c)) + (f"={types[c]}" if c in types else "")
            for c in cs
        )
        print(f"  {lbl:12s}: {len(cs):2d} cluster(s)  ({n_win:5d} windows)  [{ids}]")

    keep_ids = [c for c in cluster_ids if labels.get(c) not in REMOVE_LABELS]
    n_keep   = sum(int((df['cluster'] == c).sum()) for c in keep_ids)
    typed    = [c for c in keep_ids if c in types]
    if typed:
        type_summary = '  '.join(f"{types[c]}×{int((df['cluster']==c).sum())}" for c in typed)
        print(f"  Types       : {type_summary}")
    print(f"\n  Keeping {n_keep}/{len(df)} windows  ({len(df) - n_keep} removed)")


# ── Type annotation persistence ──────────────────────────────────────────────

def save_type_annotations(df: pd.DataFrame, types: dict, csv_path: Path):
    """
    Merge typed-cluster annotations into the central annotations CSV.
    Only windows belonging to a typed cluster are written.
    Existing rows with the same (File, Start Time (s)) are overwritten.
    """
    typed_clusters = [c for c in df['cluster'].unique() if c in types]
    if not typed_clusters:
        return

    new_rows = df[df['cluster'].isin(typed_clusters)][['File', 'Start Time (s)']].copy()
    new_rows['type'] = new_rows.apply(
        lambda r: types[int(df.loc[r.name, 'cluster'])], axis=1
    )

    if csv_path.exists():
        existing = pd.read_csv(csv_path)
        merged = pd.concat([existing, new_rows], ignore_index=True)
        merged = merged.drop_duplicates(subset=['File', 'Start Time (s)'], keep='last')
    else:
        merged = new_rows

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(csv_path, index=False)
    print(f"\n  [type annotations saved → {csv_path}  ({len(new_rows)} new/updated rows, {len(merged)} total)]")


# ── Next action prompt ────────────────────────────────────────────────────────

def _ask_strategy(current_algorithm: str) -> str | None:
    """Sub-prompt shown after a recluster action. Returns a STRATEGIES key or None to keep current."""
    current_key = next((k for k, v in STRATEGIES.items() if v['algorithm'] == current_algorithm), '?')
    current_desc = STRATEGIES.get(current_key, {}).get('desc', current_algorithm)
    strat_lines  = '  '.join(f"[{k}] {v['desc']}" for k, v in STRATEGIES.items())
    print(f"\n  Strategy  (current: {current_desc})")
    print(f"  {strat_lines}  [Enter] keep current")
    while True:
        raw = input("  > ").strip().lower()
        if raw == '':
            return None
        if raw in STRATEGIES:
            return raw
        print(f"  Use {'/'.join(STRATEGIES)} or Enter to keep current.")


def ask_action(min_cluster_size: int, algorithm: str = 'hdbscan') -> tuple[str, int, str | None]:
    t = min_cluster_size + 3
    l = max(3, min_cluster_size - 3)
    print(f"\n{'═'*62}")
    print("  What next?")
    print(f"  [r] Re-cluster   re-embed + {algorithm.upper()}  min_cluster_size={min_cluster_size}")
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
            return 'recluster', min_cluster_size, _ask_strategy(algorithm)
        elif raw == 't':
            return 'recluster', t, _ask_strategy(algorithm)
        elif raw == 'l':
            return 'recluster', l, _ask_strategy(algorithm)
        elif raw == 'c':
            v = input(f"  min_cluster_size [{min_cluster_size}]: ").strip()
            try:
                mcs = int(v) if v else min_cluster_size
            except ValueError:
                print("  Must be an integer.")
                continue
            return 'recluster', mcs, _ask_strategy(algorithm)
        elif raw == 'f':
            return 'final', min_cluster_size, None
        elif raw == 's':
            return 'save', min_cluster_size, None
        elif raw == 'q':
            return 'quit', min_cluster_size, None
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
    if args.algorithm == 'hdbscan' and args.min_samples:
        cmd += ['--min-samples', str(args.min_samples)]
    if args.algorithm == 'dpmm':
        cmd += ['--dpmm-max-components', str(args.dpmm_max_components),
                '--dpmm-concentration',  str(args.dpmm_concentration)]
    if args.algorithm == 'kmeans':
        cmd += ['--n-clusters', str(args.n_clusters)]
    return _run(cmd)


def run_ensemble(pca_root: Path, ensemble_out: Path, args) -> bool:
    cmd = [
        sys.executable, str(ROOT / 'clustering' / 'ensemble_cluster.py'),
        '--pca-root',         str(pca_root),
        '--output-root',      str(ensemble_out),
        '--npz-root',         str(args.npz_root),
        '--mel-start',        str(args.mel_start),
        '--n-runs',           '100',
        '--k-min',            '2',
        '--k-max',            '15',
        '--min-cluster-size', str(args.min_cluster_size),
        '--base-algorithm',   args.ensemble_base_algorithm,
    ]
    if args.mel_end:
        cmd += ['--mel-end', str(args.mel_end)]
    if args.min_samples:
        cmd += ['--min-samples', str(args.min_samples)]
    if args.ensemble_base_algorithm == 'dpmm':
        cmd += ['--dpmm-max-components', str(args.dpmm_max_components),
                '--dpmm-concentration',  str(args.dpmm_concentration)]
    return _run(cmd)


# ── Optional classifier predictions ──────────────────────────────────────────

def load_classifier(model_path: Path):
    """Load joblib model payload. Returns None silently if joblib is unavailable."""
    try:
        import joblib
        return joblib.load(model_path)
    except Exception as e:
        print(f"  [warn] Could not load classifier: {e}")
        return None


def predict_clusters_for_review(df: pd.DataFrame, npz_root: Path, payload: dict,
                                 confidence_threshold: float = 0.6) -> dict:
    """
    Returns {cluster_id: {'label': str, 'confidence': float, 'detail': str}}
    for display in the interactive review. Never raises — silently skips on error.
    """
    try:
        import utilities.configs as configs
        import utilities.feature_utils as futils
        from clustering_core import mfcc_features
    except Exception:
        return {}

    spec_cfg      = configs.get_specgram_config()
    window_frames = round(payload["window_secs"] * spec_cfg["sample_rate"] / spec_cfg["hop_length"])
    mel_start     = payload["mel_start"]
    mel_end       = payload["mel_end"]
    n_mfcc        = payload["n_mfcc"]
    model         = payload["model"]
    npz_index     = {p.name: p for p in npz_root.glob("*.npz")}

    # Predict per window
    preds, confs, clusters = [], [], []
    for _, row in df.iterrows():
        npz_path = npz_index.get(row["File"])
        if npz_path is None:
            continue
        try:
            win  = futils.get_window(npz_path, float(row["Start Time (s)"]),
                                     window_frames, mel_start, mel_end, spec_cfg)
            feat = mfcc_features(win, n_mfcc=n_mfcc).reshape(1, -1)
            preds.append(model.predict(feat)[0])
            confs.append(float(model.predict_proba(feat).max()))
            clusters.append(row["cluster"])
        except Exception:
            continue

    if not preds:
        return {}

    # Aggregate per cluster
    result = {}
    pred_arr    = np.array(preds)
    conf_arr    = np.array(confs)
    cluster_arr = np.array(clusters)

    for cid in df["cluster"].unique():
        mask = cluster_arr == cid
        if not mask.any():
            continue
        c_preds = pred_arr[mask]
        c_confs = conf_arr[mask]
        types, counts = np.unique(c_preds, return_counts=True)
        top_idx  = counts.argmax()
        top_type = types[top_idx]
        top_frac = counts[top_idx] / mask.sum()
        mean_conf = c_confs.mean()

        flags = []
        if top_frac < 0.7:
            flags.append("mixed")
        if mean_conf < confidence_threshold:
            flags.append("uncertain")

        dist = "  ".join(f"{t}:{c}" for t, c in zip(types, counts))
        detail = f"({dist})" if len(types) > 1 else ""
        flag_str = " ⚑ " + "+".join(flags) if flags else ""

        result[cid] = {
            "label":      top_type,
            "confidence": mean_conf,
            "detail":     f"{detail}{flag_str}",
        }
    return result


# ── Main loop ─────────────────────────────────────────────────────────────────

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
    # Named strategy — sets reduction + algorithm + feature mode together
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
                    choices=['mean_std', 'acoustic', 'extended_acoustic', 'mfcc', 'passthrough'],
                    help="Feature mode for UMAP embedding (default: mfcc)")
    # Clustering algorithm (overridden by --strategy if given)
    ap.add_argument('--algorithm', default='dpmm',
                    choices=['hdbscan', 'dpmm', 'kmeans'],
                    help="Clustering algorithm; overridden by --strategy (default: dpmm)")
    ap.add_argument('--n-clusters', type=int, default=10,
                    help="k for kmeans (default: 10)")
    # Optional classifier (informational only — does not affect keep/remove logic)
    ap.add_argument('--model-path', default=None,
                    help="Trained type classifier (evaluation/type_classifier.joblib). "
                         "Shows predictions per cluster — does not change labelling behaviour.")
    ap.add_argument('--confidence-threshold', type=float, default=0.6,
                    help="Below this confidence, clusters are flagged as uncertain (default: 0.6)")
    # Type annotation output
    ap.add_argument('--type-annotations-csv',
                    default=str(ROOT / 'evaluation' / 'type_annotations.csv'),
                    help="Central CSV for accumulated type labels (default: evaluation/type_annotations.csv)")
    # HDBSCAN params
    ap.add_argument('--min-cluster-size', type=int, default=8)
    ap.add_argument('--min-samples',      type=int, default=None)
    # DPMM params
    ap.add_argument('--dpmm-max-components', type=int,   default=20,
                    help="DPMM upper bound on components (default: 20)")
    ap.add_argument('--dpmm-concentration',  type=float, default=0.01,
                    help="DPMM concentration α — lower = fewer clusters (default: 0.01)")
    # Ensemble final pass
    ap.add_argument('--ensemble-base-algorithm', default='kmeans',
                    choices=['kmeans', 'dpmm'],
                    help="Base algorithm for final ensemble clustering (default: kmeans)")
    args = ap.parse_args()

    if args.strategy:
        s = STRATEGIES[args.strategy]
        args.reduction    = s['reduction']
        args.algorithm    = s['algorithm']
        args.feature_mode = s['feature_mode']
        args.pca_method   = s['pca_method']

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    min_cluster_size = args.min_cluster_size

    # Load optional classifier once at startup
    classifier = None
    if args.model_path:
        classifier = load_classifier(Path(args.model_path))
        if classifier:
            print(f"  Classifier loaded : {args.model_path}  (classes: {classifier['classes']})")

    print()
    print("  ╔══════════════════════════════════════════════════════════╗")
    print("  ║       Iterative Cluster Review — Narwhal Detection       ║")
    print("  ╚══════════════════════════════════════════════════════════╝")
    print(f"  Output root       : {output_root}")
    print(f"  NPZ root          : {args.npz_root}")

    # Run initial clustering on the provided PCA root
    init_clusters_dir = output_root / "clusters_init"
    if args.algorithm == 'dpmm':
        algo_desc = f"DPMM  max_components={args.dpmm_max_components}  α={args.dpmm_concentration}"
    elif args.algorithm == 'kmeans':
        algo_desc = f"kmeans  k={args.n_clusters}"
    else:
        algo_desc = f"HDBSCAN  min_cluster_size={min_cluster_size}"
    print(f"\n  Running initial clustering on {args.pca_root}  [{algo_desc}]...")
    if not run_cluster_step(Path(args.pca_root), init_clusters_dir, min_cluster_size, args):
        print("[error] Initial clustering failed.")
        sys.exit(1)
    current_clusters_dir = init_clusters_dir

    pass_n = 1

    while True:
        # Load current clusters
        try:
            df = load_clusters_csv(current_clusters_dir)
        except FileNotFoundError as e:
            print(f"\n[error] {e}")
            sys.exit(1)

        print(f"\n  Loaded {len(df)} windows from {current_clusters_dir}")

        # Compute classifier predictions (informational only)
        cluster_preds = {}
        if classifier:
            cluster_preds = predict_clusters_for_review(
                df, Path(args.npz_root), classifier, args.confidence_threshold)

        # Interactive review
        labels, types = review_pass(df, current_clusters_dir, pass_n,
                                    cluster_preds=cluster_preds)

        # Summary
        show_summary(labels, types, df)

        # Persist labels + types
        labels_file = output_root / f"pass_{pass_n}_labels.json"
        with open(labels_file, 'w') as f:
            json.dump({
                'labels': {str(k): v for k, v in labels.items()},
                'types':  {str(k): v for k, v in types.items()},
            }, f, indent=2)
        print(f"\n  [labels saved → {labels_file}]")

        # Save type annotations to central CSV
        save_type_annotations(df, types, Path(args.type_annotations_csv))

        # Decide action
        action, min_cluster_size, strategy_key = ask_action(min_cluster_size, args.algorithm)
        if strategy_key:
            s = STRATEGIES[strategy_key]
            args.reduction    = s['reduction']
            args.algorithm    = s['algorithm']
            args.feature_mode = s['feature_mode']
            args.pca_method   = s['pca_method']
            print(f"  Strategy → {s['desc']}")

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
        if args.reduction == 'pca':
            embed_desc = f"PCA  method={args.pca_method}  n_components={args.n_components}"
        else:
            embed_desc = f"UMAP  feature={args.feature_mode}  n_components={args.n_components}  n_neighbors={args.n_neighbors}  min_dist={args.min_dist}"

        if args.algorithm == 'dpmm':
            algo_desc = f"{args.algorithm.upper()}  max_components={args.dpmm_max_components}  α={args.dpmm_concentration}"
        elif args.algorithm == 'hdbscan':
            algo_desc = f"{args.algorithm.upper()}  min_cluster_size={min_cluster_size}"
        else:
            algo_desc = f"{args.algorithm.upper()}  k={args.n_clusters}"

        method_desc = f"{embed_desc}  |  {algo_desc}"
        print(f"\n  Re-embedding {len(filtered_df)} windows  [{embed_desc}]...")
        if not run_reduction(filtered_csv, reduction_dir, args, plot=is_final):
            print(f"[error] {args.reduction.upper()} step failed — check output above.")
            sys.exit(1)

        if is_final:
            ensemble_dir  = pass_dir / 'ensemble'
            ensemble_desc = f"{args.ensemble_base_algorithm.upper()}  100 runs  min_cluster_size={args.min_cluster_size}"
            if args.ensemble_base_algorithm == 'dpmm':
                ensemble_desc += f"  max_components={args.dpmm_max_components}  α={args.dpmm_concentration}"
            print(f"\n  Running final ensemble clustering  [{embed_desc}  |  {ensemble_desc}]...")
            if not run_ensemble(reduction_dir, ensemble_dir, args):
                print("[error] Ensemble clustering failed — check output above.")
                sys.exit(1)
            print(f"\n  Final clustering complete → {ensemble_dir}")
            print("  Entering review of final clusters...")
            current_clusters_dir = ensemble_dir
        else:
            # action == 'recluster'
            cluster_dir = pass_dir / 'clusters'
            print(f"\n  Clustering  [{method_desc}]...")
            if not run_cluster_step(reduction_dir, cluster_dir, min_cluster_size, args):
                print("[error] Clustering step failed — check output above.")
                sys.exit(1)
            print(f"\n  Pass {pass_n} done → {cluster_dir}")
            current_clusters_dir = cluster_dir

        pass_n += 1


if __name__ == "__main__":
    main()
