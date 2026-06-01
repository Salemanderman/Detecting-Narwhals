import argparse
import csv
import numpy as np
from collections import Counter
from pprint import pformat
from pathlib import Path
import sys
import torch
import torchaudio
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import utilities.utils as utils
import utilities.configs as configs


def _to_int(x):
    if isinstance(x, int):
        return x
    if hasattr(x, "item"):
        return int(x.item())
    return int(x)


def _get_base_files(dataset):
    """Return (base_dataset, list_of_file_paths) handling plain Dataset or Subset."""
    if isinstance(dataset, Subset):
        return dataset.dataset, [dataset.dataset.files[i] for i in dataset.indices]
    return dataset, dataset.files


def _filter_by_length(dataset, start_secs, target_sr=64_000):
    """Keep only files whose 1-second-snapped duration matches the most common length."""
    base, files = _get_base_files(dataset)

    lengths = {}
    for p in tqdm(files, desc="Scanning file lengths", unit="file", leave=False):
        try:
            info = torchaudio.info(str(p))
        except Exception as e:
            print(f"  [skip] {Path(p).name}: {e}")
            continue
        sr  = info.sample_rate
        s   = int(start_secs * sr)
        dur = max(0, round((info.num_frames - s) / sr) - 1) * sr  # same formula as AudioDataset
        lengths[str(p)] = int(dur / sr * target_sr)

    if not lengths:
        raise RuntimeError(f"No audio files found under {base.root_dir}")

    mode_len = Counter(lengths.values()).most_common(1)[0][0]
    valid = {p for p, l in lengths.items() if l == mode_len}
    n_skipped = len(files) - len(valid)
    if n_skipped:
        print(f"[info] Skipping {n_skipped} file(s) with unexpected length "
              f"(expected ~{mode_len / target_sr:.0f} s after crop).")

    valid_indices = [i for i, p in enumerate(base.files) if str(p) in valid]
    return Subset(base, valid_indices)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio-root",  required=True, help="Root folder containing audio files.")
    ap.add_argument("--output-root", required=True, help="Output folder for .npz features and index.")
    ap.add_argument("--subset-len", type=int, default=0, help="Optionally limit to a subset of data.")
    ap.add_argument("--towsey", action="store_true", default=False, help="Apply Towsey (2013) modal noise removal after spectrogram computation.")
    ap.add_argument("--towsey-N", type=float, default=0.0, dest="towsey_N", help="Towsey N: std devs above modal background added to threshold (default 0.0).")
    ap.add_argument("--audio-crop-start-secs", type=int, default=5, dest="audio_crop_start_secs", help="Seconds to cut from the start of each recording (default: 5).")
    ap.add_argument("--linear-freq", action="store_true", default=False, help="Use linear frequency scale instead of mel scale.")
    ap.add_argument("--n-mels", type=int, default=None, dest="n_mels", help="Number of mel bins (default: from config). Ignored if --linear-freq is set.")
    ap.add_argument("--num-workers", type=int, default=0, dest="num_workers", help="DataLoader worker processes (default: 4, use 0 on Windows if errors occur).")
    ap.add_argument("--batch-size", type=int, default=32, dest="batch_size", help="Files per GPU batch (default: 32).")
    args = ap.parse_args()

    audio_root  = Path(args.audio_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    start_secs = args.audio_crop_start_secs
    batch_size = args.batch_size
    print(f" [info] cropping first {start_secs} s from start. snapping end to 1-second boundary.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Input:  {audio_root}")
    print(f"Output: {output_root}")
    print(f"Device: {device}")

    # Build dataset, apply optional subset, then filter to standard-length files only
    dataset = utils.AudioDataset(audio_root, target_sr=64_000, start_secs=start_secs)
    if args.subset_len > 0:
        dataset = Subset(dataset, list(range(min(args.subset_len, len(dataset)))))
    dataset = _filter_by_length(dataset, start_secs)

    # All files are now the same length — default collate stacks them without padding
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=args.num_workers, shuffle=False)
    print(f"Files: {len(dataset)}  |  Batches: {len(loader)}")

    # Spectrogram transform
    specgram_config = configs.get_specgram_config()
    if args.linear_freq:
        specgram_config["n_mels"] = None
    elif args.n_mels is not None:
        specgram_config["n_mels"] = args.n_mels
    logmel_transf = utils.PipelineSpecgram(specgram_config=specgram_config).to(device)
    logmel_transf.eval()
    print("Specgram config:\n" + pformat(specgram_config, indent=2, sort_dicts=False))

    index_rows = []
    print("\n[info] Starting feature extraction...")

    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting spectrograms", unit="batch"):
            paths         = batch["path"]
            srs           = batch["sample_rate"]
            waveforms_gpu = batch["waveform"].to(device=device, dtype=torch.float32)

            # Batch GPU compute: (B, 1, T) → (B, 1, bins, T_frames)
            feats = logmel_transf(waveforms_gpu)

            for b in range(waveforms_gpu.size(0)):
                wav_path = Path(paths[b])
                sr_val   = _to_int(srs[b])

                try:
                    feat = feats[b, 0].cpu().numpy()  # (bins, T_frames)
                    if args.towsey:
                        feat = utils.towsey_noise_removal(feat, N=args.towsey_N)
                    feat = torch.from_numpy(feat).unsqueeze(0)  # (1, bins, T_frames)

                    try:
                        out_dir = output_root / wav_path.parent.relative_to(audio_root)
                    except ValueError:
                        out_dir = output_root
                    out_dir.mkdir(parents=True, exist_ok=True)

                    out_path = out_dir / (wav_path.stem + ".npz")
                    np.savez_compressed(
                        str(out_path),
                        feature=feat.numpy(),
                        sr=sr_val,
                        source_path=str(wav_path),
                    )

                    index_rows.append({
                        "source_path":  str(wav_path),
                        "feature_path": str(out_path),
                        "sr":           sr_val,
                        "shape":        list(feat.shape),
                    })

                except Exception as e:
                    print(f"[error] {wav_path}: {e}", file=sys.stderr)

    print(f"[done] Extracted features for {len(index_rows)} files.")

    index_csv = output_root / "features_index.csv"
    with index_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["source_path", "feature_path", "sr", "shape"])
        writer.writeheader()
        writer.writerows(index_rows)
    print(f"[index] {index_csv}")


if __name__ == "__main__":
    main()
