import argparse
import csv
import numpy as np
from pprint import pformat
from pathlib import Path
import sys
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm


def _int_or_none(v):
    return None if str(v).lower() == "none" else int(v)

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio-root",  required=True, help="Root folder containing audio files.")
    ap.add_argument("--output-root", required=True, help="Output folder for .npz features and index.")
    ap.add_argument("--subset-len", type=int, default=0, help="Optionally limit to a subset of data.")
    ap.add_argument("--towsey", action="store_true", default=False, help="Apply Towsey (2013) modal noise removal after spectrogram computation.")
    ap.add_argument("--towsey-N", type=float, default=0.0, dest="towsey_N", help="Towsey N: std devs above modal background added to threshold (default 0.0).")
    ap.add_argument("--audio-crop-start-secs", type=int, default=5, dest="audio_crop_start_secs", help="Seconds to cut from the start of each recording (default: 5).")
    ap.add_argument("--n-mels",       type=_int_or_none, default=None, dest="n_mels",       help="Number of mel bins (default: from config)")
    ap.add_argument("--num-workers",  type=int, default=4,  dest="num_workers", help="DataLoader worker processes for parallel file loading (default: 4)")
    ap.add_argument("--batch-size",   type=int, default=16,              help="Files processed per GPU batch (default: 16, reduce if GPU runs out of memory)")
    args = ap.parse_args()

    audio_root  = Path(args.audio_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    subset_len = args.subset_len

    start_secs  = args.audio_crop_start_secs
    print(f" [info] cutting off first {start_secs} seconds of each recording.")
    end_secs = 265 # Cut off endings so each file is same length
    print(f" [info] cutting off last {270 - end_secs} seconds of each recording.")
    batch_size = args.batch_size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Input:  {audio_root}")
    print(f"Output: {output_root}")
    print(f"Device: {device}")

    # Setup dataset and dataloader
    dataset = utils.AudioDataset(audio_root, target_sr=64_000, start_secs=start_secs, end_secs=end_secs)
    if subset_len > 0:
        dataset = Subset(dataset, list(range(min(subset_len, len(dataset))))) # takes first N samples
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=utils.max_len_collate,
                         num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
    print(f"Files: {len(dataset)}")
    print(f"Batches: {len(loader)}")

    # Perform log-mel  spectrogram transformation
    specgram_config = configs.get_specgram_config()
    specgram_config["use_towsey"] = args.towsey
    specgram_config["towsey_N"] = args.towsey_N
    specgram_config["n_mels"] = args.n_mels  # None = linear STFT bins, int = mel bins
    logmel_transf   = utils.PipelineSpecgram(specgram_config=specgram_config).to(device)
    logmel_transf.eval()
    print("Specgram config:\n" + pformat(specgram_config, indent=2, sort_dicts=False))

    # Extract features and save to NPZ files and metadata
    index_rows = []

    print("\n[info] Starting feature extraction...")

    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting spectrograms", unit="batch"):
            waveforms = batch["waveforms"]
            srs = batch["sample_rates"]
            paths = batch["paths"]

            if waveforms.ndim == 2:  # (B, T) -> (B, 1, T)
                waveforms = waveforms.unsqueeze(1)

            # Run STFT + mel on the whole batch at once on GPU
            wf_batch = waveforms.to(device=device, dtype=torch.float32)
            specs = logmel_transf.spec(wf_batch)                          # (B, 1, F, T)
            if logmel_transf.mel_scale is not None:
                specs = logmel_transf.mel_scale(specs)                    # (B, 1, M, T)
            specs = logmel_transf.to_db(specs / (1e-6 ** 2))             # (B, 1, bins, T)

            # Move full batch to CPU in one transfer and free GPU memory immediately
            specs = specs.cpu()
            del wf_batch

            for b in range(specs.size(0)):
                wav_path = Path(paths[b])
                sr_val   = _to_int(srs[b])
                feat     = specs[b]                                       # (1, bins, T)

                # Towsey is numpy-based so runs per-file on CPU
                if logmel_transf.use_towsey:
                    feat = logmel_transf._apply_towsey(feat)

                try:
                    # Mirror input folder structure under output_root.
                    try:
                        out_dir = output_root / wav_path.parent.relative_to(audio_root)
                    except ValueError:
                        out_dir = output_root

                    out_dir.mkdir(parents=True, exist_ok=True)

                    out_path = out_dir / (wav_path.stem + ".npz")
                    np.savez_compressed(
                        str(out_path),
                        feature=feat.detach().cpu().numpy(),
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

    # Save index CSV
    index_csv = output_root / "features_index.csv"
    with index_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["source_path", "feature_path", "sr", "shape"])
        writer.writeheader()
        writer.writerows(index_rows)
    print(f"[index] {index_csv}")


if __name__ == "__main__":
    main()
