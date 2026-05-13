import argparse
import csv
import numpy as np
from pprint import pformat
from pathlib import Path
import sys
import torch
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio-root",  required=True, help="Root folder containing audio files.")
    ap.add_argument("--output-root", required=True, help="Output folder for .npz features and index.")
    ap.add_argument("--subset-len", type=int, default=0, help="Optionally limit to a subset of data.")
    ap.add_argument("--towsey", action="store_true", default=False, help="Apply Towsey (2013) modal noise removal after spectrogram computation.")
    ap.add_argument("--towsey-N", type=float, default=0.0, dest="towsey_N", help="Towsey N: std devs above modal background added to threshold (default 0.0).")
    ap.add_argument("--audio-crop-start-secs", type=int, default=5, dest="audio_crop_start_secs", help="Seconds to cut from the start of each recording (default: 5).")
    ap.add_argument("--linear-freq", action="store_true", default=False, dest="linear_freq", help="Use linear STFT frequency bins instead of mel-scale (sets n_mels=None).")
    ap.add_argument("--n-mels", type=int, default=None, dest="n_mels", help="Number of mel bins (default: from config). Ignored if --linear-freq is set.")
    args = ap.parse_args()

    audio_root  = Path(args.audio_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    subset_len = args.subset_len

    start_secs  = args.audio_crop_start_secs
    print(f" [info] cutting off first {start_secs} seconds of each recording.")
    end_secs = 265 # Cut off endings so each file is same length
    print(f" [info] cutting off last {270 - end_secs} seconds of each recording.")
    batch_size = 32

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Input:  {audio_root}")
    print(f"Output: {output_root}")
    print(f"Device: {device}")

    # Setup dataset and dataloader
    dataset = utils.AudioDataset(audio_root, target_sr=64_000, start_secs=start_secs, end_secs=end_secs)
    if subset_len > 0:
        dataset = Subset(dataset, list(range(min(subset_len, len(dataset))))) # takes first N samples
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=utils.max_len_collate) # Only shuffle data when training.
    print(f"Files: {len(dataset)}")
    print(f"Batches: {len(loader)}")

    # Perform log-mel  spectrogram transformation
    specgram_config = configs.get_specgram_config()
    specgram_config["use_towsey"] = args.towsey
    specgram_config["towsey_N"] = args.towsey_N
    if args.linear_freq:
        specgram_config["n_mels"] = None
    elif args.n_mels is not None:
        specgram_config["n_mels"] = args.n_mels
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

            for b in range(waveforms.size(0)):
                wav_path = Path(paths[b])
                wf = waveforms[b].to(device=device, dtype=torch.float32)
                sr_val = _to_int(srs[b])

                try:
                    feat = logmel_transf(wf)

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
