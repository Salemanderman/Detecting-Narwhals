import argparse
import csv
import numpy as np
from pathlib import Path
import sys
import torch
from torch.utils.data import DataLoader, Subset

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
    ap.add_argument("--input-root",  required=True, help="Root folder containing audio files.")
    ap.add_argument("--output-root", required=True, help="Output folder for .npz features and index.")
    ap.add_argument("--subset-len", type=int, default=0, help="Optionally limit to a subset of data.")
    args = ap.parse_args()

    input_root  = Path(args.input_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    subset_len = args.subset_len

    skip_secs  = 5
    batch_size = 32

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Input:  {input_root}")
    print(f"Output: {output_root}")
    print(f"Device: {device}")

    # --- Dataset and dataloader ---
    dataset = utils.AudioDataset(input_root, target_sr=64_000, skip_secs=skip_secs, mode="crop", max_secs=None)
    if subset_len > 0:
        dataset = Subset(dataset, list(range(min(subset_len, len(dataset))))) # takes first N samples
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=utils.max_len_collate) # Only shuffle data when training.
    print(f"Files:   {len(dataset)}")
    print(f"Batches: {len(loader)}")

    # --- Log-mel transform pipeline ---
    specgram_config = configs.get_specgram_config()
    logmel_transf   = utils.PipelineSpecgram(specgram_config=specgram_config).to(device)
    logmel_transf.eval()
    print(f"Specgram config: {specgram_config}")

    # --- Feature extraction ---
    index_rows = []

    with torch.no_grad():
        for i, batch in enumerate(loader, 1):
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
                        out_dir = output_root / wav_path.relative_to(input_root).parent
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

            if i % 10 == 0:
                print(f"[info] Processed {i * batch_size} / {len(dataset)} files.")

    print(f"[done] Extracted features for {len(index_rows)} files.")

    # --- Save index CSV ---
    index_csv = output_root / "features_index.csv"
    with index_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["source_path", "feature_path", "sr", "shape"])
        writer.writeheader()
        writer.writerows(index_rows)
    print(f"[index] {index_csv}")


if __name__ == "__main__":
    main()
