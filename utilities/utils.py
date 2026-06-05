import numpy as np
from pathlib import Path
import torch 
import torchaudio as ta
import torchaudio.transforms as T
import torch.nn.functional as F
from torch.utils.data import Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _towsey_single_chunk(
    spec: np.ndarray,
    N: float,
    smooth_window: int,
    neighbourhood: bool,
    neighbourhood_threshold: float,
) -> np.ndarray:
    """Apply Towsey noise removal to a single chunk (n_bins, n_frames)."""
    n_bins, n_frames = spec.shape
    n_hist_bins = max(10, n_frames // 8)
    kernel = np.ones(smooth_window) / smooth_window
    noise_profile = np.zeros(n_bins, dtype=np.float64)

    for i in range(n_bins):
        row = spec[i].astype(np.float64)
        lo, hi = row.min(), row.max()
        if lo >= hi:
            noise_profile[i] = lo
            continue

        counts, edges = np.histogram(row, bins=n_hist_bins, range=(lo, hi))
        counts_s = np.convolve(counts.astype(float), kernel, mode='same')

        # Modal bin, capped at 95th percentile bin to guard against all-signal rows
        modal_idx = int(np.argmax(counts_s))
        cap = int(0.95 * n_hist_bins)
        if modal_idx > cap:
            modal_idx = cap
        modal_val = 0.5 * (edges[modal_idx] + edges[modal_idx + 1])

        # Std estimate: walk left from mode until 68% of sub-modal counts are covered
        std_est = 0.0
        if N > 0 and modal_idx > 0:
            sub = counts_s[:modal_idx]
            total = sub.sum()
            if total > 0:
                cumsum = np.cumsum(sub[::-1])
                idx = min(int(np.searchsorted(cumsum, 0.68 * total)), modal_idx - 1)
                std_est = (idx + 1) * (hi - lo) / n_hist_bins

        noise_profile[i] = modal_val + N * std_est

    # Step B: smooth noise profile across frequency bins, then subtract.
    pad = len(kernel) // 2
    noise_profile = np.convolve(np.pad(noise_profile, pad, mode='edge'), kernel, mode='valid')
    result = np.maximum(spec - noise_profile[:, np.newaxis], 0.0).astype(np.float32)

    # Step C: zero pixels whose 9-bin × 3-frame local average is below threshold
    if neighbourhood and n_bins >= 9 and n_frames >= 3:
        from scipy.ndimage import uniform_filter
        avg = uniform_filter(result, size=(9, 3), mode='reflect')
        result = np.where(avg < neighbourhood_threshold, 0.0, result)

    return result


def towsey_noise_removal(
    spec: np.ndarray,
    N: float = 0.0,
    smooth_window: int = 5,
    neighbourhood: bool = True,
    neighbourhood_threshold: float = 2.0,
    chunk_frames: int = 7500,
) -> np.ndarray:
    """
    Towsey (2013) adaptive modal noise subtraction, applied in one-minute chunks.

    Towsey (2013) recommends processing one-minute segments independently so the
    modal noise estimate adapts to slowly-varying background conditions. At 64 kHz
    sample rate with hop_length=512, one minute is 7500 frames (the default).

    Steps from the paper:
      A. Per-bin: histogram (length = n_frames/8) → modal value → subtract (modal + N*std)
      B. Smooth noise profile across frequency bins before subtraction
      C. Neighbourhood suppression: zero pixels whose 9×3 local average < threshold

    Args:
        spec: (n_bins, n_frames) spectrogram in dB.
        N: std devs above mode added to threshold (default 0.0; >0.1 removes signal).
        smooth_window: moving-average width for histogram and profile smoothing.
        neighbourhood: apply Step C local suppression.
        neighbourhood_threshold: pixels with local average below this are zeroed (~2 dB).
        chunk_frames: frames per processing chunk (default 7500 ≈ 1 min at 64 kHz/512 hop).
    Returns:
        Noise-removed spectrogram, same shape as input, values >= 0.
    """
    n_bins, n_frames = spec.shape

    if n_frames <= chunk_frames:
        return _towsey_single_chunk(spec, N, smooth_window, neighbourhood, neighbourhood_threshold)

    chunks = []
    start = 0
    while start < n_frames:
        end = min(start + chunk_frames, n_frames)
        chunks.append(_towsey_single_chunk(
            spec[:, start:end], N, smooth_window, neighbourhood, neighbourhood_threshold))
        start = end

    return np.concatenate(chunks, axis=1)


class AudioDataset(Dataset):
    """Audio dataset for hydroacoustic recordings from Inglefield Bredning Fjord, Greenland.
    Returns a dict with keys "waveform" (C, T), "sample_rate" (int), "path" (str).
    Crops start_secs from the start and snaps the end to 1-second boundaries, always
    removing at least 1 second. Files within 1 second of the same length produce identical
    sample counts, so batches stack without padding using the default DataLoader collate.
    """
    def __init__(self, root_dir, target_sr=64000, start_secs=5):
        self.root_dir = Path(root_dir) # Root data folder.
        self.files = list(self.root_dir.rglob("*.wav"))  # Searches for pattern in subfolders.
        self.target_sr = target_sr # Can change from original raw 64 kHz to common 16 kHz.
        self.start_secs = start_secs

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        path = self.files[idx]
        wf, sr = ta.load(str(path))
        # For the entire dataset, each waveform shape is [C, T] where C = 1 (mono), 
        # T (the number of frames) is approximately = 172 million,
        # and is recorded at sample rate = 64 kHz.
        
        s_idx  = int(self.start_secs * sr)
        # Round to nearest second then subtract 1: files within ±0.5 s of each other
        # produce the same length, and at least 1 second is always removed from the end.
        dur    = max(0, round((wf.shape[-1] - s_idx) / sr) - 1) * sr
        e_idx  = s_idx + dur

        wf = wf[:, s_idx:e_idx] if e_idx > s_idx else torch.zeros((wf.shape[0], 1))

        if sr != self.target_sr:
            wf = ta.functional.resample(wf, sr, self.target_sr)
            sr = self.target_sr

        item = {"waveform": wf, "sample_rate": sr, "path": str(path)}

        return item


# A preprocessing pipeline class for audio features. Inherits methods "eval" and "train"
# from torch.nn.Module.
class PipelineSpecgram(torch.nn.Module):
    def __init__(self, specgram_config:dict):
        super().__init__()
        # Basic spectrogram settings.
        self.sample_rate = specgram_config["sample_rate"]
        self.n_fft = specgram_config["n_fft"]
        self.win_length = specgram_config["win_length"]
        self.hop_length = specgram_config["hop_length"]
        self.window_fn = specgram_config["window_fn"]
        # Optional spectrogram settings.
        self.resample_rate = specgram_config.get("resample_rate", None)
        self.mel_bins = specgram_config.get("n_mels", None)
        self.power = specgram_config.get("power", 2.0)
        self.f_min = specgram_config.get("f_min", 0.0)
        self.f_max = specgram_config.get("f_max", self.sample_rate // 2)
        self.to_db = T.AmplitudeToDB(stype="power") # Decibel conversion.

        if self.resample_rate is not None and self.resample_rate != self.sample_rate:
            self.resample = T.Resample(orig_freq=self.sample_rate, new_freq=self.resample_rate)
            self.effective_sr = self.resample_rate
        else:
            self.resample = torch.nn.Identity() # Placeholder identity.
            self.effective_sr = self.sample_rate

        # Setup spectrogram. 
        self.spec = T.Spectrogram(
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            window_fn=self.window_fn,
            power=self.power,
            center=True,
            pad_mode="reflect"
        )

        # Mel scale transformation.
        if self.mel_bins is not None:
            # Use f_max from config, or default to Nyquist frequency
            f_max_val = self.f_max if self.f_max is not None else self.effective_sr / 2.0

            # Mel scale produces power mel bands.
            self.mel_scale = T.MelScale(
                n_mels = self.mel_bins,
                sample_rate = self.effective_sr,
                n_stft = self.n_fft // 2 + 1, # = n_freqs
                f_min = self.f_min,
                f_max = f_max_val, # = 32 kHz according to the Nyquist theorem.
                mel_scale="htk", # Default is "htk". 
                # norm="slaney"
            )
        else:
            self.mel_scale = None 
    
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        x = self.resample(waveform)
        spec = self.spec(x)  # (..., F, T) power spectrogram

        if self.mel_scale is not None:
            mel = self.mel_scale(spec)  # (..., M, T)
            return self.to_db(mel / (1e-6**2))
        else:
            return self.to_db(spec / (1e-6**2))
