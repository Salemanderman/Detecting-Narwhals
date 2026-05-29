import numpy as np
from pathlib import Path
import torch 
import torchaudio as ta
import torchaudio.transforms as T
import torch.nn.functional as F
from torch.utils.data import Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def towsey_noise_removal(
    spec: np.ndarray,
    N: float = 0.0,
    smooth_window: int = 5,
    neighbourhood: bool = True,
    neighbourhood_threshold: float = 2.0,
) -> np.ndarray:
    """
    Towsey (2013) adaptive modal noise subtraction.

    For each frequency bin, builds a histogram of intensity values across all time
    frames, finds the modal (most frequent) value as the background estimate, and
    subtracts it. N standard deviations above the mode can optionally be added to
    the threshold (paper recommends N=0.0 for spectrograms; N>0.1 removes signal).

    Steps from the paper:
      A. Per-bin: histogram → modal value → subtract (modal + N*std) → truncate to 0
      B. Smooth noise profile across frequency bins before subtraction
      C. Neighbourhood suppression: zero pixels whose 9×3 local average < threshold

    Args:
        spec: (n_bins, n_frames) spectrogram, any scale (dB or linear).
        N: std devs above mode added to threshold (default 0.0).
        smooth_window: moving-average width for histogram and profile smoothing.
        neighbourhood: apply Step C local suppression.
        neighbourhood_threshold: pixels with local average below this are zeroed.
            Use ~2.0 for dB spectrograms, ~0.015 for linear-scale spectrograms.
    Returns:
        Noise-removed spectrogram, same shape as input, values >= 0.
    """
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
    # Edge-pad with the boundary value before convolving so the 2 outermost bins
    # get a full-window average instead of a zero-padded (underestimated) one.
    pad = len(kernel) // 2
    noise_profile = np.convolve(np.pad(noise_profile, pad, mode='edge'), kernel, mode='valid')
    result = np.maximum(spec - noise_profile[:, np.newaxis], 0.0).astype(np.float32)

    # Step C: zero pixels whose 9-bin × 3-frame local average is below threshold
    if neighbourhood and n_bins >= 9 and n_frames >= 3:
        from scipy.ndimage import uniform_filter
        avg = uniform_filter(result, size=(9, 3), mode='reflect')
        result = np.where(avg < neighbourhood_threshold, 0.0, result)

    return result


class AudioDataset(Dataset):
    """Audio dataset for hydroacoustic recordings from Inglefield Bredning Fjord, Greenland.
    Returns a dict with keys "waveform" (C, T), "sample_rate" (int), "path" (str).
    All files are cropped to the same length via start_secs + crop_end_secs snapping,
    so batches can be stacked without padding using the default DataLoader collate.
    """
    def __init__(self, root_dir, target_sr=64000, start_secs=5, crop_end_secs=5):
        self.root_dir = Path(root_dir) # Root data folder.
        self.files = list(self.root_dir.rglob("*.wav")) # Searches for pattern in subfolders.
        self.target_sr = target_sr # Can change from original raw 64 kHz to common 16 kHz.
        self.start_secs = start_secs
        self.crop_end_secs = crop_end_secs

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        path = self.files[idx]
        wf, sr = ta.load(str(path))
        # For the entire dataset, each waveform shape is [C, T] where C = 1 (mono), 
        # T (the number of frames) is approximately = 172 million,
        # and is recorded at sample rate = 64 kHz.
        
        s_idx  = int(self.start_secs * sr)
        snap   = int(self.crop_end_secs * sr)
        dur    = ((wf.shape[-1] - s_idx) // snap) * snap
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

# Helper function to reduce the number of sample points in audio data tensors.
def reduce_tensor(w, max_pts):
    """
    Reduces a 1D tensor w to exactly max_pts elements. 
    If w has more than max_pts elements, it is downsampled using fixed indices.
    If w has fewer than max_pts elements, it is padded with zeros at the end.
    Args:
        w: 1D torch.Tensor.
        max_pts: Maximum number of elements in the output tensor.
    Returns:
        w_small: Reduced or padded 1D tensor of length max_pts.
    """
    # Downsample to max_pts using fixed indices.
    n_elms = w.numel()
    if n_elms == max_pts:
        return w 
    
    if n_elms == 0:
        return torch.zeros(max_pts, device=w.device, dtype=w.dtype)
    
    if n_elms >= max_pts:
        k = torch.arange(max_pts, device=w.device)
        idx = torch.floor(k.to(torch.float32) * (n_elms / float(max_pts))).to(torch.long)
        # idx = torch.linspace(0, n_elms - 1, steps=max_pts, device=w.device).to(torch.long)
        w_small = w.index_select(0, idx)
    else:
        # Padding needed at dataset level to build X (data loader pads at batch-level).
        w_small = F.pad(w, (0, max_pts - n_elms))
    return w_small

# Builds feature matrix Z as inputs to clustering models.
def tensors_to_array(dataloader, transform, max_pts=None, dtype=np.float32, device=device):
    """
    Builds a (N, max_pts) feature matrix Z of type NumPy array. Z used as input for clustering 
    algorithms. The size of Z can be reduced by downsampling based on a maximum number of 
    points per audio recording.
    Args:
        dataloader: PyTorch DataLoader batches of audio waveforms.
        max_pts: Maximum number of points per feature.  
    Returns:
        X: NumPy array of shape (n samples, n features).
        ids: List of audio file identifiers corresponding to each row in X.
    """

    rows, ids = [], []
    transform.eval()

    with torch.no_grad(): # Disables gradient computations.
        for batch in dataloader:
            waves = batch["waveforms"]
            paths = batch["paths"]
            lengths = batch["lengths"]

            if device is None:
                device_ = waves.device
            else:
                device_ = device

            B, C, Tn = waves.shape

            for b in range(B):
                L = int(lengths[b].item()) if lengths is not None else Tn
                L = max(0, min(L, Tn))
                w = waves[b, :, :L][0]
                
                T0 = w.numel()
                if max_pts is not None:
                    if T0 == 0:
                        w = torch.zeros(max_pts, dtype=w.dtype, device=w.device)
                    elif T0 > max_pts:
                        k = torch.arange(max_pts, device=w.device)
                        idx = torch.floor(k.to(torch.float32) * (T0 / float(max_pts))).to(torch.long)
                        w = w.index_select(0, idx)
                    elif T0 < max_pts:
                        w = F.pad(w, (0, max_pts - T0)) 

                # Let x be the input waveform. .view reshapes to [1, 1, T] for transformation.
                # The shape matches [B, C, T].
                x = w.view(1, 1, -1).to(device=device, dtype=torch.float32)
                
                feat = transform(x).squeeze(0)  # Transforms into [C, F, T] by removing the B dimension.
                # where Channels (C = 1, mono), F = n_mels = 128, and T = n time frames.
                # T contains the observations. The slice feat[:, :, t] is a feature vector at time frame t
                # across all mel bins. 

                # Mean across time frames and std across time.
                mu = feat.mean(dim=-1, keepdim=False) 
                sig = feat.std(dim=-1, keepdim=False) 
                vec = torch.cat([mu, sig], dim=0)
                # Final feature vector shape: (2 * n_mels)
                vec = vec.reshape(-1)
                rows.append(vec.detach().cpu().numpy().astype(dtype))

                p = paths[b]
                ids.append(Path(p).name if isinstance(p, (str, Path)) else str(p))
                
    
    ids = np.array(ids, dtype=str)            

    # Concatenate rows into Z.
    # Z: (n samples, n features.)
    Z = np.stack(rows, axis=0)
    return Z, ids

# Computes descriptive statistics: peak amplitudes, mean amplitudes, 
# root mean squares and zero-crossing rates
def compute_stats(w, sr, length, skip_secs):
            
            n_chans, n_frames = w.shape 

            empty_dict = dict(duration_sec = 0, peak=0.0, mean_abs=0.0, rms=0.0, zcr_hz=0.0)
            if n_frames <= 0:
                   return empty_dict

            s_idx = int(sr * float(skip_secs))
            s_idx = max(0, min(s_idx, n_frames)) # Ensure no 0 length files. 

            wf = w[:, :n_frames].clone()
            if s_idx > 0:
                wf[:, :s_idx] = 0.0 # Mute first corrupted seconds.
            
            # Duration in seconds.
            duration_sec = max((n_frames - s_idx) / float(sr), 0.0)

            # Wave without the muted clip.
            if s_idx < n_frames: 
                  wclip = wf[:, s_idx:] 
            else: 
                  return empty_dict

            if wclip.numel() == 0:
                  return empty_dict

            peak = wclip.abs().amax().amax().item()
            mean_abs = wclip.abs().mean().item()
            # Root mean square.
            rms = wclip.pow(2).mean().sqrt().item()
            # Zero-Crossing Rate.
            silence_band = 10e-11
            wclip_nz = torch.where(wclip == 0, silence_band, wclip)
            signs = torch.signbit(wclip_nz)

            if wclip.shape[1] >= 2:
                  changes = signs[:, 1:] ^ signs[:, :-1]
                  zc = changes.sum().item()
                  zcr_hz = (zc / (wclip_nz.shape[1] - 1)) * sr
            else:
                  zcr_hz = 0.0

            collected_stats = dict(
                  duration_sec = float(duration_sec),
                  peak_abs = float(peak),
                  mean_abs = float(mean_abs),
                  rms = float(rms),
                  zcr_hz = float(zcr_hz)
            )

            return collected_stats    