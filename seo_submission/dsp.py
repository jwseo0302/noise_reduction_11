"""Traditional DSP speech enhancement. No clean/reference audio is used here."""
from dataclasses import dataclass, asdict
import numpy as np
from scipy import signal, ndimage


@dataclass(frozen=True)
class Params:
    noise_percentile: float = 15.0
    strength: float = 1.0
    gain_floor: float = 0.10
    temporal_smoothing: float = 0.70


def enhance_channel(x, sr, p: Params):
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 0 or np.max(np.abs(x)) < 1e-10:
        return x.copy()
    frame = min(512 if sr <= 16000 else 1024, len(x))
    if frame < 16:
        return x.copy()
    overlap = frame * 3 // 4
    _, _, z = signal.stft(x, fs=sr, window='hann', nperseg=frame,
                          noverlap=overlap, boundary='zeros', padded=True)
    power = np.abs(z)**2
    # Per-frequency estimate from the quietest frames; noisy signal only.
    noise = np.percentile(power, p.noise_percentile, axis=1)
    noise = ndimage.median_filter(noise, size=5, mode='nearest')
    noise = np.maximum(noise, 1e-12)
    posterior = power / (p.strength * noise[:, None] + 1e-12)
    gains = np.empty_like(power)
    prev_gain = np.ones(power.shape[0])
    prev_post = np.ones(power.shape[0])
    # Decision-directed a-priori SNR estimate, with temporal gain smoothing.
    for t in range(power.shape[1]):
        prior = 0.96 * (prev_gain**2) * prev_post + 0.04 * np.maximum(posterior[:, t]-1.0, 0.0)
        gain = prior / (prior + 1.0)
        gain = np.maximum(gain, p.gain_floor)
        if t:
            gain = p.temporal_smoothing * prev_gain + (1.0-p.temporal_smoothing) * gain
        gains[:, t] = gain
        prev_gain = gain
        prev_post = posterior[:, t]
    _, y = signal.istft(z * gains, fs=sr, window='hann', nperseg=frame,
                         noverlap=overlap, input_onesided=True, boundary=True)
    # Preserve sample count exactly. Avoid artificial peak normalisation.
    y = y[:len(x)]
    if len(y) < len(x):
        y = np.pad(y, (0, len(x)-len(y)))
    return np.clip(y, -1.0, 1.0)


def enhance(x, sr, p: Params):
    x = np.asarray(x)
    if x.ndim == 1:
        return enhance_channel(x, sr, p)
    if x.ndim == 2:
        return np.stack([enhance_channel(x[:, c], sr, p) for c in range(x.shape[1])], axis=1)
    raise ValueError('WAV must be mono or multichannel with shape [samples, channels]')
