"""v11: noisy-only time-frequency residual gain correction on frozen v7.

This is an experimental classical DSP estimator, not a published IMCRA implementation.
No Clean WAV, metadata or PESQ enters enhance().
"""
from dataclasses import dataclass
import numpy as np
from scipy import signal, ndimage
from dsp_v7 import ParamsV7, enhance as v7_enhance

@dataclass(frozen=True)
class ParamsV11:
    v7_params: dict
    mix: float = 0.0  # 0 means exactly frozen v7
    noise_percentile: float = 20.0
    window_frames: int = 31
    correction_strength: float = 0.5
    gain_floor: float = 0.7
    freq_smoothing: float = 1.0


def _residual_channel(noisy, base, sr, p):
    n = len(noisy)
    if n < 32 or np.max(np.abs(noisy), initial=0.0) < 1e-10:
        return np.asarray(base, dtype=np.float64).copy()
    frame = min(512 if sr <= 16000 else 1024, n)
    hop = max(1, frame // 4)
    kwargs = dict(fs=sr, window='hann', nperseg=frame,
                  noverlap=frame-hop, boundary='zeros', padded=True)
    _, _, z = signal.stft(noisy, **kwargs)
    power = np.abs(z) ** 2
    # Local minimum-statistics-inspired noise estimate with static anchoring.
    # min_filter is intentionally symmetric / offline, permitted by challenge.
    kernel = max(3, int(p.window_frames) | 1)
    smoothed = ndimage.uniform_filter1d(power, size=3, axis=1, mode='nearest')
    local_min = ndimage.minimum_filter1d(smoothed, size=kernel, axis=1, mode='nearest')
    static = np.percentile(power, p.noise_percentile, axis=1)[:, None]
    noise = np.clip(1.5 * local_min, 0.25 * static, 3.0 * static)
    noise = 0.65 * noise + 0.35 * static
    # Further suppress only where the estimate says noise dominates. Bound the
    # residual filter to protect speech: gain_floor >= 0.6 in supplied search.
    gamma = power / (noise + 1e-12)
    mask = np.clip(gamma / (gamma + p.correction_strength), p.gain_floor, 1.0)
    if p.freq_smoothing > 0:
        mask = ndimage.gaussian_filter1d(mask, p.freq_smoothing, axis=0, mode='nearest')
    # Apply gain to v7 processed signal, not noisy, making the effect measurable
    # independently of a second full-waveform denoiser.
    _, _, zb = signal.stft(base, **kwargs)
    if zb.shape != z.shape:
        raise RuntimeError('Inconsistent STFT shapes')
    _, y = signal.istft(zb * mask, fs=sr, window='hann', nperseg=frame,
                         noverlap=frame-hop, input_onesided=True, boundary=True)
    if len(y) < n:
        y = np.pad(y, (0, n-len(y)))
    return y[:n]


def enhance_from_baseline(x, base, sr, p: ParamsV11):
    """Apply residual correction to an already computed v7 output (Train speedup)."""
    if not 0 <= p.mix <= 1 or not 0 < p.gain_floor <= 1:
        raise ValueError('mix/gain_floor out of range')
    x = np.asarray(x, dtype=np.float64)
    base = np.asarray(base, dtype=np.float64)
    if x.shape != base.shape:
        raise ValueError('Noisy and baseline shapes differ')
    if p.mix == 0.0:
        return base.copy()
    if x.ndim == 1:
        residual = _residual_channel(x, base, sr, p)
    elif x.ndim == 2:
        residual = np.column_stack([_residual_channel(x[:,i], base[:,i], sr, p)
                                    for i in range(x.shape[1])])
    else:
        raise ValueError('Expected mono or [samples, channels]')
    return np.clip((1-p.mix)*base + p.mix*residual, -1, 1)


def enhance(x, sr, p: ParamsV11):
    x = np.asarray(x, dtype=np.float64)
    base = v7_enhance(x, sr, ParamsV7(**p.v7_params))
    return enhance_from_baseline(x, base, sr, p)
