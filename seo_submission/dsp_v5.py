"""Traditional DSP speech enhancement: v3 Wiener or Ephraim-Malah log-MMSE.
Inference uses ONLY the noisy waveform. No ML, clean references, or metadata.
"""
from dataclasses import dataclass
import numpy as np
from scipy import signal, ndimage, special
from dsp import Params as WienerParams, enhance as wiener_enhance


@dataclass(frozen=True)
class ParamsV5:
    method: str = 'logmmse'  # 'wiener_v3', 'logmmse'
    noise_percentile: float = 20.0
    strength: float = 2.6
    gain_floor: float = 0.16
    temporal_smoothing: float = 0.5
    adaptive_mix: float = 0.0
    noise_window: int = 25
    prior_alpha: float = 0.96
    freq_sigma: float = 0.0


def _channel(x, sr, p):
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n == 0 or np.max(np.abs(x)) < 1e-10:
        return x.copy()
    frame = min(512 if sr <= 16000 else 1024, n)
    if frame < 16:
        return x.copy()
    overlap = frame * 3 // 4
    _, _, z = signal.stft(x, fs=sr, window='hann', nperseg=frame,
                          noverlap=overlap, boundary='zeros', padded=True)
    power = np.abs(z) ** 2
    base_noise = np.percentile(power, p.noise_percentile, axis=1)
    base_noise = ndimage.median_filter(base_noise, size=5, mode='nearest')
    base_noise = np.maximum(base_noise, 1e-12)
    if p.adaptive_mix > 0:
        window = max(3, int(p.noise_window) | 1)
        rolling = ndimage.percentile_filter(power, percentile=p.noise_percentile,
                                            size=(1, window), mode='nearest')
        rolling = ndimage.median_filter(rolling, size=(5, 1), mode='nearest')
        rolling = np.clip(rolling, 0.25 * base_noise[:, None], 4 * base_noise[:, None])
        noise = (1 - p.adaptive_mix) * base_noise[:, None] + p.adaptive_mix * rolling
    else:
        noise = base_noise[:, None]
    gamma = np.clip(power / (p.strength * noise + 1e-12), 1e-8, 1e6)
    gains = np.empty_like(power)
    prev_gain = np.ones(power.shape[0])
    prev_gamma = np.ones(power.shape[0])
    for t in range(power.shape[1]):
        xi = (p.prior_alpha * prev_gain**2 * prev_gamma +
              (1-p.prior_alpha) * np.maximum(gamma[:, t] - 1.0, 0.0))
        xi = np.clip(xi, 1e-8, 1e6)
        nu = np.maximum(xi * gamma[:, t] / (1 + xi), 1e-8)
        # E[log amplitude] gain: xi/(1+xi) * exp(0.5*E1(nu)).
        # E1 diverges near 0, so explicitly clip the input and resulting gain.
        log_gain = np.log(xi / (1 + xi)) + 0.5 * special.exp1(nu)
        gain = np.exp(np.clip(log_gain, -30.0, 0.0))
        gain = np.maximum(gain, p.gain_floor)
        if t:
            gain = p.temporal_smoothing * prev_gain + (1 - p.temporal_smoothing) * gain
        if p.freq_sigma > 0:
            gain = ndimage.gaussian_filter1d(gain, p.freq_sigma, mode='nearest')
            gain = np.maximum(gain, p.gain_floor)
        gains[:, t] = gain
        prev_gain = gain
        prev_gamma = gamma[:, t]
    _, y = signal.istft(z * gains, fs=sr, window='hann', nperseg=frame,
                         noverlap=overlap, input_onesided=True, boundary=True)
    if len(y) < n:
        y = np.pad(y, (0, n - len(y)))
    return np.clip(y[:n], -1., 1.)


def enhance(x, sr, p: ParamsV5):
    if p.method == 'wiener_v3':
        return wiener_enhance(x, sr, WienerParams(
            noise_percentile=p.noise_percentile, strength=p.strength,
            gain_floor=p.gain_floor, temporal_smoothing=p.temporal_smoothing))
    if p.method != 'logmmse':
        raise ValueError(f'Unknown DSP method: {p.method}')
    arr = np.asarray(x)
    if arr.ndim == 1:
        return _channel(arr, sr, p)
    if arr.ndim == 2:
        return np.stack([_channel(arr[:, i], sr, p) for i in range(arr.shape[1])], axis=1)
    raise ValueError('Expected mono or [samples, channels]')
