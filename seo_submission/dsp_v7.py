"""v7 adaptive Wiener/Log-MMSE mixture, entirely deterministic classical DSP.

Only the noisy signal and frozen Train-selected parameters are used at inference.
"""
from dataclasses import dataclass
import numpy as np
from scipy import signal
from dsp_v6 import ParamsV6, enhance as enhance_v6

FEATURES = ('flatness', 'spectral_flux', 'lowband_ratio')


@dataclass(frozen=True)
class ParamsV7:
    base_alpha: float
    feature: str
    slope: float
    center: float
    scale: float
    wiener_params: dict
    logmmse_params: dict


def noisy_features(x, sr):
    """Signal-only descriptors; mono summary, no clean audio / CSV fields.

    Measures on STFT: spectral flatness, positive spectral flux and low-band
    energy fraction. Features are descriptive, not trained classifiers.
    """
    a = np.asarray(x, dtype=np.float64)
    if a.ndim == 2:
        a = a.mean(axis=1)
    if a.ndim != 1:
        raise ValueError('Expected mono or (samples, channels)')
    if len(a) < 32 or np.max(np.abs(a)) < 1e-10:
        return {name: 0.0 for name in FEATURES}
    nseg = min(512 if sr <= 16000 else 1024, len(a))
    noverlap = nseg * 3 // 4
    f, _, z = signal.stft(a, fs=sr, nperseg=nseg, noverlap=noverlap,
                          window='hann', boundary=None, padded=False)
    power = np.maximum(np.abs(z)**2, 1e-16)
    energy = power.sum(axis=0)
    # Skip the least energetic frames when possible, reducing silence bias.
    active = energy >= np.percentile(energy, 35)
    if not np.any(active):
        active = np.ones(len(energy), dtype=bool)
    chosen = power[:, active]
    flatness = np.exp(np.mean(np.log(chosen), axis=0)) / np.mean(chosen, axis=0)
    flatness_value = float(np.median(flatness))

    # Normalized positive spectral increments in adjacent magnitude frames.
    mag = np.sqrt(power)
    mag /= np.maximum(np.linalg.norm(mag, axis=0, keepdims=True), 1e-12)
    if mag.shape[1] > 1:
        flux = np.maximum(np.diff(mag, axis=1), 0.0)
        spectral_flux = float(np.median(np.linalg.norm(flux, axis=0)))
    else:
        spectral_flux = 0.0

    low_mask = f <= min(700.0, 0.45 * sr)
    lowband_ratio = float(np.sum(chosen[low_mask]) / np.maximum(np.sum(chosen), 1e-12))
    return {'flatness': flatness_value,
            'spectral_flux': spectral_flux,
            'lowband_ratio': lowband_ratio}


def mixing_alpha(x, sr, p: ParamsV7):
    if p.feature not in FEATURES and p.feature != 'constant':
        raise ValueError(f'Unknown feature {p.feature}')
    if not (0.0 <= p.base_alpha <= 1.0):
        raise ValueError('base_alpha must be in [0,1]')
    if p.feature == 'constant' or p.slope == 0:
        return float(p.base_alpha)
    features = noisy_features(x, sr)
    normalized = np.clip((features[p.feature] - p.center) / max(p.scale, 1e-9), -1, 1)
    return float(np.clip(p.base_alpha + p.slope * normalized, 0.0, 1.0))


def enhance(x, sr, p: ParamsV7):
    alpha = mixing_alpha(x, sr, p)
    return enhance_v6(x, sr, ParamsV6(alpha=alpha, wiener_params=p.wiener_params,
                                      logmmse_params=p.logmmse_params))
