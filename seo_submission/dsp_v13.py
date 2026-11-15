"""v13: classical multi-resolution STFT residual attenuation, applied to v12.

The enhancement path reads noisy audio and saved DSP parameters only.
Clean audio and PESQ are used solely by the external Train/Validation scripts.
This is an experimental multi-resolution post-filter, not a published estimator.
"""
from dataclasses import dataclass
import numpy as np
from scipy import signal, ndimage
from dsp_v12 import ParamsV12, enhance as v12_enhance


@dataclass(frozen=True)
class ParamsV13:
    v12_params: dict
    mix: float = 0.0                # zero reproduces v12 exactly
    frame_size: int = 1024         # alternate STFT length (256/1024 @ 16k)
    correction_strength: float = 0.35
    gain_floor: float = 0.85
    low_energy_fraction: float = 0.20
    freq_sigma: float = 1.0


def _postfilter_channel(noisy, baseline, sr, params):
    n = len(noisy)
    if n < 32 or np.max(np.abs(noisy), initial=0) < 1e-12:
        return baseline.copy()
    frame = min(params.frame_size if sr <= 16000 else 2*params.frame_size, n)
    hop = max(1, frame // 4)
    kw = dict(fs=sr, window='hann', nperseg=frame, noverlap=frame-hop,
              nfft=frame, boundary='zeros', padded=True)
    _, centers, noisy_spec = signal.stft(noisy, **kw)
    _, _, base_spec = signal.stft(baseline, **kw)
    psd = np.abs(noisy_spec)**2
    # Select complete, low-energy frames on the noisy input only.
    loc = np.rint(centers * sr).astype(int)
    valid = np.flatnonzero((loc >= frame // 2) & (loc + frame // 2 <= n))
    if len(valid) == 0:
        valid = np.arange(psd.shape[1])
    frame_energy = np.mean(psd[:, valid], axis=0)
    take = max(1, int(np.ceil(params.low_energy_fraction * len(valid))))
    idx = valid[np.argsort(frame_energy, kind='stable')[:take]]
    static_psd = ndimage.uniform_filter1d(
        psd[:, idx].mean(axis=1), size=3, mode='nearest')[:, None]
    # Local smoothing handles nonstationarity; static floor reduces speech leakage.
    local_psd = ndimage.uniform_filter1d(psd, size=3, axis=1, mode='nearest')
    local_psd = ndimage.minimum_filter1d(local_psd, size=21, axis=1, mode='nearest')
    noise_psd = np.maximum(0.55*static_psd + 0.45*1.5*local_psd, 1e-14)
    posterior = psd / (noise_psd + 1e-12)
    mask = np.clip(posterior / (posterior + params.correction_strength),
                   params.gain_floor, 1.0)
    if params.freq_sigma > 0:
        mask = ndimage.gaussian_filter1d(mask, params.freq_sigma,
                                        axis=0, mode='nearest')
    _, processed = signal.istft(base_spec*mask, fs=sr, window='hann',
                                nperseg=frame, noverlap=frame-hop,
                                nfft=frame, boundary=True, input_onesided=True)
    if len(processed) < n:
        processed = np.pad(processed, (0,n-len(processed)))
    return processed[:n]


def enhance_from_baseline(noisy, baseline, sr, params: ParamsV13):
    """Used by Train to cache expensive v12 processing."""
    if not (0 <= params.mix <= 1 and 0 < params.gain_floor <= 1
            and params.correction_strength >= 0 and params.frame_size >= 64
            and 0 < params.low_energy_fraction <= 1 and params.freq_sigma >= 0):
        raise ValueError('Invalid v13 parameter')
    noisy = np.asarray(noisy, dtype=np.float64)
    baseline = np.asarray(baseline, dtype=np.float64)
    if noisy.shape != baseline.shape:
        raise ValueError('Input and baseline must have identical shape')
    if params.mix == 0:
        return baseline.copy()
    if noisy.ndim == 1:
        corrected = _postfilter_channel(noisy, baseline, sr, params)
    elif noisy.ndim == 2:
        corrected = np.column_stack([
            _postfilter_channel(noisy[:,i], baseline[:,i], sr, params)
            for i in range(noisy.shape[1])])
    else:
        raise ValueError('Expected mono or [samples,channels]')
    return np.clip((1-params.mix)*baseline + params.mix*corrected, -1, 1)


def enhance(noisy, sr, params: ParamsV13):
    baseline = v12_enhance(noisy, sr, ParamsV12(**params.v12_params))
    return enhance_from_baseline(noisy, baseline, sr, params)
