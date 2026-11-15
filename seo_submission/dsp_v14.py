"""v14: frequency-band adaptive spectral reconstruction on top of v13.

Classical DSP only. Inference uses the noisy input and fixed JSON parameters.
No clean/reference signal or noise-category metadata is accessed at inference.

v14 reconstructs a bandwise mixture of the existing v12 and v13 outputs,
with optional mild noise-dominance weighting. The exact v13 output is a
built-in identity candidate (all recovery and attenuation coefficients = 0).
"""
from dataclasses import dataclass
import numpy as np
from scipy import signal, ndimage
from dsp_v12 import ParamsV12, enhance as enhance_v12
from dsp_v13 import ParamsV13, enhance as enhance_v13, enhance_from_baseline


@dataclass(frozen=True)
class ParamsV14:
    v13_params: dict
    # Fraction of v12 spectrum recovered from v13 in three frequency bands.
    low_recovery: float = 0.0
    mid_recovery: float = 0.0
    high_recovery: float = 0.0
    # Additional mild suppression in high-noise bins. Zero means none.
    low_suppression: float = 0.0
    mid_suppression: float = 0.0
    high_suppression: float = 0.0
    frame_size: int = 512
    low_cut_hz: float = 500.0
    high_cut_hz: float = 2800.0
    transition_hz: float = 150.0
    min_gain: float = 0.80


def _smooth_bands(frequencies, values, lo, hi, transition):
    """Smoothly interpolate three band coefficients; avoid boundary artifacts."""
    frequencies = np.asarray(frequencies, dtype=np.float64)
    a = np.clip((frequencies - (lo-transition)) / (2*transition), 0, 1)
    b = np.clip((frequencies - (hi-transition)) / (2*transition), 0, 1)
    a = a*a*(3-2*a)
    b = b*b*(3-2*b)
    return values[0]*(1-a) + values[1]*a*(1-b) + values[2]*b


def _channel(noisy, v12, v13, sr, p):
    n = len(noisy)
    if n < 32:
        return v13.copy()
    frame = min(p.frame_size if sr <= 16000 else 2*p.frame_size, n)
    hop = max(1, frame//4)
    kw = dict(fs=sr, window='hann', nperseg=frame, noverlap=frame-hop,
              nfft=frame, boundary='zeros', padded=True)
    frequencies, _, spec_noisy = signal.stft(noisy, **kw)
    _, _, spec_12 = signal.stft(v12, **kw)
    _, _, spec_13 = signal.stft(v13, **kw)
    recovery = _smooth_bands(frequencies,
        (p.low_recovery, p.mid_recovery, p.high_recovery),
        p.low_cut_hz, p.high_cut_hz, p.transition_hz)[:, None]
    suppression = _smooth_bands(frequencies,
        (p.low_suppression, p.mid_suppression, p.high_suppression),
        p.low_cut_hz, p.high_cut_hz, p.transition_hz)[:, None]
    mixed = spec_13 + recovery * (spec_12-spec_13)
    if np.any(suppression > 0):
        # Estimate noise dominance from the observed noisy waveform only.
        power = np.abs(spec_noisy)**2
        local = ndimage.minimum_filter1d(
            ndimage.uniform_filter1d(power, size=3, axis=1, mode='nearest'),
            size=21, axis=1, mode='nearest')
        dominance = np.clip((1.5*local)/(power + 1e-12), 0, 1)
        mask = np.clip(1 - suppression * dominance, p.min_gain, 1)
        mixed = mixed * mask
    _, result = signal.istft(mixed, fs=sr, window='hann', nperseg=frame,
                             noverlap=frame-hop, nfft=frame,
                             boundary=True, input_onesided=True)
    if len(result) < n:
        result = np.pad(result, (0,n-len(result)))
    return result[:n]


def enhance_from_baselines(noisy, v12, v13, sr, p: ParamsV14):
    for key in ('low_recovery','mid_recovery','high_recovery',
                'low_suppression','mid_suppression','high_suppression'):
        val = getattr(p, key)
        if not (0 <= val <= 1):
            raise ValueError(f'{key} must be in [0,1]')
    if not (p.frame_size >= 64 and p.low_cut_hz > p.transition_hz > 0
            and p.high_cut_hz > p.low_cut_hz + 2*p.transition_hz
            and 0 < p.min_gain <= 1):
        raise ValueError('Invalid v14 STFT or frequency settings')
    noisy, v12, v13 = [np.asarray(v,dtype=np.float64) for v in (noisy,v12,v13)]
    if noisy.shape != v12.shape or noisy.shape != v13.shape:
        raise ValueError('Noisy, v12 and v13 shapes differ')
    if (p.low_recovery == p.mid_recovery == p.high_recovery == 0 and
        p.low_suppression == p.mid_suppression == p.high_suppression == 0):
        return v13.copy()
    if noisy.ndim == 1:
        y = _channel(noisy,v12,v13,sr,p)
    elif noisy.ndim == 2:
        y = np.column_stack([_channel(noisy[:,i],v12[:,i],v13[:,i],sr,p)
                             for i in range(noisy.shape[1])])
    else:
        raise ValueError('Input must be [samples] or [samples,channels]')
    return np.clip(y, -1, 1)


def enhance(noisy, sr, p: ParamsV14):
    base13_params = ParamsV13(**p.v13_params)
    if (p.low_recovery == p.mid_recovery == p.high_recovery == 0 and
        p.low_suppression == p.mid_suppression == p.high_suppression == 0):
        # No unnecessary STFT round trip: reproduce v13 exactly.
        return enhance_v13(noisy, sr, base13_params)
    base12 = enhance_v12(noisy, sr, ParamsV12(**base13_params.v12_params))
    base13 = enhance_from_baseline(noisy, base12, sr, base13_params)
    return enhance_from_baselines(noisy, base12, base13, sr, p)
