"""v12: TA baseline low-energy-frame PSD inside v11 residual STFT correction.
No Clean WAV, metadata or evaluation score enters enhancement.
The baseline PSD estimator follows the method in noise_reduction_baselines.ipynb,
cell 'Common STFT and noise estimation' (512-sample Hann, 128-hop, valid frames,
lowest-energy 20% averaged PSD, 3-bin smoothing). This is an adaptation to v11.
"""
from dataclasses import dataclass
import numpy as np
from scipy import signal, ndimage
from dsp_v11 import ParamsV11, enhance as v11_enhance

@dataclass(frozen=True)
class ParamsV12:
    v11_params: dict
    mix: float = 0.0          # 0 exactly reproduces v11
    low_energy_fraction: float = 0.20
    static_weight: float = 1.0  # 0: local v11 estimate, 1: TA whole-frame PSD
    correction_strength: float = 0.5
    gain_floor: float = 0.75
    freq_smoothing: float = 1.0


def ta_noise_psd(power, centers, length, sr, frame, fraction):
    """TA baseline selection adapted from noise_reduction_baselines.ipynb cell 18.
    centers are sample coordinates (rounded), excluding STFT boundary padding.
    """
    if not 0 < fraction <= 1:
        raise ValueError('low_energy_fraction must be in (0,1]')
    c = np.rint(centers * sr).astype(int)
    full = np.flatnonzero((c >= frame//2) & (c + frame//2 <= length))
    if full.size == 0:
        full = np.arange(power.shape[1])
    energies = power[:, full].mean(axis=0)
    take = max(1, int(np.ceil(fraction * len(full))))
    selected = full[np.argsort(energies, kind='stable')[:take]]
    estimate = power[:, selected].mean(axis=1)
    return np.maximum(ndimage.uniform_filter1d(estimate, size=3, mode='nearest'), 1e-18)


def _one_channel(noisy, base, sr, p):
    n = len(noisy)
    if n < 32 or np.max(np.abs(noisy), initial=0) < 1e-10:
        return np.asarray(base, dtype=float).copy()
    frame = min(512 if sr <= 16000 else 1024, n)
    hop = frame // 4
    kw = dict(fs=sr, window='hann', nperseg=frame, noverlap=frame-hop,
              nfft=frame, boundary='zeros', padded=True)
    _, centers, z = signal.stft(noisy, **kw)
    power = np.abs(z)**2
    static = ta_noise_psd(power, centers, n, sr, frame, p.low_energy_fraction)[:,None]
    local_smooth = ndimage.uniform_filter1d(power, size=3, axis=1, mode='nearest')
    local_min = ndimage.minimum_filter1d(local_smooth, size=31, axis=1, mode='nearest')
    local = np.clip(1.5*local_min, 0.25*static, 3.0*static)
    local = 0.65*local + 0.35*static
    noise = p.static_weight*static + (1-p.static_weight)*local
    gamma = power / (noise + 1e-12)
    mask = np.clip(gamma/(gamma+p.correction_strength), p.gain_floor, 1.0)
    if p.freq_smoothing > 0:
        mask = ndimage.gaussian_filter1d(mask, p.freq_smoothing, axis=0, mode='nearest')
    _, _, zb = signal.stft(base, **kw)
    if zb.shape != z.shape:
        raise ValueError('STFT shape mismatch')
    _, y = signal.istft(zb*mask, fs=sr, window='hann', nperseg=frame,
                        noverlap=frame-hop, nfft=frame, input_onesided=True,
                        boundary=True)
    if len(y)<n:
        y=np.pad(y,(0,n-len(y)))
    return y[:n]


def enhance_from_baseline(x, base, sr, p: ParamsV12):
    if not (0<=p.mix<=1 and 0<=p.static_weight<=1 and 0<p.gain_floor<=1
            and p.correction_strength>=0 and p.freq_smoothing>=0):
        raise ValueError('Invalid v12 parameters')
    x=np.asarray(x,dtype=float);base=np.asarray(base,dtype=float)
    if x.shape!=base.shape:
        raise ValueError('Noisy and v11 shapes differ')
    if p.mix==0:
        return base.copy()
    if x.ndim==1:
        corrected=_one_channel(x,base,sr,p)
    elif x.ndim==2:
        corrected=np.column_stack([_one_channel(x[:,i],base[:,i],sr,p)
                                    for i in range(x.shape[1])])
    else:
        raise ValueError('Expected mono or [samples,channels]')
    return np.clip((1-p.mix)*base+p.mix*corrected,-1,1)


def enhance(x,sr,p:ParamsV12):
    base=v11_enhance(x,sr,ParamsV11(**p.v11_params))
    return enhance_from_baseline(x,base,sr,p)
