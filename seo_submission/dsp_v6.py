"""v6 hybrid DSP inference. Only noisy audio reaches enhancement functions.

The fixed two branches are the original v3 decision-directed Wiener method and
v5 log-MMSE method. Train/Validation clean audio is never used during inference.
"""
from dataclasses import dataclass
import numpy as np
from dsp import Params as WienerParams, enhance as wiener_enhance
from dsp_v5 import ParamsV5, enhance as v5_enhance


@dataclass(frozen=True)
class ParamsV6:
    alpha: float
    wiener_params: dict
    logmmse_params: dict


def enhance(x, sr, params: ParamsV6):
    """Return same-shape waveform; alpha=0 is exact v3, alpha=1 exact v5 Log-MMSE."""
    alpha = float(params.alpha)
    if not np.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError('alpha must be between 0 and 1')
    w = WienerParams(**params.wiener_params)
    l = ParamsV5(**params.logmmse_params)
    if l.method != 'logmmse':
        raise ValueError('v6 requires a genuine logmmse branch, not wiener_v3')
    # Avoid unnecessary computation and preserve exact branch baseline outputs.
    if alpha == 0.0:
        return wiener_enhance(x, sr, w)
    if alpha == 1.0:
        return v5_enhance(x, sr, l)
    yw = wiener_enhance(x, sr, w)
    yl = v5_enhance(x, sr, l)
    if yw.shape != yl.shape:
        raise ValueError('Branch output shapes differ')
    return np.clip((1.0-alpha)*yw + alpha*yl, -1.0, 1.0)


def build_params(alpha, v3_cfg, v5_cfg):
    allowed_w = ('noise_percentile', 'strength', 'gain_floor', 'temporal_smoothing')
    w = {key: float(v3_cfg[key]) for key in allowed_w}
    l = dict(v5_cfg)
    if l.get('method') != 'logmmse':
        raise ValueError('best_params_v5.json is not Log-MMSE. Use a Log-MMSE config from v5 Train results.')
    return ParamsV6(alpha=float(alpha), wiener_params=w, logmmse_params=l)
