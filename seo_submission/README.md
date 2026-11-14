# Speech Denoising Challenge — Traditional DSP (v14)

## Algorithm
No machine learning or deep learning. A fixed-parameter, noisy-only signal processing pipeline using STFT-domain spectral denoising, a Wiener/Log-MMSE hybrid, low-energy-frame noise estimation, multi-resolution residual filtering, and version-specific post-filtering. See `dsp*.py` and `AI_USAGE.md` for implementation details. The DSP heuristics are not claimed to be exact published implementations.

## Setup
Python 3.11 recommended. From the repository root:

```powershell
python -m pip install -r requirements.txt
```

## Hidden-test denoising (no Clean reference needed)
Put noisy WAV files in `input/`, then execute:

```powershell
python denoise_v14.py --input input --output output --config best_params_v14.json
```

All input WAV basenames are preserved, as are sample rate, channels and number of samples. Output is 32-bit float WAV (PCM subtype may differ).

## Validation metrics
See `validation_summary_v14.json` and `validation_results_v14.csv` **if bundled**. These are measured using paired Clean references strictly for assessment, not for inference. Dataset: 164 Validation pairs, PESQ-WB at 16 kHz.

## Reproducibility and limitations
Parameters in `best_params_v14.json` were selected using Train pairs. Noisy-only inference is deterministic. This repository is the frozen submission version; Train/Validation WAV data and development scripts are intentionally excluded. Validation was inspected repeatedly during earlier development, which can bias model selection; hidden-test performance is unknown.

## Acknowledgments
The teaching assistant's `noise_reduction_baselines.ipynb` informed low-energy-frame PSD estimation. See `AI_USAGE.md` for AI assistance and the project's own code.
