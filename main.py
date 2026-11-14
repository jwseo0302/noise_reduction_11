from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import istft, stft


def reduce_noise(samples, sample_rate):
    original_dtype = samples.dtype
    audio = samples.astype(np.float64)

    # 8비트 unsigned WAV의 무음 값은 128이다.
    offset = 128.0 if original_dtype == np.dtype("uint8") else 0.0
    audio -= offset

    frame_length = min(len(audio), max(2, round(0.032 * sample_rate)))
    if frame_length < 2:
        return samples.copy()

    overlap = frame_length // 2
    _, _, spectrum = stft(
        audio,
        fs=sample_rate,
        window="hann",
        nperseg=frame_length,
        noverlap=overlap,
        axis=0,
        boundary="zeros",
        padded=True,
    )
    power = np.abs(spectrum) ** 2

    # 에너지가 가장 낮은 20%의 프레임에서 잡음 전력을 추정한다.
    candidates = power[..., 1:-1] if power.shape[-1] > 2 else power
    frame_energy = candidates.mean(axis=tuple(range(candidates.ndim - 1)))
    quiet_count = max(1, int(np.ceil(0.20 * len(frame_energy))))
    quiet_frames = np.argsort(frame_energy)[:quiet_count]
    noise_power = np.median(candidates[..., quiet_frames], axis=-1, keepdims=True)

    # 추정한 잡음 전력을 빼고, 진폭 이득은 최소 0.25로 유지한다.
    strength = 8.0
    min_gain = 0.25
    safe_power = np.maximum(power, np.finfo(np.float64).eps)
    gain_squared = 1.0 - strength * noise_power / safe_power
    gain = np.sqrt(np.clip(gain_squared, min_gain**2, 1.0))

    _, restored = istft(
        spectrum * gain,
        fs=sample_rate,
        window="hann",
        nperseg=frame_length,
        noverlap=overlap,
        freq_axis=0,
        time_axis=-1,
        boundary=True,
    )

    # 복원 신호를 입력과 같은 샘플 수와 채널 배열로 맞춘다.
    restored = np.moveaxis(restored, -1, 0)[:len(samples)] + offset
    if np.issubdtype(original_dtype, np.integer):
        limits = np.iinfo(original_dtype)
        restored = np.clip(np.rint(restored), limits.min, limits.max)

    return restored.astype(original_dtype)


def main():
    project_dir = Path(__file__).resolve().parent
    input_dir = project_dir / "input"
    output_dir = project_dir / "output"

    if not input_dir.is_dir():
        raise SystemExit("main.py 옆에 input 폴더를 만들어 주세요.")

    wav_paths = sorted(
        path for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".wav"
    )
    if not wav_paths:
        raise SystemExit("input 폴더 바로 아래에 WAV 파일을 넣어 주세요.")

    output_dir.mkdir(exist_ok=True)

    for input_path in wav_paths:
        sample_rate, samples = wavfile.read(input_path)
        cleaned_samples = reduce_noise(samples, sample_rate)

        output_path = output_dir / input_path.name
        wavfile.write(output_path, sample_rate, cleaned_samples)

        saved_rate, saved_samples = wavfile.read(output_path)
        if saved_rate != sample_rate or saved_samples.shape != samples.shape:
            raise RuntimeError(f"WAV 규격이 달라졌습니다. {input_path.name}")

        channels = 1 if samples.ndim == 1 else samples.shape[1]
        print(
            f"{input_path.name} 잡음 제거 완료 | "
            f"{sample_rate} Hz | {channels}채널 | {len(samples)}샘플"
        )


if __name__ == "__main__":
    main()