import argparse
import csv
from pathlib import Path

import numpy as np
from pesq import pesq
from scipy.io import wavfile


def read_audio(path):
    if not path.is_file():
        raise SystemExit(f"WAV 파일을 찾을 수 없습니다: {path}")

    rate, samples = wavfile.read(path)
    if rate != 16000 or samples.ndim != 1 or samples.size == 0:
        raise SystemExit(f"16 kHz 모노 WAV가 필요합니다: {path}")

    audio = samples.astype(np.float64)
    if np.issubdtype(samples.dtype, np.unsignedinteger):
        midpoint = (np.iinfo(samples.dtype).max + 1) / 2
        audio = (audio - midpoint) / midpoint
    elif np.issubdtype(samples.dtype, np.signedinteger):
        audio /= abs(float(np.iinfo(samples.dtype).min))

    return audio


def snr_db(clean, audio):
    signal_energy = np.sum(clean**2)
    error_energy = np.sum((clean - audio)**2)
    if error_energy == 0:
        return float("inf")
    return float(10 * np.log10(signal_energy / error_energy))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("validation_dir", type=Path)
    args = parser.parse_args()

    project = Path(__file__).resolve().parent
    noisy_dir = args.validation_dir / "noisy"
    clean_dir = args.validation_dir / "clean"

    if not noisy_dir.is_dir() or not clean_dir.is_dir():
        raise SystemExit("지정한 validation 경로 안에 noisy와 clean 폴더가 필요합니다.")

    noisy_paths = sorted(noisy_dir.glob("*.wav"))
    if not noisy_paths:
        raise SystemExit("validation/noisy에 WAV 파일이 없습니다.")

    rows = []
    for noisy_path in noisy_paths:
        stem = noisy_path.stem
        utterance = stem[:-6] if stem.endswith("_noisy") else stem
        names = {f"{utterance}_clean.wav", f"{utterance}.wav", noisy_path.name}
        matches = [clean_dir / name for name in names if (clean_dir / name).is_file()]

        if len(matches) != 1:
            raise SystemExit(
                f"Clean 파일을 하나로 짝지을 수 없습니다: {noisy_path.name}"
            )

        clean = read_audio(matches[0])
        noisy = read_audio(noisy_path)
        enhanced = read_audio(project / "output" / noisy_path.name)

        if clean.shape != noisy.shape or clean.shape != enhanced.shape:
            raise SystemExit(f"세 WAV의 샘플 수가 다릅니다: {noisy_path.name}")
        if not np.any(clean):
            raise SystemExit(f"Clean WAV가 무음입니다: {matches[0].name}")

        try:
            pesq_before = pesq(16000, clean, noisy, "wb")
            pesq_after = pesq(16000, clean, enhanced, "wb")
        except Exception as error:
            raise SystemExit(f"PESQ 계산 실패 ({noisy_path.name}): {error}") from error

        rows.append({
            "file": noisy_path.name,
            "pesq_before": pesq_before,
            "pesq_after": pesq_after,
            "snr_before_db": snr_db(clean, noisy),
            "snr_after_db": snr_db(clean, enhanced),
        })

    csv_path = project / "validation_scores.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"평가 파일 수: {len(rows)}")
    for metric, label in [
        ("pesq_before", "평균 PESQ-WB (처리 전)"),
        ("pesq_after", "평균 PESQ-WB (처리 후)"),
        ("snr_before_db", "평균 SNR dB (처리 전)"),
        ("snr_after_db", "평균 SNR dB (처리 후)"),
    ]:
        print(f"{label}: {np.mean([row[metric] for row in rows]):.3f}")
    print(f"파일별 점수 저장: {csv_path.name}")


if __name__ == "__main__":
    main()