## 1. 실행 환경 및 설치 방법

Python 3.11 사용을 권장합니다.

GitHub 저장소를 내려받은 후, **저장소 내부의 `seo_submission` 폴더로 이동**합니다.

```powershell
cd seo_submission
python -m pip install -r requirements.txt
```

필요한 Python 패키지와 버전은 `requirements.txt`에 명시되어 있습니다.

## 2. 잡음 제거 실행 방법

다음 명령어를 실행해주세요.

```powershell
mkdir input
mkdir output
```

잡음이 포함된 WAV 파일을 `seo_submission/input/` 폴더에 넣습니다. 

이후 `seo_submission` 폴더에서 다음 명령어를 실행합니다.

```powershell
python denoise_v14.py --input input --output output --config best_params_v14.json
```

프로그램은 `input` 폴더의 WAV 파일을 처리하여 `output` 폴더에 결과를 저장합니다.

출력 파일은 다음 조건을 유지합니다.

- 입력과 동일한 파일명
- 동일한 샘플링 주파수
- 동일한 채널 수
- 동일한 샘플 수

출력 WAV는 32-bit Float 형식으로 저장되며, 입력 파일의 PCM 세부 형식과는 다를 수 있습니다.

실제 잡음 제거 과정에서는 Clean 정답 음성이 필요하지 않습니다.

## 4. Validation 평가 결과

Validation 데이터 164쌍을 이용해 성능을 평가하였습니다. PESQ는 16 kHz 음성에 대한 PESQ-WB 기준입니다.

| 평가 지표 | 원본 Noisy | 잡음 제거 후(v14) |
|---|---:|---:|
| 평균 PESQ-WB | 1.39467 | **1.71281** |
| 평균 SNR (dB) | 9.63415 | **9.97058** |

- **PESQ-WB 개선량:** +0.31814
- **SNR 개선량:** +0.33643 dB

상세 평가 결과는 다음 파일에 저장되어 있습니다.

- `validation_summary_v14.json`: 전체 Validation 평가 요약
- `validation_results_v14.csv`: 파일별 Validation 평가 결과

Clean 음성은 성능 평가를 위한 정답으로만 사용했으며, 실제 잡음 제거 과정에서는 사용하지 않았습니다.
