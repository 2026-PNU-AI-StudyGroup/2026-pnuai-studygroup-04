# 뇌 MRI 병변 Segmentation — 데이터 파이프라인 & 모델 (CBAS)

PNU AI학습공동체 "노예(진)" 팀 프로젝트의 **데이터 수집/전처리 · 라벨링 준비 · h5 변환 · Segmentation 모델 개발** 파트입니다.
전체 프로젝트 소개는 저장소 루트의 중간보고서를 참고하세요.

## 구성

```
requirements.txt
.gitattributes           # weights/*.pt Git LFS 추적 설정
src/
├── dicom2png.ipynb       # 라벨링 도구용 DICOM → PNG 변환 (데이터 수집/라벨링 준비)
├── dicom2h5.ipynb        # DICOM + 라벨 PNG → 학습용 h5 변환
├── train.py              # (권장) 환자 단위 5-fold 학습 스크립트 — 아래 "검증 방법론" 참고
├── evaluate_test.py       # fold별 held-out test set 슬라이스 단위 Dice/IoU 평가
├── evaluate_lesion_matching.py  # fold별 held-out test 환자 병변 단위 TP/FP/FN 평가
├── Main.ipynb            # 초기 탐색용 노트북 — 슬라이스 단위 분할이라 데이터 누수 있음(참고용, 학습엔 train.py 사용)
├── Evaluation.ipynb      # 학습 결과 평가
├── logs/
│   ├── CBAGS_grouped_5fold_train.log   # 5-fold 전체 학습 실행 로그 (실행 증거)
│   └── README.md
├── networks/
│   ├── CBAS.py            # CBAM 기반 Attention U-Net Segmentation 모델
│   └── modules.py          # 공통 블록 (DoubleConv, SEBlock)
├── utils/
│   ├── dataset.py           # Dataset, 정상 슬라이스 대표 샘플링(NormalSliceSelector)
│   ├── utils.py              # 학습/추론 보조 함수, 지표
│   └── loss.py                # Focal / Dice / BCE / GUL 손실 함수
└── options/
    └── hyper_parameters.py     # 하이퍼파라미터 및 경로(HP)
inference/
├── predict_fusion.py       # CBAS(segmentation) + YOLOv11x(detection) 결합 추론
├── demo_app.py              # 시연 영상 녹화용 로컬 데모 웹페이지 (Gradio)
├── Yolo_train_bagging.py   # YOLOv12n bagging(bootstrap 앙상블, 8-bag) 학습 스크립트
└── Yolo_val_bagging.py     # bag별 예측을 WBF(Weighted Boxes Fusion)로 합쳐 평가하는 스크립트
volume_measurement/
├── functions.py                          # 병변 박스 추출, voxel 부피 계산 등 공통 함수
├── volume_tracking.py                    # 슬라이스 간 병변 추적 + 부피 계산 파이프라인
├── lesion_matching.py                    # 3D 병변 단위 IoU 매칭 (TP/FP/FN)
├── measure_and_compare.py                # GT vs CBAS-only vs CBAS+YOLO 부피/개수/병변매칭 비교 스크립트
├── RESULTS.md                            # 5-fold 재학습 최종 결과 (Dice/IoU, 병변 단위 P/R/F1)
├── test_eval_dice_iou_5fold.json         # RESULTS.md 원본 수치 (슬라이스 단위)
├── lesion_matching_results_5fold_heldout.csv    # RESULTS.md 원본 수치 (케이스별, 익명 CaseID)
└── lesion_matching_summary_5fold_heldout.json   # RESULTS.md 원본 수치 (전체 집계)
weights/
└── README.md      # 가중치 비공개 안내 (DUA/IRB 확인 전까지 .pt 파일 미포함)
tests/
├── test_model_and_losses.py   # 모델 forward shape·파라미터 수·손실함수 NaN 방지 스모크 테스트
└── test_lesion_matching.py    # 3D 병변 IoU 매칭 정확성 테스트
LICENSE
```

## 데이터

**실제 학습·검증에는 양산부산대학교병원 원내 DICOM·라벨 데이터를 사용했습니다.** 환자 개인정보가 포함된 보안 데이터이므로 이 저장소에는 데이터 자체를 포함하지 않으며(`.gitignore`로 `data/` 전체 제외), 코드와 파이프라인만 공개합니다.

**코드를 직접 재현해보고 싶다면** 아래 공개 데이터셋을 다운로드해 동일 구조로 배치하면 됩니다. (용량이 커서 저장소에는 데이터를 올리지 않고 링크만 남깁니다.)

- [ISLES 2022](https://doi.org/10.5281/zenodo.7153326) (Zenodo, 계정/DUA 불필요) — DWI/ADC/FLAIR + 뇌졸중 병변 마스크. 본 프로젝트와 동일하게 DWI 기반이라 구조가 가장 잘 맞습니다.
- 다운로드 후 아래 구조로 배치하세요.

```
data/results_train_cleaned/<그룹>/<환자>/{pre|post 또는 preop|postop}/*.dcm
data/results_train_cleaned/<그룹>/<환자>/labelmask/{pre|post}/*.png
```

- 각 시퀀스는 DWI b-value 0/1000이 섞여 있으며, `dicom2h5.ipynb`가 DICOM 헤더의 b-value 태그로 b=1000만 자동 선별합니다.

## 실행 방법

```bash
conda create -n cbas python=3.10 -y
conda activate cbas
pip install -r requirements.txt
```

노트북/스크립트는 모두 **`src/` 디렉토리를 기준**으로 상대경로(`../data`, `../res`)를 사용합니다.

```bash
cd src
jupyter notebook   # dicom2h5.ipynb 실행용
```

1. `dicom2h5.ipynb` — h5 데이터셋 생성 (`data/h5/` 에 저장)
2. `train.py` — 환자 단위 5-fold 학습 (`res/<모델명>/model/fold_*/` 에 체크포인트 저장)
   ```bash
   python train.py --name CBAGS_grouped --gpu 0
   ```
   Jupyter 커널 대신 독립 프로세스인 `.py` 스크립트로 만든 이유: Jupyter 커널은 셀을 여러 번
   재실행하면 이전 CUDA 텐서/컨텍스트가 완전히 해제되지 않고 누적되는 경우가 있어, 장시간(수십 시간)
   학습에는 매 실행이 깨끗한 프로세스로 끝나는 `.py` 스크립트가 GPU 메모리 관리 면에서 더 안전합니다.
3. `Evaluation.ipynb` — 학습 곡선·지표 확인
4. `evaluate_test.py` — fold별 held-out test set 슬라이스 단위 Dice/IoU 평가 (`python evaluate_test.py --name CBAGS_grouped --gpu 0`)
5. `evaluate_lesion_matching.py` — fold별 held-out test 환자 병변 단위 TP/FP/FN 평가 (`python evaluate_lesion_matching.py --name CBAGS_grouped --gpu 0`)

## 검증 방법론 (중요)

이전 버전(`Main.ipynb` + `utils.k_fold_split`)은 **슬라이스 인덱스를 기준으로 셔플**해서 5-fold를
나눴습니다. 그 결과 같은 환자의 인접 슬라이스, 심지어 같은 환자의 pre/post 슬라이스가 train과
test에 동시에 들어가는 **데이터 누수**가 있었고(재검증 결과 전 fold에서 환자 100% 중복), 이 상태로
낸 성능 지표는 일반화 성능의 근거로 쓸 수 없습니다.

`train.py`는 이를 두 가지로 수정했습니다.

1. **환자 단위 분리**: `get_patient_groups()`로 각 슬라이스가 어느 환자 소속인지 구하고, 환자
   리스트 자체를 K-Fold로 나눈 뒤 그 환자 집합으로만 train/val/test 슬라이스를 필터링합니다.
   같은 환자가 두 split에 동시에 존재하면 `assert`로 즉시 실패하도록 방어 코드를 넣었습니다.
2. **평가셋 유병률 보존**: 기존에는 병변:정상 슬라이스를 1:1로 인위 조정(클러스터링 대표 샘플링)한
   데이터를 train/val/test 전체에 공통으로 썼습니다. `train.py`는 이 대표 샘플링을 **train 환자에게만**
   적용하고, val/test는 해당 환자들의 슬라이스를 자연 비율 그대로 사용합니다.

이 방식으로 재학습한 결과와, `evaluate_test.py`/`evaluate_lesion_matching.py`로 각 fold의
**진짜 held-out 환자**에 대해 측정한 Dice/IoU·병변 단위 Precision/Recall/F1은
`volume_measurement/RESULTS.md`에 정리했습니다.

## 모델 개요 (CBAS)

- 인코더-디코더(U-Net 계열) + Squeeze-Excitation 블록 + CBAM(Channel/Spatial Attention) bottleneck
- 출력 3종(main / coarse / seg4) 멀티스케일 supervision
- 손실: Focal + Dice + BCE + GUL(General Union Loss) 가중합
- 정상 슬라이스는 ResNet18 특징 기반 KMeans로 대표 샘플링하며, 병변 슬라이스와 **동일 b-value**에서만 후보를 선택하도록 필터링되어 있습니다 (`utils/dataset.py`의 `NormalSliceSelector.get_lesion_bvalue`).

## Detection-Segmentation Fusion 추론 (CBAS + YOLOv11x)

CBAS(segmentation)와 YOLOv11x(detection)는 **서로 독립적으로 학습된 별개의 가중치**입니다. "합쳐진 하나의
모델/가중치"는 존재하지 않으며, 같은 슬라이스를 두 모델에 각각 넣어 나온 결과(픽셀 마스크 vs bounding box)를
추론 코드에서 결합(spatial gating: YOLO box 밖의 CBAS 마스크 픽셀을 제거)합니다. 자세한 배경은
`docs/paper`의 Method 파트(Detection-Segmentation Fusion 섹션)를 참고하세요.

**가중치**: `cbas_best.pt`, `yolo_best.pt`는 양산부산대학교병원 데이터로 학습된 파생물이라 DUA/IRB 조건
확인 전까지 이 저장소에는 **포함하지 않습니다** (자세한 이유와 내부 전달 방법은 `weights/README.md` 참고).
`train.py`로 직접 재학습하면 동일한 경로에 체크포인트가 생성되며, 아래 명령은 그 경로를 가리키는
예시입니다.

```bash
python inference/predict_fusion.py \
  --dicom_dir /path/to/patient/post \
  --out_dir ./output \
  --cbas_weights weights/cbas_best.pt \
  --yolo_weights weights/yolo_best.pt
```

폴더 안 b=1000 DWI 슬라이스만 자동 선별하여, 최종 게이팅된 병변 마스크를 `out_dir`에 PNG로 저장합니다.

### 데모 웹페이지 (시연 영상 녹화용)

`inference/demo_app.py`는 이미지(DICOM 또는 PNG/JPG) 한 장을 업로드/드래그하면 CBAS 단독 예측(빨강)과
CBAS+YOLO 게이팅 후 예측(초록), 그리고 병변 검출 여부 판정을 보여주는 로컬 Gradio 앱입니다.

```bash
pip install gradio
python inference/demo_app.py --gpu 0
# 콘솔에 뜨는 http://127.0.0.1:7860 접속
```

**개인정보 보호**: 이 앱은 원내 환자 데이터를 전혀 스캔·나열하지 않습니다 — 화면에는 사용자가 직접
업로드한 이미지만 나옵니다. 화면 녹화(→ 유튜브 등 외부 제출)에 실제 환자 영상이 찍히는 것을 피하기
위한 설계이며, 시연용 이미지는 원내 데이터 대신 공개 데이터셋(위 "데이터" 섹션의 ISLES 2022)에서
준비하는 것을 권장합니다.

### YOLO Bagging 학습/평가 (참고용)

`inference/Yolo_train_bagging.py`, `inference/Yolo_val_bagging.py`는 YOLOv12n을 bootstrap 방식으로 8개
bag(서로 다른 서브셋)에 학습시키고, 검증 시 5개 bag의 예측을 Weighted Boxes Fusion(WBF)으로 합쳐 평가하는
앙상블 실험 스크립트입니다. 데이터 경로가 팀 내부 서버(`/ds/want2704/...`)로 하드코딩되어 있어 그대로
실행되지는 않으며, 학습 방법론을 기록해두기 위한 참고 코드입니다. 실제 배포에 쓰인 `weights/yolo_best.pt`는
이 bagging 방식이 아닌 단일 YOLOv11x 모델입니다 (`inference/predict_fusion.py`가 사용하는 모델).

## 성능 측정 (Volume Measurement)

`volume_measurement/measure_and_compare.py`는 각 환자 폴더에 대해 GT(사람 라벨) / CBAS 단독 예측 /
CBAS+YOLO 게이팅 예측 세 가지로 각각 병변 개수·부피(mm³)를 계산해서 비교표를 만듭니다.

**병변 단위 IoU 매칭(위치까지 검증)**: 개수·부피 "총량"만 비교하면 GT와 예측의 병변 개수가
우연히 같아도 실제로는 서로 다른 위치를 가리키는 경우를 놓칠 수 있습니다(자문 검토에서 지적된
낙관적 편향 중 하나). 이를 보완하기 위해 `volume_measurement/lesion_matching.py`가 GT/예측
마스크를 각각 3D(슬라이스 방향까지 포함) connected-component로 병변 인스턴스화한 뒤, 병변 쌍의
3D voxel IoU에 헝가리안 알고리즘을 적용해 1:1로 매칭합니다. IoU가 임계값(기본 0.1 — 병변이
작고 불규칙해서 0.5 같은 기준은 비현실적) 이상인 매칭만 TP로 인정하고, 매칭 안 된 GT는 FN,
매칭 안 된 예측은 FP로 셉니다. `measure_and_compare.py` 실행 시 CBASOnly/Gated 각각에 대해
`*_Lesion_TP/FP/FN/Precision/Recall/F1` 컬럼과 전체 micro-averaged 요약이 함께 출력·저장됩니다.
매칭 로직 자체의 정확성은 `tests/test_lesion_matching.py`(합성 3D 볼륨 기반, 실제 데이터 불필요)로 검증했습니다.

```bash
python volume_measurement/measure_and_compare.py \
  --data_root ../data/results_train_cleaned \
  --cbas_weights ../weights/cbas_best.pt \
  --yolo_weights ../weights/yolo_best.pt \
  --out_prefix result \
  --lesion_iou_threshold 0.1
```

환자 실명·등록번호 등 개인정보는 저장하지 않고, 익명 `CaseID`(`case_001`, `case_002`, ...)만 사용합니다.

**`evaluate_lesion_matching.py`**는 위와 별개로, 5-fold 전체를 cross-validation 방식으로
pooled 평가합니다(각 환자를 그 환자가 held-out이었던 fold의 가중치로만 평가 — 즉 전체
99명 전원이 학습에 한 번도 안 쓰인 모델로 진짜 held-out 평가됨). 이 결과가 아래 "재학습 최종
결과"의 병변 단위 수치입니다.

```bash
python evaluate_lesion_matching.py --name CBAGS_grouped --gpu 0
```

## 재학습 최종 결과

`train.py`(환자 단위 GroupKFold + 자연 유병률 평가)로 5-fold 재학습을 완료했습니다. 슬라이스
단위 Dice/IoU, 병변 단위 Precision/Recall/F1(CBAS 단독 vs CBAS+YOLO 게이팅) 전체 수치와 해석은
**[`volume_measurement/RESULTS.md`](volume_measurement/RESULTS.md)** 참고. 이전 버전
(환자 단위 미분리 + 평가셋 유병률 인위조정) 결과였던 `performance_comparison_anonymized.xlsx/.csv`는
삭제하고 이 결과로 완전히 교체했습니다.

요약: 게이팅 적용 시 병변 단위 FP가 644→62건(90% 감소)로 줄면서 Precision 0.39→0.86,
F1 0.52→0.80으로 개선되어, "YOLO 게이팅이 과다예측을 줄인다"는 가설이 데이터 누수 없는
진짜 held-out 평가로 확인되었습니다.

## 테스트

```bash
pip install pytest
pytest tests/ -v
```

- `test_model_and_losses.py`: 모델 forward pass의 출력 shape, 파라미터 수(29,260,629, 자문 검토 시
  확인된 값과 일치), Focal/Dice/BCE/GUL 손실 함수가 빈 마스크·전체 마스크 등 극단적인 입력에서도
  NaN/Inf 없이 안정적으로 계산되는지 확인하는 스모크 테스트.
- `test_lesion_matching.py`: 3D 병변 IoU 매칭(`lesion_matching.py`)이 완전 일치/완전 불일치/위치만
  어긋난 개수 일치("개수는 맞지만 위치는 틀림" 케이스 포함)/임계값 미만 겹침/다중 병변 헝가리안
  매칭/빈 GT·예측 등 경계 조건에서 TP·FP·FN·Precision·Recall을 정확히 계산하는지 검증(합성 3D
  볼륨만 사용, 실제 데이터 불필요).

## 실행 증거

`src/dicom2png.ipynb`, `src/dicom2h5.ipynb`는 실제로 실행되어 결과가 셀 출력에 남아 있습니다.
다만 원내 데이터를 대상으로 하는 셀은 출력에 실제 환자 폴더명·파일 경로가 그대로 찍히기 때문에
**개인정보 유출 방지를 위해 그 셀들만 출력을 비워뒀고**, 대신 각 노트북 맨 아래 "실행 검증용
데모(합성 익명 데이터)" 셀에서 무작위 합성 DICOM으로 동일한 함수(`get_h5` 등)를 실행한 결과를
확인할 수 있습니다. `train.py`의 실행 증거는 `src/logs/CBAGS_grouped_5fold_train.log`(5-fold
전체, 2026-08-19~08-20 실제 학습 타임스탬프 포함, 개인정보 없음)로 남겼습니다 —
자세한 내용은 `src/logs/README.md` 참고.
