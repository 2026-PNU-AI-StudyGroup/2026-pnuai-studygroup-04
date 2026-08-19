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
├── Main.ipynb            # 초기 탐색용 노트북 — 슬라이스 단위 분할이라 데이터 누수 있음(참고용, 학습엔 train.py 사용)
├── Evaluation.ipynb      # 학습 결과 평가
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
├── Yolo_train_bagging.py   # YOLOv12n bagging(bootstrap 앙상블, 8-bag) 학습 스크립트
└── Yolo_val_bagging.py     # bag별 예측을 WBF(Weighted Boxes Fusion)로 합쳐 평가하는 스크립트
volume_measurement/
├── functions.py                          # 병변 박스 추출, voxel 부피 계산 등 공통 함수
├── volume_tracking.py                    # 슬라이스 간 병변 추적 + 부피 계산 파이프라인
├── measure_and_compare.py                # GT vs CBAS-only vs CBAS+YOLO 부피/개수 비교 실행 스크립트
└── performance_comparison_anonymized.xlsx/.csv   # 실측 비교 결과 (개인정보 제외, 익명 CaseID)
weights/
└── README.md      # 가중치 비공개 안내 (DUA/IRB 확인 전까지 .pt 파일 미포함)
tests/
└── test_model_and_losses.py   # 모델 forward shape·파라미터 수·손실함수 NaN 방지 스모크 테스트
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

### YOLO Bagging 학습/평가 (참고용)

`inference/Yolo_train_bagging.py`, `inference/Yolo_val_bagging.py`는 YOLOv12n을 bootstrap 방식으로 8개
bag(서로 다른 서브셋)에 학습시키고, 검증 시 5개 bag의 예측을 Weighted Boxes Fusion(WBF)으로 합쳐 평가하는
앙상블 실험 스크립트입니다. 데이터 경로가 팀 내부 서버(`/ds/want2704/...`)로 하드코딩되어 있어 그대로
실행되지는 않으며, 학습 방법론을 기록해두기 위한 참고 코드입니다. 실제 배포에 쓰인 `weights/yolo_best.pt`는
이 bagging 방식이 아닌 단일 YOLOv11x 모델입니다 (`inference/predict_fusion.py`가 사용하는 모델).

## 성능 측정 (Volume Measurement)

`volume_measurement/measure_and_compare.py`는 각 환자 폴더에 대해 GT(사람 라벨) / CBAS 단독 예측 /
CBAS+YOLO 게이팅 예측 세 가지로 각각 병변 개수·부피(mm³)를 계산해서 비교표를 만듭니다.

```bash
python volume_measurement/measure_and_compare.py \
  --data_root ../data/results_train_cleaned \
  --cbas_weights ../weights/cbas_best.pt \
  --yolo_weights ../weights/yolo_best.pt \
  --out_prefix result
```

환자 실명·등록번호 등 개인정보는 저장하지 않고, 익명 `CaseID`(`case_001`, `case_002`, ...)만 사용합니다.

> ⚠️ **`performance_comparison_anonymized.xlsx`의 수치는 위 "검증 방법론"에서 설명한 구버전
> (환자 단위 미분리 + 평가셋 유병률 인위조정) 파이프라인으로 산출된 것으로, **신뢰할 수 있는 성능
> 근거가 아닙니다.** `train.py`(환자 단위 GroupKFold + 자연 유병률 평가)로 재학습을 진행 중이며,
> 완료되는 대로 이 표와 파일을 교체할 예정입니다. 게이팅으로 개수/부피 과다예측이 줄어드는
> **방향성** 자체는 재현되었지만, 정확한 수치(%, MAE 등)는 재학습 결과로 다시 봐야 합니다.

## 테스트

```bash
pip install pytest
pytest tests/ -v
```

모델 forward pass의 출력 shape, 파라미터 수(29,260,629, 자문 검토 시 확인된 값과 일치), 그리고
Focal/Dice/BCE/GUL 손실 함수가 빈 마스크·전체 마스크 등 극단적인 입력에서도 NaN/Inf 없이 안정적으로
계산되는지 확인하는 스모크 테스트입니다.
