# 뇌 MRI 병변 Segmentation — 데이터 파이프라인 & 모델 (CBAS)

PNU AI학습공동체 "노예(진)" 팀 프로젝트의 **데이터 수집/전처리 · 라벨링 준비 · h5 변환 · Segmentation 모델 개발** 파트입니다.
전체 프로젝트 소개는 저장소 루트의 중간보고서를 참고하세요.

## 구성

```
requirements.txt
src/
├── dicom2png.ipynb   # 라벨링 도구용 DICOM → PNG 변환 (데이터 수집/라벨링 준비)
├── dicom2h5.ipynb     # DICOM + 라벨 PNG → 학습용 h5 변환
├── Main.ipynb         # CBAS 모델 5-fold 학습
├── Evaluation.ipynb   # 학습 결과 평가
├── networks/
│   ├── CBAS.py         # CBAM 기반 Attention U-Net Segmentation 모델
│   └── modules.py       # 공통 블록 (DoubleConv, SEBlock)
├── utils/
│   ├── dataset.py        # Dataset, 정상 슬라이스 대표 샘플링(NormalSliceSelector)
│   ├── utils.py           # 학습/추론 보조 함수, 지표
│   └── loss.py             # Focal / Dice / BCE / GUL 손실 함수
└── options/
    └── hyper_parameters.py  # 하이퍼파라미터 및 경로(HP)
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

노트북은 모두 **`src/` 디렉토리에서 Jupyter를 실행**하는 것을 기준으로 상대경로(`../data`, `../res`)를 사용합니다.

```bash
cd src
jupyter notebook
```

1. `dicom2h5.ipynb` — h5 데이터셋 생성 (`data/h5/` 에 저장)
2. `Main.ipynb` — 5-fold 학습 (`res/<모델명>/model/fold_*/` 에 체크포인트 저장)
3. `Evaluation.ipynb` — 학습 곡선·지표 확인

## 모델 개요 (CBAS)

- 인코더-디코더(U-Net 계열) + Squeeze-Excitation 블록 + CBAM(Channel/Spatial Attention) bottleneck
- 출력 3종(main / coarse / seg4) 멀티스케일 supervision
- 손실: Focal + Dice + BCE + GUL(General Union Loss) 가중합
- 정상 슬라이스는 ResNet18 특징 기반 KMeans로 대표 샘플링하며, 병변 슬라이스와 **동일 b-value**에서만 후보를 선택하도록 필터링되어 있습니다 (`utils/dataset.py`의 `NormalSliceSelector.get_lesion_bvalue`).
