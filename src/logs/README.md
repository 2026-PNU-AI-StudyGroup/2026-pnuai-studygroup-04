# CBAGS_grouped_5fold_train.log

`train.py --name CBAGS_grouped`로 돌린 5-fold 전체 학습의 실제 실행 로그(2026-08-19 07:41 ~
2026-08-20 18:21, 각 줄 타임스탬프 포함, 환자 개인정보 없음 — 확인 완료).

**fold0/fold1(08/19)과 fold2~4(08/20) 사이 속도가 크게 다른 이유**: 학습 도중 CPU 스레드
과다 스폰 버그를 발견해서(`torch`/`cv2`가 기본으로 서버 코어 수만큼 스레드를 써서 GPU가
놀고 있었음 — 자세한 내용은 `train.py` 상단의 `torch.set_num_threads(2)` 관련 주석과 커밋
로그 참고) fold2부터 `num_threads` 제한을 건 코드로 재시작했습니다. epoch당 시간이
~206초 → ~15~20초로 줄었을 뿐, 데이터·하이퍼파라미터·모델 구조는 전혀 바뀌지 않아서
fold0/1과 fold2~4의 성능 지표는 그대로 비교 가능합니다. fold0/1이 재시작 전 기록,
fold2~4가 재시작 후 기록이라는 사실을 숨기지 않고 로그를 있는 그대로 이어붙였습니다.

fold별 최종 결과 요약은 `volume_measurement/RESULTS.md` 참고.
