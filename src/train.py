"""
CBAS-UNet2D 5-fold 학습 (환자 단위 분리, 자연 유병률 검증/평가셋).

Main.ipynb를 .py 스크립트로 옮기면서 자문의견 두 가지를 같이 반영:
  1) k_fold_split이 슬라이스 인덱스를 셔플해서 같은 환자가 train/test에 동시에
     들어가던 데이터 누수 -> 환자(폴더) 단위로 완전히 분리
  2) 병변:정상 슬라이스를 1:1로 인위 조정한 데이터가 val/test 평가에도 그대로
     쓰이던 문제 -> val/test는 정상 슬라이스를 클러스터링으로 솎아내지 않고
     "해당 fold의 held-out 환자가 가진 슬라이스 전체(자연 비율)"를 그대로 사용.
     정상 슬라이스 대표 샘플링(클러스터링)은 train 환자에게서만 적용.

노트북이 아니라 .py로 분리한 이유: Jupyter 커널은 셀을 재실행할 때마다 이전
CUDA 텐서/컨텍스트가 완전히 해제되지 않고 누적되는 경우가 있어 장시간 학습에
불리함. 매 실행이 독립 프로세스인 .py 스크립트가 GPU 메모리 관리 면에서 더
안전하고, nohup/백그라운드 실행·로그 확인도 더 쉬움.
"""
import os
import sys
import logging
import argparse
import contextlib
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, Subset
from torchvision import transforms
from sklearn.model_selection import KFold, train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.utils import *
from utils.loss import *
from utils.dataset import Dicom2dDataset, NormalSliceSelector
from networks.CBAS import SuperEnhancedAFMSUNet2D
from options.hyper_parameters import HP


def get_patient_groups(dataset, context=0):
    """dataset(전체, only_lesion=False)의 각 샘플이 어느 환자 소속인지 반환.
    meta 경로 '.../<병변그룹>/<환자폴더>/<phase>/xxxxx.dcm' 구조를 이용.
    """
    groups = []
    for i in range(len(dataset)):
        center_meta = dataset.meta[i][context]
        parts = center_meta.replace('\\', '/').split('/')
        patient_folder = parts[-3]
        lesion_group = parts[-4]
        groups.append(f'{lesion_group}/{patient_folder}')
    return np.array(groups)


def build_fold_datasets(full_dataset, lesion_dataset, full_groups, lesion_groups,
                         train_patients, val_patients, test_patients,
                         seed, context, device):
    """환자 집합(train/val/test)이 주어졌을 때, 이 fold의 실제 학습/검증/평가
    Dataset을 만든다. val/test는 자연 비율 그대로, train만 병변:정상 1:1로
    대표 샘플링."""

    val_idx = [i for i, g in enumerate(full_groups) if g in val_patients]
    test_idx = [i for i, g in enumerate(full_groups) if g in test_patients]
    val_subset = Subset(full_dataset, val_idx)
    test_subset = Subset(full_dataset, test_idx)

    train_lesion_idx = [i for i, g in enumerate(lesion_groups) if g in train_patients]
    train_lesion_subset = Subset(lesion_dataset, train_lesion_idx)

    train_normal_candidates = [
        i for i, g in enumerate(full_groups)
        if g in train_patients and not np.any(full_dataset.label[i] > 0)
    ]

    n_clusters = max(1, len(train_lesion_idx))  # train 병변 슬라이스 수만큼 정상도 뽑아 1:1 유지
    selector = NormalSliceSelector(
        full_dataset, seed=seed, device=device, n_clusters=n_clusters,
        context=context, candidate_indices=train_normal_candidates,
    )
    normal_subset, *_ = selector.run_and_make_subset(visualize=False)

    train_dataset = ConcatDataset([train_lesion_subset, normal_subset])

    logging.info(
        f'  train 환자 {len(train_patients)}명 (병변슬라이스 {len(train_lesion_idx)} + '
        f'정상대표 {len(normal_subset)}) / val 환자 {len(val_patients)}명 '
        f'(자연비율 {len(val_subset)}슬라이스) / test 환자 {len(test_patients)}명 '
        f'(자연비율 {len(test_subset)}슬라이스)'
    )
    return train_dataset, val_subset, test_subset


def grouped_kfold_patients(unique_patients, k, seed):
    """환자 목록을 k-fold로 나누고, 남은 trainval은 다시 train/val로 나눈다.
    전부 '환자' 단위(문자열 리스트)라서 sklearn의 plain KFold/train_test_split로 충분함
    (각 환자가 정확히 한 번씩만 등장하므로 GroupKFold가 필요한 상황이 아님 -
    실제 슬라이스 레벨 누수 방지는 이 함수가 나눈 환자 집합을 그대로 슬라이스
    필터링에 쓰는 build_fold_datasets에서 보장됨)."""
    patients = np.array(sorted(unique_patients))
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    for fold, (trainval_i, test_i) in enumerate(kf.split(patients)):
        trainval_patients = patients[trainval_i]
        test_patients = set(patients[test_i])
        train_p, val_p = train_test_split(trainval_patients, test_size=0.25, random_state=seed)
        yield fold, set(train_p), set(val_p), test_patients


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='CBAGS_grouped')
    parser.add_argument('--root_dir', default='../data/h5')
    parser.add_argument('--context', type=int, default=0)
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=None, help='지정 시 hyper_parameters의 기본 epochs를 덮어씀')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--only_fold', type=int, default=None, help='특정 fold 하나만 학습(디버그용)')
    args = parser.parse_args()

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    torch.set_default_dtype(torch.float32)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(torch.cuda.is_available(), ':', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')

    name = args.name
    context = args.context

    logging.basicConfig(
        filename=f'{name}.out', filemode='w', level=logging.INFO,
        format='[%(asctime)s] %(message)s', datefmt='%m/%d %H:%M:%S',
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)

    tf = transforms.Compose([
        transforms.Lambda(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8))
    ])
    target_dir = ['5mm 미만', '5mm 이상', 'normal', '추가 병변']

    logging.info('전체 데이터셋(자연 비율) 로드 중...')
    full_dataset = Dicom2dDataset(root_dir=args.root_dir, target_dir=target_dir, transform=tf, context=context, only_lesion=False)
    logging.info('병변 전용 데이터셋 로드 중...')
    lesion_dataset = Dicom2dDataset(root_dir=args.root_dir, target_dir=target_dir, transform=tf, context=context, only_lesion=True)

    full_groups = get_patient_groups(full_dataset, context)
    lesion_groups = get_patient_groups(lesion_dataset, context)
    unique_patients = sorted(set(full_groups))
    logging.info(f'전체 환자 수: {len(unique_patients)}')

    hp_dummy = HP(name=name, model=None)  # 경로 준비용

    loss_keys = ['Focal', 'Dice', 'BCE', 'GUL']
    rate_keys = ['Precision', 'Recall', 'Dice', 'Specificity', 'ACC']

    for fold, train_patients, val_patients, test_patients in grouped_kfold_patients(unique_patients, args.k, seed=42):
        if args.only_fold is not None and fold != args.only_fold:
            continue

        logging.info(f'[Fold {fold}]')
        assert not (train_patients & val_patients), '누수: train/val 환자 겹침'
        assert not (train_patients & test_patients), '누수: train/test 환자 겹침'
        assert not (val_patients & test_patients), '누수: val/test 환자 겹침'

        hp = HP(model=SuperEnhancedAFMSUNet2D(out_channels=1, context=context), name=name)
        if args.epochs is not None:
            hp.epochs = args.epochs

        train_subset, val_subset, test_subset = build_fold_datasets(
            full_dataset, lesion_dataset, full_groups, lesion_groups,
            train_patients, val_patients, test_patients,
            seed=hp.seed, context=context, device=device,
        )

        model = hp.model
        optimizer = torch.optim.AdamW(model.parameters(), lr=hp.optimizer_lr, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=hp.scheduler_step, gamma=hp.scheduler_gamma)

        train_loader, valid_loader, test_loader = get_loaders(
            train_subset, val_subset, test_subset, batch_size=hp.batch_size,
            collate_fn=custom_collate_fn, worker_init_fn=seed_worker(),
            g=torch.Generator().manual_seed(hp.seed),
        )
        data_loader = [train_loader, valid_loader]

        if hp.multi_gpu:
            model = nn.DataParallel(model)
        model = model.to(hp.device)
        torch.cuda.empty_cache()

        Focal_func, BCE_func, dice_func, gul_func = FocalLoss(), BCE(), DiceLoss(), GeneralUnionLoss()
        Recall_func, Precision_func, DiceScore_func = Recall(), Precision(), DiceScore()
        Specificity_func, ACC_func = Specificity(), ACC()

        epoch_s, epoch_e = 1, hp.epochs + 1
        scaler = torch.cuda.amp.GradScaler()
        loss_epoch = torch.zeros([epoch_e, 2, len(loss_keys)])
        rate_epoch = torch.zeros([epoch_e, 2, len(rate_keys)])
        best_metric = float('inf')

        for epoch in range(epoch_s, epoch_e):
            for i, phase in enumerate(['Train', 'Valid']):
                if phase == 'Train':
                    model.train()
                    context_manager = contextlib.nullcontext()
                else:
                    model.eval()
                    context_manager = torch.no_grad()

                loss_batch = torch.zeros([len(data_loader[i]), len(loss_keys)])
                rate_batch = torch.zeros([len(data_loader[i]), len(rate_keys)])
                with context_manager:
                    for k, data in enumerate(data_loader[i]):
                        X, y = data[0].to(hp.device), data[1].to(hp.device)

                        with torch.cuda.amp.autocast(enabled=False):
                            p_main, p_coarse, p_seg4 = model(X)

                            Focal_res = Focal_func(p_main, y)
                            Dice_res = dice_func(p_main, y)
                            BCE_res = BCE_func(p_main, y)
                            GUL_res = gul_func(p_main, y)

                            p_coarse_up = F.interpolate(p_coarse, size=y.shape[2:], mode='bilinear', align_corners=False)
                            p_seg4_up = F.interpolate(p_seg4, size=y.shape[2:], mode='bilinear', align_corners=False)

                            Focal_coarse = Focal_func(p_coarse_up, y) * 0.3
                            Dice_coarse = dice_func(p_coarse_up, y) * 0.3
                            GUL_coarse = gul_func(p_coarse_up, y) * 0.3
                            Focal_seg4 = Focal_func(p_seg4_up, y) * 0.2
                            Dice_seg4 = dice_func(p_seg4_up, y) * 0.2

                            loss_batch[k, 0] = Focal_res.detach()
                            loss_batch[k, 1] = Dice_res.detach()
                            loss_batch[k, 2] = BCE_res.detach()
                            loss_batch[k, 3] = GUL_res.detach()

                            loss_final = (Focal_res * hp.focal_loss_weight + Dice_res * hp.dice_loss_weight
                                          + BCE_res * hp.bce_loss_weight + GUL_res * hp.gul_loss_weight
                                          + (Focal_coarse + Dice_coarse + GUL_coarse) * 0.3
                                          + (Dice_seg4 + Focal_seg4) * 0.2)

                        if phase == 'Train':
                            optimizer.zero_grad()
                            scaler.scale(loss_final).backward()
                            scaler.step(optimizer)
                            scaler.update()

                        with torch.no_grad():
                            rate_batch[k, 0] = Precision_func(p_main, y)
                            rate_batch[k, 1] = Recall_func(p_main, y)
                            rate_batch[k, 2] = DiceScore_func(p_main, y)
                            rate_batch[k, 3] = Specificity_func(p_main, y)
                            rate_batch[k, 4] = ACC_func(torch.sigmoid(p_main), y)

                loss_epoch[epoch, i] = torch.mean(loss_batch.cpu(), axis=0)
                rate_epoch[epoch, i] = torch.mean(rate_batch.cpu(), axis=0)

            if scheduler is not None:
                scheduler.step()

            if epoch % hp.monitoring_cycle == 0:
                msg = f'{epoch:5.0f}/{hp.epochs:5.0f} '
                for l in range(len(loss_keys)):
                    msg += f'({loss_epoch[epoch,0,l]:6.4f}, {loss_epoch[epoch,1,l]:6.4f}) '
                msg += '* '
                for l in range(len(rate_keys)):
                    msg += f'({rate_epoch[epoch,0,l]:6.4f}, {rate_epoch[epoch,1,l]:6.4f}) '
                logging.info(msg)

                history = {'loss': loss_epoch, 'rate': rate_epoch, 'loss_keys': loss_keys, 'rate_keys': rate_keys}
                make_dir(f'{hp.path_model}/fold_{fold}')
                torch.save(history, f'{hp.path_model}/fold_{fold}/history.pt')

                if epoch % hp.save_cycle == 0:
                    torch.save(model.state_dict(), f'{hp.path_model}/fold_{fold}/model_{epoch}.pt')

            if loss_epoch[epoch, 1, 3] < best_metric:
                best_metric = loss_epoch[epoch, 1, 3].item()
                torch.save(model.state_dict(), f'{hp.path_model}/fold_{fold}/best.pt')

        logging.info(f'[Fold {fold}] 학습 종료. best GUL(valid)={best_metric:.4f}')
        del model, optimizer, scheduler, train_loader, valid_loader, test_loader
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
