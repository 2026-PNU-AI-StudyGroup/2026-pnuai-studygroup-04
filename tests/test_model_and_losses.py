"""
CBAS 모델 forward pass 및 손실 함수의 최소 동작을 확인하는 스모크 테스트.
GPU/실제 데이터 없이 CPU + 합성(synthetic) 텐서만으로 돌아간다.

실행: (repo 루트에서) pytest tests/ -v
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import pytest

from networks.CBAS import SuperEnhancedAFMSUNet2D
from utils.loss import FocalLoss, DiceLoss, BCE, GeneralUnionLoss


@pytest.fixture
def model():
    return SuperEnhancedAFMSUNet2D(out_channels=1, context=0)


def test_forward_pass_output_shapes(model):
    """입력 (B,1,192,192) -> main/coarse/seg4 세 출력의 shape이 예상대로 나오는지 확인."""
    x = torch.randn(2, 1, 192, 192)
    p_main, p_coarse, p_seg4 = model(x)

    assert p_main.shape == (2, 1, 192, 192)
    assert p_coarse.shape[0] == 2 and p_coarse.shape[1] == 1
    assert p_seg4.shape[0] == 2 and p_seg4.shape[1] == 1
    # bottleneck/enc4 단에서 나오는 coarse 출력이라 원본보다 훨씬 작아야 함
    assert p_coarse.shape[-1] < 192
    assert p_seg4.shape[-1] < 192


def test_parameter_count(model):
    """자문 검토에서 확인된 파라미터 수(29,260,629)와 일치하는지 회귀 테스트."""
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params == 29_260_629


@pytest.mark.parametrize('loss_cls', [FocalLoss, DiceLoss, BCE, GeneralUnionLoss])
def test_loss_no_nan_on_empty_mask(loss_cls):
    """라벨이 전부 0(병변 없는 슬라이스)이어도 손실이 NaN/Inf로 터지지 않는지 확인."""
    pred = torch.randn(2, 1, 192, 192)
    target = torch.zeros(2, 1, 192, 192)

    loss_fn = loss_cls()
    loss = loss_fn(pred, target)

    assert torch.isfinite(loss).all(), f'{loss_cls.__name__}이(가) 빈 마스크에서 NaN/Inf를 출력함'


@pytest.mark.parametrize('loss_cls', [FocalLoss, DiceLoss, BCE, GeneralUnionLoss])
def test_loss_no_nan_on_full_mask(loss_cls):
    """라벨이 전부 1(전체가 병변)인 극단적인 경우도 안정적인지 확인."""
    pred = torch.randn(2, 1, 192, 192)
    target = torch.ones(2, 1, 192, 192)

    loss_fn = loss_cls()
    loss = loss_fn(pred, target)

    assert torch.isfinite(loss).all(), f'{loss_cls.__name__}이(가) 전체 병변 마스크에서 NaN/Inf를 출력함'
