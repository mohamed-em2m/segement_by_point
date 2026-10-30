from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from pointseg.constants import IGNORE_INDEX
from pointseg.losses import PartialCrossEntropyLoss, PartialFocalLoss


def _tiny_batch():
    logits = torch.tensor(
        [
            [
                [[2.0, -1.0, 0.5]],
                [[0.1, 1.5, -0.5]],
            ]
        ],
        dtype=torch.float32,
    )
    target = torch.tensor([[[0, 1, IGNORE_INDEX]]], dtype=torch.long)
    return logits, target


def test_gamma_zero_matches_cross_entropy():
    logits, target = _tiny_batch()
    focal = PartialFocalLoss(gamma=0.0)
    ce = PartialCrossEntropyLoss()
    expected = F.cross_entropy(logits, target, ignore_index=IGNORE_INDEX)
    assert torch.allclose(focal(logits, target), expected, atol=1e-6)
    assert torch.allclose(focal(logits, target), ce(logits, target), atol=1e-6)


def test_hand_computed_focal_value():
    logits, target = _tiny_batch()
    gamma = 2.0
    log_p = F.log_softmax(logits, dim=1)
    log_p_t = log_p.gather(1, target.clamp(0, 1).unsqueeze(1)).squeeze(1)
    p_t = log_p_t.exp()
    focal = -((1 - p_t) ** gamma) * log_p_t
    valid = target != IGNORE_INDEX
    expected = (focal * valid).sum() / valid.sum()
    loss = PartialFocalLoss(gamma=gamma)(logits, target)
    assert torch.allclose(loss, expected, atol=1e-6)


def test_ignored_logits_do_not_change_loss():
    logits, target = _tiny_batch()
    loss = PartialFocalLoss(gamma=2.0)(logits, target)
    perturbed = logits.clone()
    perturbed[0, :, 0, 2] += 100.0
    loss_perturbed = PartialFocalLoss(gamma=2.0)(perturbed, target)
    assert torch.allclose(loss, loss_perturbed)


def test_ignored_pixels_have_zero_gradient():
    logits, target = _tiny_batch()
    logits.requires_grad_(True)
    PartialFocalLoss(gamma=2.0)(logits, target).backward()
    assert torch.count_nonzero(logits.grad[0, :, 0, 2]) == 0


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_finite_with_extreme_logits(dtype):
    logits = torch.tensor(
        [
            [
                [[1e4, -1e4]],
                [[-1e4, 1e4]],
            ]
        ],
        dtype=dtype,
    )
    target = torch.tensor([[[0, 1]]], dtype=torch.long)
    loss = PartialFocalLoss(gamma=2.0)(logits, target)
    assert torch.isfinite(loss)


def test_all_ignored_default_is_differentiable_zero():
    logits = torch.randn(1, 7, 4, 4, requires_grad=True)
    target = torch.full((1, 4, 4), IGNORE_INDEX, dtype=torch.long)
    loss = PartialFocalLoss()(logits, target)
    assert loss.item() == 0.0
    loss.backward()
    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad) == 0


def test_all_ignored_strict_raises():
    logits = torch.randn(1, 7, 4, 4)
    target = torch.full((1, 4, 4), IGNORE_INDEX, dtype=torch.long)
    with pytest.raises(ValueError):
        PartialFocalLoss(strict=True)(logits, target)
    with pytest.raises(ValueError):
        PartialCrossEntropyLoss(strict=True)(logits, target)


def test_higher_gamma_lowers_confident_loss():
    logits = torch.tensor([[[[10.0]], [[0.0]]]])
    target = torch.tensor([[[0]]], dtype=torch.long)
    low = PartialFocalLoss(gamma=0.0)(logits, target)
    high = PartialFocalLoss(gamma=2.0)(logits, target)
    assert high < low


def test_alpha_is_applied_per_class():
    logits = torch.tensor([[[[2.0]], [[0.0]]]])
    target = torch.tensor([[[0]]], dtype=torch.long)
    base = PartialFocalLoss(gamma=0.0)(logits, target)
    weighted = PartialFocalLoss(gamma=0.0, alpha=[2.0, 1.0])(logits, target)
    assert torch.allclose(weighted, 2.0 * base, atol=1e-6)


def test_shape_validation():
    with pytest.raises(ValueError):
        PartialFocalLoss()(torch.randn(1, 7, 4, 4), torch.zeros(1, 3, 4, dtype=torch.long))
