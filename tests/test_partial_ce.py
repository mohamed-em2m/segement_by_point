from __future__ import annotations

import torch
import torch.nn.functional as F

from pointseg.constants import IGNORE_INDEX
from pointseg.losses import PartialCrossEntropyLoss


def test_partial_ce_matches_pytorch_mean():
    logits = torch.randn(2, 7, 8, 8)
    target = torch.randint(0, 7, (2, 8, 8))
    target[0, :4, :4] = IGNORE_INDEX
    expected = F.cross_entropy(logits, target, ignore_index=IGNORE_INDEX, reduction="mean")
    assert torch.allclose(PartialCrossEntropyLoss()(logits, target), expected, atol=1e-6)


def test_partial_ce_ignores_unlabelled_gradients():
    logits = torch.randn(1, 7, 4, 4, requires_grad=True)
    target = torch.randint(0, 7, (1, 4, 4))
    target[0, 0, 0] = IGNORE_INDEX
    PartialCrossEntropyLoss()(logits, target).backward()
    assert torch.count_nonzero(logits.grad[0, :, 0, 0]) == 0
