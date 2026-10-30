"""Explicit partial cross-entropy loss for sparse point supervision.

``sum_i CE(pred_i, GT_i) * MASK_labeled,i / sum_i MASK_labeled,i``

This is kept as a separate class because the assignment lists partial CE as the
base formulation. ``PartialFocalLoss(gamma=0, alpha=None)`` must match it; that
equivalence is covered by tests.
"""

from __future__ import annotations

import logging

import torch.nn.functional as F
from torch import Tensor, nn

from pointseg.constants import IGNORE_INDEX

logger = logging.getLogger(__name__)


class PartialCrossEntropyLoss(nn.Module):
    """Cross-entropy averaged over labelled pixels only.

    Args:
        ignore_index: target value for unlabelled pixels (255).
        strict: raise instead of returning a differentiable zero when a batch
            contains no labelled pixels.
    """

    def __init__(
        self,
        ignore_index: int = IGNORE_INDEX,
        strict: bool = False,
    ) -> None:
        super().__init__()
        self.ignore_index = int(ignore_index)
        self.strict = bool(strict)

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        if logits.ndim != 4:
            raise ValueError(f"logits must be [B, C, H, W], got {tuple(logits.shape)}")
        if target.shape != logits.shape[:1] + logits.shape[2:]:
            raise ValueError(
                f"target shape {tuple(target.shape)} does not match logits {tuple(logits.shape)}"
            )
        logits_f = logits.float()
        target = target.long()
        num_valid = (target != self.ignore_index).sum()
        if num_valid.item() == 0:
            message = "batch contains no labelled pixels; partial loss is undefined"
            if self.strict:
                raise ValueError(message)
            logger.warning(message)
            return logits_f.sum() * 0.0

        total = F.cross_entropy(
            logits_f,
            target,
            ignore_index=self.ignore_index,
            reduction="sum",
        )
        return total / num_valid.to(logits_f.dtype)

    def extra_repr(self) -> str:
        return f"ignore_index={self.ignore_index}, strict={self.strict}"
