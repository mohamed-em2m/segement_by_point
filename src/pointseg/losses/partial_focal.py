"""Partial focal loss for sparse point supervision.

The assignment defines::

    L_pFCE = sum_i [ FocalLoss(pred_i, GT_i) * MASK_labeled,i ] / sum_i MASK_labeled,i

Only pixels whose target is not the ignore index contribute to either the
numerator or the denominator. Per-pixel focal loss is::

    FL = -alpha_t * (1 - p_t)^gamma * log(p_t)
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from pointseg.constants import IGNORE_INDEX

logger = logging.getLogger(__name__)


class PartialFocalLoss(nn.Module):
    """Focal loss averaged over labelled (non-ignored) pixels only.

    Args:
        gamma: focusing parameter. ``gamma=0`` with ``alpha=None`` is exactly
            partial cross-entropy.
        alpha: optional per-class weighting. A scalar applies to every class; a
            sequence must have one entry per class. Weights should be estimated
            from the sparse labels only, never from full masks.
        ignore_index: target value for unlabelled pixels (255).
        strict: raise instead of returning a differentiable zero when a batch
            contains no labelled pixels at all.

    Notes:
        The calculation is always performed in float32, even under AMP, because
        ``(1 - p_t) ** gamma`` is sensitive to low-precision ``p_t``. Ignored
        pixels are masked out before the sum, so their logits receive exactly
        zero gradient.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float | Sequence[float] | None = None,
        ignore_index: int = IGNORE_INDEX,
        strict: bool = False,
    ) -> None:
        super().__init__()
        if gamma < 0:
            raise ValueError("gamma must be >= 0")
        self.gamma = float(gamma)
        self.ignore_index = int(ignore_index)
        self.strict = bool(strict)
        if alpha is None:
            self.register_buffer("alpha", None)
        else:
            alpha_t = torch.as_tensor(alpha, dtype=torch.float32)
            if alpha_t.ndim > 1:
                raise ValueError("alpha must be a scalar or a 1-D sequence")
            self.register_buffer("alpha", alpha_t)

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        if logits.ndim != 4:
            raise ValueError(f"logits must be [B, C, H, W], got {tuple(logits.shape)}")
        if target.shape != logits.shape[:1] + logits.shape[2:]:
            raise ValueError(
                f"target shape {tuple(target.shape)} does not match logits {tuple(logits.shape)}"
            )
        num_classes = logits.shape[1]
        if self.alpha is not None and self.alpha.numel() not in (1, num_classes):
            raise ValueError(
                f"alpha has {self.alpha.numel()} entries but there are {num_classes} classes"
            )

        logits_f = logits.float()
        target = target.long()
        valid = target != self.ignore_index
        num_valid = valid.sum()
        if num_valid.item() == 0:
            message = "batch contains no labelled pixels; partial loss is undefined"
            if self.strict:
                raise ValueError(message)
            logger.warning(message)
            return logits_f.sum() * 0.0

        # Clamp only for the gather; ignored positions are masked out afterwards.
        safe_target = target.clamp(min=0, max=num_classes - 1)
        log_pt = F.log_softmax(logits_f, dim=1).gather(1, safe_target.unsqueeze(1)).squeeze(1)
        p_t = log_pt.exp()
        focal = -((1.0 - p_t).clamp_min(0.0) ** self.gamma) * log_pt

        if self.alpha is not None:
            if self.alpha.numel() == 1:
                focal = focal * self.alpha[0]
            else:
                focal = focal * self.alpha[safe_target]

        loss = (focal * valid).sum() / num_valid.to(logits_f.dtype)
        return loss

    def extra_repr(self) -> str:
        alpha = "none" if self.alpha is None else self.alpha.detach().cpu().tolist()
        return (
            f"gamma={self.gamma}, alpha={alpha}, "
            f"ignore_index={self.ignore_index}, strict={self.strict}"
        )
