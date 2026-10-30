"""Segmentation metrics accumulated over an entire split.

The confusion matrix is accumulated once over all validation images, so every
pixel has the same weight (this is not a per-image average). Pixels whose
ground truth is the ignore index are excluded. A class with no ground-truth
pixels in the split is reported as ``NaN`` and excluded from the mean, which is
the documented policy used by the report.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from pointseg.constants import CLASS_NAMES, IGNORE_INDEX, NUM_CLASSES


class ConfusionMatrix:
    """Pixel-level confusion matrix for semantic segmentation."""

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        ignore_index: int = IGNORE_INDEX,
    ) -> None:
        self.num_classes = int(num_classes)
        self.ignore_index = int(ignore_index)
        self.matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update(
        self, prediction: np.ndarray | torch.Tensor, target: np.ndarray | torch.Tensor
    ) -> None:
        prediction = _to_numpy(prediction).reshape(-1)
        target = _to_numpy(target).reshape(-1)
        if prediction.shape != target.shape:
            raise ValueError("prediction and target must have the same number of pixels")
        valid = target != self.ignore_index
        if not np.any(valid):
            return
        pred = prediction[valid].astype(np.int64)
        tgt = target[valid].astype(np.int64)
        if pred.min(initial=0) < 0 or pred.max(initial=0) >= self.num_classes:
            raise ValueError("prediction contains class ids outside the valid range")
        if tgt.min(initial=0) < 0 or tgt.max(initial=0) >= self.num_classes:
            raise ValueError("target contains class ids outside the valid range")
        indices = tgt * self.num_classes + pred
        counts = np.bincount(indices, minlength=self.num_classes**2)
        self.matrix += counts.reshape(self.num_classes, self.num_classes)

    def compute(self) -> dict[str, Any]:
        matrix = self.matrix.astype(np.float64)
        true_positive = np.diag(matrix)
        gt_count = matrix.sum(axis=1)
        pred_count = matrix.sum(axis=0)
        union = gt_count + pred_count - true_positive

        with np.errstate(divide="ignore", invalid="ignore"):
            iou = true_positive / union
            f1 = 2.0 * true_positive / (gt_count + pred_count)

        # A class absent from the ground truth is undefined, not zero.
        absent = gt_count == 0
        iou = np.where(absent, np.nan, iou)
        f1 = np.where(absent, np.nan, f1)

        total = matrix.sum()
        pixel_accuracy = float(true_positive.sum() / total) if total > 0 else float("nan")

        return {
            "per_class_iou": iou.tolist(),
            "per_class_f1": f1.tolist(),
            "per_class_gt_pixels": gt_count.astype(np.int64).tolist(),
            "per_class_pred_pixels": pred_count.astype(np.int64).tolist(),
            "miou": float(np.nanmean(iou)) if np.any(~absent) else float("nan"),
            "mean_f1": float(np.nanmean(f1)) if np.any(~absent) else float("nan"),
            "pixel_accuracy": pixel_accuracy,
            "num_valid_pixels": int(total),
            "class_names": list(CLASS_NAMES[: self.num_classes]),
            "confusion_matrix": self.matrix.tolist(),
        }


def _to_numpy(value: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)
