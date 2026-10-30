from __future__ import annotations

import numpy as np
import pytest

from pointseg.constants import IGNORE_INDEX
from pointseg.evaluation.metrics import ConfusionMatrix


def test_hand_computed_metrics():
    cm = ConfusionMatrix(num_classes=3)
    prediction = np.array([[0, 1], [2, 2]])
    target = np.array([[0, 1], [0, 2]])
    cm.update(prediction, target)
    metrics = cm.compute()
    np.testing.assert_allclose(metrics["per_class_iou"], [0.5, 1.0, 0.5])
    assert metrics["miou"] == pytest.approx(2 / 3)
    assert metrics["pixel_accuracy"] == pytest.approx(0.75)
    np.testing.assert_allclose(metrics["per_class_f1"], [2 / 3, 1.0, 2 / 3])


def test_ignore_index_is_excluded():
    cm = ConfusionMatrix(num_classes=2)
    cm.update(np.array([[0, 1]]), np.array([[0, IGNORE_INDEX]]))
    metrics = cm.compute()
    assert metrics["num_valid_pixels"] == 1
    assert metrics["pixel_accuracy"] == 1.0


def test_absent_class_is_nan_and_excluded():
    cm = ConfusionMatrix(num_classes=3)
    cm.update(np.array([[0, 1]]), np.array([[0, 1]]))
    metrics = cm.compute()
    assert np.isnan(metrics["per_class_iou"][2])
    assert metrics["miou"] == pytest.approx(1.0)


def test_perfect_prediction():
    cm = ConfusionMatrix(num_classes=7)
    target = np.arange(7 * 7).reshape(7, 7) % 7
    cm.update(target, target)
    metrics = cm.compute()
    assert metrics["miou"] == pytest.approx(1.0)
    assert metrics["mean_f1"] == pytest.approx(1.0)
    assert metrics["pixel_accuracy"] == pytest.approx(1.0)


def test_out_of_range_prediction_raises():
    cm = ConfusionMatrix(num_classes=3)
    with pytest.raises(ValueError):
        cm.update(np.array([[3]]), np.array([[0]]))
