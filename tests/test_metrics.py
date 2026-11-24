import numpy as np
import pytest

from audioscope.eval.metrics import accuracy, confusion_matrix, fold_metrics, macro_f1, per_class_accuracy, summarize


def test_confusion_and_scores_against_sklearn():
    sk = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(0)
    y = rng.integers(0, 7, 300)
    p = np.where(rng.random(300) < 0.6, y, rng.integers(0, 7, 300))
    cm = confusion_matrix(y, p, 7)
    assert np.array_equal(cm, sk.confusion_matrix(y, p, labels=range(7)))
    assert accuracy(y, p) == pytest.approx(sk.accuracy_score(y, p))
    assert macro_f1(cm) == pytest.approx(sk.f1_score(y, p, average="macro"))
    assert np.allclose(per_class_accuracy(cm), sk.recall_score(y, p, average=None))


def test_perfect_and_summary():
    m = fold_metrics([0, 1, 2, 2], [0, 1, 2, 2], 3)
    assert m["accuracy"] == 1.0 and m["macro_f1"] == 1.0
    s = summarize([0.5, 0.7])
    assert s["mean"] == pytest.approx(0.6)
    assert s["std"] == pytest.approx(np.std([0.5, 0.7], ddof=1))
