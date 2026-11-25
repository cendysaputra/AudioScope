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


def test_ensemble_weight_choice_uses_validation_accuracy():
    import numpy as np

    from audioscope.eval.ensemble import choose_weights, weight_grid

    grid = weight_grid(3, 0.5)
    assert len(grid) == 6 and all(abs(w.sum() - 1) < 1e-9 for w in grid)
    y = np.array([0, 1, 0, 1])
    good = np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3], [0.4, 0.6]])
    bad = good[:, ::-1]
    w = choose_weights([bad, good], y, step=0.1)
    assert w[1] > w[0]
    p = w[0] * bad + w[1] * good
    assert (p.argmax(1) == y).all()
