"""Average the class probabilities of finished cross-validation runs, fold by fold.

For each official fold, the fold checkpoints of every member run score that fold's
validation clips and test clips. The member weights are chosen on the validation
clips only (grid search over the simplex, step ``--step``); the test fold is scored
once with the chosen weights. Every member must have used the same validation
split, which the script checks.

    python eval/ensemble.py --runs eval/runs/A eval/runs/B --out eval/runs/ensemble-v1
"""

from __future__ import annotations

import argparse
import datetime as dt
import itertools
import json
import time
from pathlib import Path

import numpy as np
import torch

from ..data.catalog import load_catalog
from ..data.dataset import load_waveform_cache, split_indices
from ..features.frontend import FeatureConfig
from ..inference import Predictor
from ..train.loop import TRAIN_DEFAULTS
from ..utils import environment_report, get_logger, git_revision
from .metrics import fold_metrics, summarize
from .report import plot_confusion, write_confusion_csv, write_json


@torch.no_grad()
def _probs(pred: Predictor, waves: torch.Tensor, batch_size: int = 32) -> np.ndarray:
    out = []
    for i in range(0, len(waves), batch_size):
        out.append(torch.softmax(pred.model(pred.pipe(waves[i : i + batch_size])), dim=1).float())
    return torch.cat(out).numpy()


def weight_grid(n_members: int, step: float) -> list[np.ndarray]:
    ticks = int(round(1 / step))
    grid = []
    for combo in itertools.product(range(ticks + 1), repeat=n_members - 1):
        if sum(combo) <= ticks:
            grid.append(np.array([*combo, ticks - sum(combo)], dtype=float) / ticks)
    return grid


def choose_weights(val_probs: list[np.ndarray], y_val: np.ndarray, step: float) -> np.ndarray:
    """Highest validation accuracy; ties broken by validation log-loss, then by closeness to equal weights."""
    equal = np.full(len(val_probs), 1 / len(val_probs))
    best, best_key = None, None
    for w in weight_grid(len(val_probs), step):
        p = sum(wi * pi for wi, pi in zip(w, val_probs))
        acc = float((p.argmax(1) == y_val).mean())
        nll = float(-np.log(p[np.arange(len(y_val)), y_val] + 1e-12).mean())
        key = (-acc, round(nll, 6), float(np.abs(w - equal).sum()))
        if best_key is None or key < best_key:
            best, best_key = w, key
    return best


def run_ensemble(runs: list[str], out_dir: str, step: float = 0.1, threads: int | None = None,
                 cache_dir: str = "data/cache") -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    run_id = out.name
    logger = get_logger("audioscope", out / "train.log")
    if threads:
        torch.set_num_threads(threads)
    rev = git_revision()
    logger.info("run %s  ensemble of %s  git %s", run_id, runs, rev)
    (out / "environment.txt").write_text(environment_report())

    members = []
    for r in runs:
        m = json.loads((Path(r) / "metrics.json").read_text())
        if not m.get("reportable"):
            raise ValueError(f"{r} is not a reportable run")
        members.append({"dir": Path(r), "metrics": m})
    dataset = {m["metrics"]["dataset"] for m in members}
    if len(dataset) != 1:
        raise ValueError(f"members use different datasets: {dataset}")
    folds = members[0]["metrics"]["folds"]

    first_cfg = torch.load(members[0]["dir"] / f"fold{folds[0]}" / "model.pt", map_location="cpu",
                           weights_only=False)["config"]
    catalog = load_catalog(first_cfg["dataset"]["name"], first_cfg["dataset"]["root"])
    wave_cache: dict[str, torch.Tensor] = {}

    def waves_for(fcfg: FeatureConfig) -> torch.Tensor:
        key = f"{fcfg.sample_rate}_{fcfg.num_samples}"
        if key not in wave_cache:
            wave_cache[key] = load_waveform_cache(catalog, fcfg, cache_dir, logger)
        return wave_cache[key]

    per_fold, cms, preds_rows, t0 = [], [], [], time.time()
    for k in folds:
        val_probs, test_probs, splits = [], [], None
        for mem in members:
            ckpt = torch.load(mem["dir"] / f"fold{k}" / "model.pt", map_location="cpu", weights_only=False)
            tcfg = {**TRAIN_DEFAULTS, **ckpt["config"].get("train", {})}
            seed = int(tcfg["seed"]) * 100 + k
            tr, va, te = split_indices(catalog, k, tcfg["val_fraction"], seed=seed)
            if splits is None:
                splits = (va, te)
            elif not (np.array_equal(splits[0], va) and np.array_equal(splits[1], te)):
                raise ValueError(f"fold {k}: {mem['dir']} used a different validation or test split")
            pred = Predictor(ckpt)
            w = waves_for(pred.fcfg)
            val_probs.append(_probs(pred, w[va]))
            test_probs.append(_probs(pred, w[te]))
            logger.info("fold %d: scored %s (val acc %.4f)", k, mem["dir"].name,
                        (val_probs[-1].argmax(1) == catalog.targets[va]).mean())
        va, te = splits
        y_val, y_test = np.asarray(catalog.targets)[va], np.asarray(catalog.targets)[te]
        weights = choose_weights(val_probs, y_val, step)
        p_val = sum(wi * pi for wi, pi in zip(weights, val_probs))
        p_test = sum(wi * pi for wi, pi in zip(weights, test_probs))
        y_pred = p_test.argmax(1)
        m = fold_metrics(y_test, y_pred, catalog.num_classes)
        cm = m.pop("confusion_matrix")
        cms.append(cm)
        write_confusion_csv(cm, catalog.class_names, out / f"confusion_fold{k}.csv")
        per_fold.append({"fold": k, **m, "weights": weights.round(4).tolist(),
                         "val_accuracy": float((p_val.argmax(1) == y_val).mean())})
        for i, idx in enumerate(te):
            preds_rows.append((catalog.filenames[idx], k, int(y_test[i]), int(y_pred[i]), float(p_test[i, y_pred[i]])))
        logger.info("fold %d weights %s  val acc %.4f  TEST accuracy %.4f  macro_f1 %.4f", k,
                    weights.round(2).tolist(), per_fold[-1]["val_accuracy"], m["accuracy"], m["macro_f1"])

    all_folds = sorted(set(catalog.folds.tolist()))
    complete = sorted(folds) == all_folds and all(m["metrics"]["complete"] for m in members)
    total_cm = np.sum(cms, axis=0)
    write_confusion_csv(total_cm, catalog.class_names, out / "confusion_matrix.csv")
    plot_confusion(total_cm, catalog.class_names, out / "confusion_matrix.png", f"{run_id}: summed over test folds {folds}")
    with open(out / "predictions.csv", "w") as fh:
        fh.write("filename,fold,target,pred,confidence\n")
        for row in preds_rows:
            fh.write(f"{row[0]},{row[1]},{row[2]},{row[3]},{row[4]:.4f}\n")
    (out / "config.yaml").write_text(
        "# Probability-averaging ensemble; weights chosen per fold on the validation split.\n"
        f"members:\n" + "".join(f"  - {m['dir'].as_posix()}  # {m['metrics']['git_revision'][:7]}\n" for m in members)
        + f"weight_step: {step}\n"
    )
    pca = np.array([f["per_class_accuracy"] for f in per_fold])
    metrics = {
        "run_id": run_id,
        "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "dataset": catalog.name,
        "model": "ensemble(" + ", ".join(m["metrics"]["model"] for m in members) + ")",
        "config_path": str(out / "config.yaml"),
        "config_name": run_id,
        "members": [m["dir"].as_posix() for m in members],
        "git_revision": rev,
        "folds": folds,
        "complete": complete,
        "reportable": complete and not rev.endswith("-dirty"),
        "summary": {
            "accuracy": summarize([f["accuracy"] for f in per_fold]),
            "macro_f1": summarize([f["macro_f1"] for f in per_fold]),
        },
        "per_fold": per_fold,
        "per_class_accuracy_mean": dict(zip(catalog.class_names, np.nanmean(pca, axis=0).round(4).tolist())),
        "class_names": catalog.class_names,
        "wall_seconds": round(time.time() - t0, 1),
    }
    write_json(metrics, out / "metrics.json")
    s = metrics["summary"]
    logger.info("DONE %s: accuracy %.4f ± %.4f  macro_f1 %.4f ± %.4f (complete=%s)", run_id,
                s["accuracy"]["mean"], s["accuracy"]["std"], s["macro_f1"]["mean"], s["macro_f1"]["std"], complete)
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, help="reportable run directories with fold checkpoints")
    ap.add_argument("--out", required=True)
    ap.add_argument("--step", type=float, default=0.1, help="weight grid step")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--cache-dir", default="data/cache")
    a = ap.parse_args()
    run_ensemble(a.runs, a.out, a.step, a.threads, a.cache_dir)


if __name__ == "__main__":
    main()
