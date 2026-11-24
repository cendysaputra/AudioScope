"""K-fold cross-validation on the official folds. Writes a reportable run.

Library entry point for ``eval/crossval.py``; also runnable as
``python -m audioscope.eval.crossval --config ... --out eval/runs/<run-id>``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import time
from pathlib import Path

import numpy as np
import torch

from ..config import dump_config, load_config
from ..data import load_catalog, load_waveform_cache
from ..features import FeatureConfig
from ..train.loop import save_checkpoint, train_fold
from ..utils import environment_report, get_logger, git_revision, resolve_device
from .metrics import fold_metrics, summarize
from .report import plot_confusion, write_confusion_csv, write_json


def run_crossval(
    config_path: str | Path,
    out_dir: str | Path,
    folds: list[int] | None = None,
    device: str = "auto",
    threads: int | None = None,
    cache_dir: str | None = "data/cache",
    save_checkpoints: bool = True,
    overrides: dict | None = None,
) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    run_id = out.name
    logger = get_logger("audioscope", out / "train.log")
    if threads:
        torch.set_num_threads(threads)

    cfg = load_config(config_path)
    if overrides:
        from ..config import deep_merge

        cfg = deep_merge(cfg, overrides)
    cfg["config_path"] = str(config_path)
    dump_config(cfg, out / "config.yaml")
    (out / "environment.txt").write_text(environment_report())
    rev = git_revision()
    logger.info("run %s  config %s  git %s", run_id, config_path, rev)

    catalog = load_catalog(cfg["dataset"]["name"], cfg["dataset"]["root"])
    fcfg = FeatureConfig.from_dict(cfg.get("features"))
    all_folds = sorted(set(catalog.folds.tolist()))
    folds = folds or all_folds
    logger.info("dataset %s: %d clips, %d classes, folds %s", catalog.name, len(catalog), catalog.num_classes, all_folds)
    waves = load_waveform_cache(catalog, fcfg, cache_dir, logger)
    dev = resolve_device(device)

    per_fold, cms = [], []
    preds_rows = []
    t0 = time.time()
    for k in folds:
        res = train_fold(cfg, catalog, waves, k, dev, logger, run_id=run_id, cache_dir=cache_dir)
        y_pred = res.test_logits.argmax(axis=1)
        m = fold_metrics(res.test_targets, y_pred, catalog.num_classes)
        cm = m.pop("confusion_matrix")
        cms.append(cm)
        write_confusion_csv(cm, catalog.class_names, out / f"confusion_fold{k}.csv")
        per_fold.append(
            {
                "fold": k,
                **m,
                "best_epoch": res.best_epoch,
                "epochs_run": res.epochs_run,
                "best_val_accuracy": res.best_val_accuracy,
                "train_seconds": round(res.train_seconds, 1),
            }
        )
        (out / f"history_fold{k}.json").write_text(json.dumps(res.history, indent=1))
        probs = torch.softmax(torch.from_numpy(res.test_logits), dim=1).numpy()
        for i, idx in enumerate(res.test_indices):
            preds_rows.append((catalog.filenames[idx], k, int(res.test_targets[i]), int(y_pred[i]), float(probs[i, y_pred[i]])))
        if save_checkpoints:
            save_checkpoint(res.checkpoint, out / f"fold{k}" / "model.pt")
        logger.info("fold %d TEST accuracy %.4f  macro_f1 %.4f", k, m["accuracy"], m["macro_f1"])

    complete = sorted(folds) == all_folds
    total_cm = np.sum(cms, axis=0)
    write_confusion_csv(total_cm, catalog.class_names, out / "confusion_matrix.csv")
    plot_confusion(
        total_cm,
        catalog.class_names,
        out / "confusion_matrix.png",
        f"{run_id}: summed over test folds {folds}",
    )
    with open(out / "predictions.csv", "w") as fh:
        fh.write("filename,fold,target,pred,confidence\n")
        for row in preds_rows:
            fh.write(f"{row[0]},{row[1]},{row[2]},{row[3]},{row[4]:.4f}\n")

    pca = np.array([f["per_class_accuracy"] for f in per_fold])
    metrics = {
        "run_id": run_id,
        "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "dataset": catalog.name,
        "model": cfg["model"]["name"],
        "config_path": str(config_path),
        "config_name": cfg.get("name"),
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
    logger.info(
        "DONE %s: accuracy %.4f ± %.4f  macro_f1 %.4f ± %.4f over folds %s (complete=%s)",
        run_id,
        s["accuracy"]["mean"],
        s["accuracy"]["std"],
        s["macro_f1"]["mean"],
        s["macro_f1"]["std"],
        folds,
        complete,
    )
    return metrics


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Official-fold cross-validation for AudioScope.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True, help="run directory, e.g. eval/runs/<run-id>")
    ap.add_argument("--folds", type=int, nargs="*", default=None, help="subset of folds (makes the run non-reportable)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--cache-dir", default="data/cache")
    ap.add_argument("--no-checkpoints", action="store_true")
    args = ap.parse_args(argv)
    run_crossval(
        args.config,
        args.out,
        folds=args.folds,
        device=args.device,
        threads=args.threads,
        cache_dir=args.cache_dir,
        save_checkpoints=not args.no_checkpoints,
    )


if __name__ == "__main__":
    main()
