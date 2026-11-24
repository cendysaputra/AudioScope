#!/usr/bin/env python
"""Run official-fold cross-validation and write a run directory.

    python eval/crossval.py --config configs/baseline.yaml --out eval/runs/baseline-v1

Writes config.yaml, metrics.json, train.log, environment.txt, per-fold and
summed confusion matrices (CSV + PNG), predictions.csv, and one checkpoint
per fold. Only runs with ``"reportable": true`` in metrics.json (all folds,
clean git tree) may be quoted in the README.
"""

from audioscope.eval.crossval import main

if __name__ == "__main__":
    main()
