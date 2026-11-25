#!/usr/bin/env python
"""Print a Markdown results table from eval/runs/*/metrics.json.

    python eval/results_table.py [--all]

Only reportable runs are listed unless --all is given. The README results
section is pasted from this output, so every number in it traces to a file.
"""

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="eval/runs")
    ap.add_argument("--all", action="store_true", help="include non-reportable runs")
    args = ap.parse_args()
    rows = []
    for p in sorted(Path(args.runs).glob("*/metrics.json")):
        m = json.loads(p.read_text())
        if not (m.get("reportable") or args.all):
            continue
        a, f = m["summary"]["accuracy"], m["summary"]["macro_f1"]
        folds = " · ".join(f"{100 * x['accuracy']:.2f}".rstrip("0").rstrip(".") for x in m["per_fold"])
        rows.append((a["mean"], (
            f"| [`{m['run_id']}`]({p.as_posix()}) | {m['model']} | "
            f"**{100 * a['mean']:.1f}** ± {100 * a['std']:.1f} | {folds} | {100 * f['mean']:.1f} ± {100 * f['std']:.1f} | "
            f"`{m['git_revision'][:7]}` |"
        )))
    print("| Run (links to metrics.json) | Model | Accuracy % | Per-fold accuracy % (folds 1–5) | Macro F1 % | Git |")
    print("| --- | --- | --- | --- | --- | --- |")
    print("\n".join(r for _, r in sorted(rows, key=lambda t: -t[0])))


if __name__ == "__main__":
    main()
