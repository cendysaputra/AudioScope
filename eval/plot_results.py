#!/usr/bin/env python
"""Plot ESC-50 5-fold accuracy of every reportable run in eval/runs/ (README figure).

    python eval/plot_results.py --out docs/results.png

Each row: one run. Small dots are the five held-out test folds, the large dot is
the mean, the bar is ± one sample standard deviation. Values come only from
``metrics.json`` files with ``"reportable": true``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LABELS = {
    "baseline-v1": "Baseline CNN",
    "augment-specaugment-v1": "+ SpecAugment",
    "augment-mixup-v1": "+ mixup",
    "augment-timeshift-v1": "+ time shift",
    "augment-noise-v1": "+ noise",
    "augment-combined-v1": "+ all four",
    "transfer-resnet18-v1": "ResNet-18 (ImageNet), fine-tuned",
    "transfer-panns-cnn14-frozen-v1": "PANNs Cnn14 (AudioSet), linear probe",
    "transfer-panns-cnn14-fc1-v1": "PANNs Cnn14, fc1 + head trained",
    "ensemble-cnn14fc1-resnet18-v1": "Ensemble: Cnn14 + ResNet-18",
}
GROUPS = [  # (title, run-id prefix test, color) in fixed categorical order
    ("CNN trained from scratch", lambda r: r.startswith(("baseline", "augment")), "#257de0"),
    ("ImageNet transfer", lambda r: r.startswith("transfer-resnet"), "#f47230"),
    ("AudioSet transfer", lambda r: r.startswith(("transfer-panns", "ensemble")), "#00b894"),
]
INK, INK2, GRID = "#191919", "#666666", "#e6e6e6"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", default="eval/runs")
    ap.add_argument("--out", default="docs/results.png")
    args = ap.parse_args()

    runs = []
    for p in sorted(Path(args.runs).glob("*/metrics.json")):
        m = json.loads(p.read_text())
        if m.get("reportable"):
            runs.append(m)
    order = list(LABELS)
    runs.sort(key=lambda m: order.index(m["run_id"]) if m["run_id"] in order else len(order))

    fig, ax = plt.subplots(figsize=(9, 0.4 * len(runs) + 1.4), dpi=200)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    for y, m in enumerate(runs):
        rid = m["run_id"]
        color = next(c for _, test, c in GROUPS if test(rid))
        folds = [f["accuracy"] * 100 for f in m["per_fold"]]
        mean = m["summary"]["accuracy"]["mean"] * 100
        std = m["summary"]["accuracy"]["std"] * 100
        ax.plot([mean - std, mean + std], [y, y], color=color, lw=2, solid_capstyle="round", zorder=2)
        ax.scatter(folds, [y] * len(folds), s=14, color=color, alpha=0.45, linewidths=0, zorder=3)
        ax.scatter([mean], [y], s=64, color=color, edgecolors="white", linewidths=2, zorder=4)
        ax.text(102.5, y, f"{mean:.1f} ± {std:.1f}", va="center", ha="right", fontsize=9, color=INK)

    ax.axvline(81.3, color=INK2, lw=1, ls=(0, (3, 3)), zorder=1)
    ax.text(81.6, -0.75, "human listeners 81.3% (Piczak, 2015)", ha="left", va="center", fontsize=7.5, color=INK2)
    ax.text(102.5, -0.75, "mean ± std", ha="right", va="center", fontsize=7.5, color=INK2)
    ax.set_yticks(range(len(runs)))
    ax.set_yticklabels([LABELS.get(m["run_id"], m["run_id"]) for m in runs], fontsize=9, color=INK)
    ax.invert_yaxis()
    ax.set_ylim(len(runs) - 0.5, -1.2)
    ax.set_xlim(70, 103)
    ax.set_xticks(range(70, 101, 5))
    ax.set_xlabel("Test accuracy on ESC-50, % (5 official folds)", fontsize=9, color=INK2)
    ax.tick_params(axis="x", colors=INK2, labelsize=8)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, markersize=7, label=t) for t, _, c in GROUPS]
    handles.append(plt.Line2D([], [], marker="o", ls="", color=INK2, alpha=0.45, markersize=4, label="one test fold"))
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.42, 1.0), ncol=4, frameon=False, fontsize=8,
              labelcolor=INK2, handletextpad=0.3, columnspacing=1.2)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, facecolor="white")
    print(f"wrote {args.out} ({len(runs)} runs)")


if __name__ == "__main__":
    main()
