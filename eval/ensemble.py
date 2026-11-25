#!/usr/bin/env python
"""Probability-averaging ensemble of finished runs; see audioscope.eval.ensemble.

    python eval/ensemble.py --runs eval/runs/A eval/runs/B --out eval/runs/ensemble-v1
"""

from audioscope.eval.ensemble import main

if __name__ == "__main__":
    main()
