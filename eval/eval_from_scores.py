#!/usr/bin/env python3
"""Evaluate precomputed similarity scores on SnapBench.

This is the official metric entrypoint. Encode queries and gallery with
your own model, save a score matrix, then run this script.

Score file
----------
A ``.npy`` array of shape ``[n_query, n_gallery]``. Rows must follow
``benchmark/snap_bench.json`` query order. Columns must follow the
stable gallery order used by ``snapbench.SnapBench`` (sorted
``(local, caption)``, IDs ``gallery_00000`` ...).

Example
-------
    python eval/eval_from_scores.py --scores scores_clean.npy --k 1
    python eval/eval_from_scores.py --scores scores_clean.npy --scores-corrupt scores_low_light_sev1.npy
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from snapbench import SnapBench
from snapbench.metrics import recall_from_scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SnapBench scores")
    parser.add_argument("--scores", required=True, help="clean or single-condition .npy scores")
    parser.add_argument("--scores-corrupt", default=None, help="optional paired corrupted scores")
    parser.add_argument("--root", default=str(ROOT), help="repository root")
    parser.add_argument("--k", type=int, default=1, help="Recall@k (paper reports R@1)")
    parser.add_argument("--out", default=None, help="optional JSON output path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bench = SnapBench(root=args.root)
    gallery_ids = bench.gallery_ids()
    positives = [q.positive_ids for q in bench.queries]

    scores = np.load(args.scores)
    r_clean = recall_from_scores(scores, gallery_ids, positives, k=args.k)
    result = {
        "n_query": len(bench),
        "n_gallery": len(gallery_ids),
        f"R@{args.k}": round(r_clean * 100, 2),
    }

    if args.scores_corrupt:
        scores_c = np.load(args.scores_corrupt)
        r_c = recall_from_scores(scores_c, gallery_ids, positives, k=args.k)
        result[f"R@{args.k}_corrupt"] = round(r_c * 100, 2)
        result[f"delta_R@{args.k}"] = round((r_c - r_clean) * 100, 2)

    print(json.dumps(result, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
