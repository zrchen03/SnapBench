#!/usr/bin/env python3
"""Apply MOOR to four precomputed embedding matrices and evaluate.

Expected files (all ``.npy``):
    --q-image   [n_query, d]     query image embeddings
    --q-text    [n_query, d]     query text embeddings
    --g-image   [n_gallery, d]   gallery image embeddings
    --g-text    [n_gallery, d]   gallery caption embeddings

Query / gallery order must match ``snapbench.SnapBench``.
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

from snapbench import SnapBench, moor_fuse, recall_from_scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MOOR on SnapBench embeddings")
    parser.add_argument("--q-image", required=True)
    parser.add_argument("--q-text", required=True)
    parser.add_argument("--g-image", required=True)
    parser.add_argument("--g-text", required=True)
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--no-whiten", action="store_true", help="skip whitening (VLM2Vec models)")
    parser.add_argument("--save-scores", default=None, help="optional path to write fused scores")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bench = SnapBench(root=args.root)
    scores = moor_fuse(
        np.load(args.q_image),
        np.load(args.q_text),
        np.load(args.g_image),
        np.load(args.g_text),
        whiten=not args.no_whiten,
    )
    r_at_k = recall_from_scores(
        scores,
        bench.gallery_ids(),
        [q.positive_ids for q in bench.queries],
        k=args.k,
    )
    result = {
        "n_query": len(bench),
        "n_gallery": len(bench.gallery),
        "whiten": not args.no_whiten,
        f"MOOR_R@{args.k}": round(float(r_at_k) * 100, 2),
    }
    print(json.dumps(result, indent=2))
    if args.save_scores:
        np.save(args.save_scores, scores)


if __name__ == "__main__":
    main()
