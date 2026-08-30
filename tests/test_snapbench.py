#!/usr/bin/env python3
"""Lightweight checks for metrics, MOOR, and the official data loader."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from snapbench import SnapBench, moor_fuse, recall_at_k, recall_from_scores
from snapbench.conditions import N_CORRUPTION_CONDITIONS, N_EVAL_STATES, all_conditions
from snapbench.moor import _bell_gate


def test_metrics() -> None:
    assert recall_at_k(["a", "b", "c"], ["c"], k=1) == 0.0
    assert recall_at_k(["a", "b", "c"], ["c"], k=3) == 1.0
    scores = np.array([[0.1, 0.9, 0.2], [0.8, 0.1, 0.0]])
    r = recall_from_scores(scores, ["x", "y", "z"], [["y"], ["x"]], k=1)
    assert abs(r - 1.0) < 1e-9


def test_bell_gate() -> None:
    g = _bell_gate(np.array([-0.2, 0.0, 0.5, 1.0]))
    assert g[0] == 0.0 and g[1] == 0.0 and g[3] == 0.0
    assert abs(g[2] - 0.0625) < 1e-9


def test_moor_shapes() -> None:
    rng = np.random.default_rng(0)
    n_q, n_g, d = 4, 16, 8
    q_i = rng.normal(size=(n_q, d))
    q_t = rng.normal(size=(n_q, d))
    g_i = rng.normal(size=(n_g, d))
    g_t = rng.normal(size=(n_g, d))
    scores, weights = moor_fuse(q_i, q_t, g_i, g_t, return_weights=True)
    assert scores.shape == (n_q, n_g)
    assert weights.w_ii.shape == (n_q,)
    assert np.isfinite(scores).all()


def test_loader_and_conditions() -> None:
    assert N_CORRUPTION_CONDITIONS == 53
    assert N_EVAL_STATES == 54
    assert len(all_conditions()) == 54

    bench = SnapBench(root=ROOT)
    assert len(bench) == 1145
    assert len(bench.gallery) == 9085
    q = bench.queries[0]
    assert q.positive_ids
    assert q.text_for("sent_replace") == "What is this?"
    assert q.image_path.exists()
    assert bench.gallery[0].image_path.exists()


if __name__ == "__main__":
    test_metrics()
    test_bell_gate()
    test_moor_shapes()
    test_loader_and_conditions()
    print("ok")
