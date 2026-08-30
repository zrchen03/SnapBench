"""MOOR: Modality-anchored, Outlier-aware, Optimal Reweighting.

Faithful implementation of Algorithm 1 in the SnapBench paper.
The adapter is training-free and operates only on frozen encoder embeddings.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MoorWeights:
    w_ii: np.ndarray
    w_tt: np.ndarray
    w_it: np.ndarray
    w_ti: np.ndarray
    r_tt: np.ndarray
    r_it: np.ndarray
    r_ti: np.ndarray


def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    denom = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.clip(denom, eps, None)


def _whiten(
    x: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
) -> np.ndarray:
    return _l2_normalize((x - mu) / sigma)


def _gallery_stats(gallery: np.ndarray, eps: float) -> tuple[np.ndarray, np.ndarray]:
    mu = gallery.mean(axis=0, keepdims=True)
    sigma = gallery.std(axis=0, keepdims=True) + eps
    return mu, sigma


def _pearson(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-wise Pearson correlation. a, b: [n_query, n_gallery]."""
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    num = (a * b).sum(axis=1)
    den = np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1))
    return num / np.clip(den, eps, None)


def _bell_gate(r: np.ndarray) -> np.ndarray:
    """g = max(0, r)^2 * max(0, 1 - r)^2. Peaks at r = 0.5."""
    return np.maximum(0.0, r) ** 2 * np.maximum(0.0, 1.0 - r) ** 2


def moor_fuse(
    q_image: np.ndarray,
    q_text: np.ndarray,
    g_image: np.ndarray,
    g_text: np.ndarray,
    *,
    whiten: bool = True,
    eps: float = 1e-6,
    return_weights: bool = False,
) -> np.ndarray | tuple[np.ndarray, MoorWeights]:
    """Fuse four similarity paths into a per-query gallery score vector.

    Args:
        q_image: query image embeddings, shape [n_query, d] or [d].
        q_text: query text embeddings, same shape as ``q_image``.
        g_image: gallery image embeddings, shape [n_gallery, d].
        g_text: gallery text embeddings, shape [n_gallery, d].
        whiten: gallery-side whitening. Set False for VLM2Vec-Full / VLM2Vec-V2
            as noted in the paper appendix.
        eps: numerical stabilizer for standard deviation and norms.
        return_weights: also return the per-query path weights and correlations.

    Returns:
        Fused scores of shape [n_query, n_gallery], or ``(scores, weights)``.
    """
    q_image = np.asarray(q_image, dtype=np.float64)
    q_text = np.asarray(q_text, dtype=np.float64)
    g_image = np.asarray(g_image, dtype=np.float64)
    g_text = np.asarray(g_text, dtype=np.float64)

    if q_image.ndim == 1:
        q_image = q_image[None, :]
        q_text = q_text[None, :]

    if whiten:
        mu_i, sigma_i = _gallery_stats(g_image, eps)
        mu_t, sigma_t = _gallery_stats(g_text, eps)
        g_i = _whiten(g_image, mu_i, sigma_i)
        g_t = _whiten(g_text, mu_t, sigma_t)
        q_i = _whiten(q_image, mu_i, sigma_i)
        q_t = _whiten(q_text, mu_t, sigma_t)
        q_i_cross = _whiten(q_image, mu_t, sigma_t)
        q_t_cross = _whiten(q_text, mu_i, sigma_i)
    else:
        g_i = _l2_normalize(g_image)
        g_t = _l2_normalize(g_text)
        q_i = _l2_normalize(q_image)
        q_t = _l2_normalize(q_text)
        q_i_cross = q_i
        q_t_cross = q_t

    s_ii = q_i @ g_i.T
    s_tt = q_t @ g_t.T
    s_it = q_i_cross @ g_t.T
    s_ti = q_t_cross @ g_i.T

    r_tt = _pearson(s_ii, s_tt)
    r_it = _pearson(s_ii, s_it)
    r_ti = _pearson(s_ii, s_ti)

    g_tt = _bell_gate(r_tt)
    g_it = _bell_gate(r_it)
    g_ti = _bell_gate(r_ti)

    w_ii = s_ii.var(axis=1)
    w_tt = g_tt * s_tt.var(axis=1)
    w_it = g_it * s_it.var(axis=1)
    w_ti = g_ti * s_ti.var(axis=1)

    weight_sum = w_ii + w_tt + w_it + w_ti
    fallback = weight_sum < eps
    weight_sum = np.where(fallback, 1.0, weight_sum)

    scores = (
        w_ii[:, None] * s_ii
        + w_tt[:, None] * s_tt
        + w_it[:, None] * s_it
        + w_ti[:, None] * s_ti
    ) / weight_sum[:, None]
    scores = np.where(fallback[:, None], s_ii, scores)

    if not return_weights:
        return scores
    weights = MoorWeights(
        w_ii=w_ii,
        w_tt=w_tt,
        w_it=w_it,
        w_ti=w_ti,
        r_tt=r_tt,
        r_it=r_it,
        r_ti=r_ti,
    )
    return scores, weights
