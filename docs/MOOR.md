# MOOR

**MOOR** (Modality-anchored, Outlier-aware, Optimal Reweighting) is a training-free fusion adapter. It reweights four gallery score paths from a frozen encoder:

- `s_II`: query image → gallery image
- `s_TT`: query text → gallery caption
- `s_IT`: query image → gallery caption
- `s_TI`: query text → gallery image

`s_II` is the anchor. Each text-involving path is gated by its Pearson correlation with `s_II` through a bell function (suppress unreliable *and* redundant paths), then scaled by score variance. The fused score is the normalized weighted sum.

```text
g = max(0, r)^2 * max(0, 1 - r)^2     # peaks at r = 0.5
w_II = Var(s_II)
w_ab = g_ab * Var(s_ab)               # ab in {TT, IT, TI}
s    = (Σ w s) / (Σ w)
```

Embeddings are gallery-whitened unless you disable it. If every gate collapses, MOOR reduces to whitened image-only retrieval.

```python
from snapbench import moor_fuse

scores = moor_fuse(q_image, q_text, g_image, g_text)
scores, weights = moor_fuse(q_image, q_text, g_image, g_text, return_weights=True)
```

MOOR is a diagnostic baseline for fixed-fusion miscalibration, not a claim of a new SOTA encoder.
