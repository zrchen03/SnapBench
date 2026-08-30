# Evaluating on SnapBench

The paper reports **macro-averaged Recall@1** over 1,145 queries, against the shared 9,085-item gallery. Clean and corrupted variants of a query share the same positives, so a robustness number is a paired delta.

## Retrieval modes

| Mode | Query | Gallery |
|------|-------|---------|
| II | image | image |
| TT | text | caption |
| IT | image + text | image + caption |
| TI | text | image |

The main tables use joint **IT → IT**. For dual-encoders, the paper's fixed fusion is the average of image-image and text-text scores. For VLM embedding models, use the official joint encoding interface.

## Official metric script

Encode with your own model. Save a score matrix `[n_query, n_gallery]` whose **row order** matches `benchmark/snap_bench.json` and whose **column order** matches `SnapBench.gallery_ids()` (sorted `(local, caption)`).

```bash
python eval/eval_from_scores.py --scores scores_clean.npy --k 1
python eval/eval_from_scores.py \
  --scores scores_clean.npy \
  --scores-corrupt scores_low_light_sev1.npy
```

## MOOR

MOOR is training-free. It needs four embedding matrices from the same frozen encoder:

```bash
python eval/run_moor.py \
  --q-image q_image.npy \
  --q-text  q_text.npy \
  --g-image g_image.npy \
  --g-text  g_text.npy
```

Pass `--no-whiten` for VLM2Vec-Full and VLM2Vec-V2 (see the paper appendix).

You can also call the library:

```python
from snapbench import moor_fuse

scores = moor_fuse(q_image, q_text, g_image, g_text)  # [n_query, n_gallery]
```

## Protocol notes (Appendix D)

- Keep the gallery fixed and precompute gallery embeddings once per model.
- Use seeded, deterministic image corruptions from `benchmark/gen_image_perturbations.py`.
- Report R@1 unless you explicitly study R@5 / R@10.
- Scope conclusions to snap-and-ask entity retrieval.

## Models in the paper

Dual-encoders: CLIP-ViT-L/14, SigLIP-SO400M, SigLIP2-SO400M, BLIP-ITM-L.

VLM embeddings: Jina-V4, Qwen3-VL-Emb-2B/8B, E5-V, VLM2Vec-Full, VLM2Vec-V2, UME-R1-7B, GME-2B/7B, Ops-MM-2B/7B, RzenEmbed-7B.

Per-model prompts and input sizes are in the paper appendix. End-to-end inference wrappers for all 16 checkpoints are not in this first release; pull requests that add a thin encoder adapter are welcome.
