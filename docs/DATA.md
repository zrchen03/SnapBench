# SnapBench Data

## Two official copies

| Copy | URL | Format |
|------|-----|--------|
| GitHub (this repo) | https://github.com/zrchen03/SnapBench | `benchmark/snap_bench.json` + Git LFS images |
| Hugging Face | https://huggingface.co/datasets/yefd/SnapBench | `ImageFolder` (`queries` / `gallery`) |

Use Hugging Face if you want `datasets.load_dataset`. Use this repository if you want the original JSON, perturbation generators, or evaluation utilities.

## GitHub fields (`snap_bench.json`)

Each of the 1,145 records has:

| Field | Meaning |
|-------|---------|
| `query_id` | `bench_0000` … |
| `query_text` | clean English question |
| `query_entity` | coarse entity tag |
| `query_image_local` | filename under `bench_images/query/` |
| `domain` | `product` / `nature` / `food` / `person` / `place` / `culture` |
| `gallery.gt` | labeled positives (`local`, `caption`) |
| `gallery.hard_negative` | same-category hard negatives |
| `text_perturbations` | 8 operators → `perturbed_text` |
| `image_perturbations` | 15 operators → `sev1` / `sev2` / `sev3` paths |

The shared retrieval gallery is reconstructed by taking the unique `(local, caption)` pairs across all queries. That yields **9,085 items** backed by **9,059** image files (a few images have more than one caption). `snapbench.SnapBench` assigns stable IDs `gallery_00000` … after sorting those pairs.

Image paths:

- Query: `bench_images/query/{query_image_local}`
- Gallery: `bench_images/gallery/{local}`
- Perturbed query: `bench_images/perturbed/{type}/sev{1,2,3}/{query_id}.jpg`

## Hugging Face fields

```python
from datasets import load_dataset

queries = load_dataset("yefd/SnapBench", "queries", split="test")
gallery = load_dataset("yefd/SnapBench", "gallery", split="test")
```

Important query columns: `text`, `positive_gallery_ids`, `hard_negative_gallery_ids`, `text_perturbations`, `image_perturbations`.
Gallery columns: `caption` and the loaded `image`.

Hugging Face IDs (`gallery_XXXXX`) come from that release and are **not** the same strings as the GitHub loader IDs. Do not mix ID spaces when scoring.

## Conditions

- 8 text operators, no severity
- 15 image operators × 3 severities = 45 image conditions
- **53 corruption conditions** in the paper
- **54 evaluation states** if you also count clean

See `snapbench/conditions.py`.
