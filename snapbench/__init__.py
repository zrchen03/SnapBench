"""SnapBench: paired snap-and-ask multimodal retrieval benchmark."""

from .conditions import IMAGE_OPERATORS, IMAGE_PRIMITIVES, TEXT_OPERATORS
from .data import GalleryItem, Query, SnapBench
from .metrics import recall_at_k, recall_from_scores
from .moor import moor_fuse

__all__ = [
    "IMAGE_OPERATORS",
    "IMAGE_PRIMITIVES",
    "TEXT_OPERATORS",
    "GalleryItem",
    "Query",
    "SnapBench",
    "moor_fuse",
    "recall_at_k",
    "recall_from_scores",
]
