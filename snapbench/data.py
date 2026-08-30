"""Load SnapBench metadata and resolve local / Hugging Face paths."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from .conditions import IMAGE_OPERATORS, SEVERITIES, TEXT_OPERATORS


def _default_root() -> Path:
    return Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class GalleryItem:
    item_id: str
    local: str
    caption: str
    image_path: Path


@dataclass
class Query:
    query_id: str
    text: str
    entity: str
    domain: str
    image_local: str
    image_path: Path
    positive_ids: list[str]
    hard_negative_ids: list[str]
    text_perturbations: dict
    image_perturbations: dict
    _bench: SnapBench = field(repr=False, compare=False)

    def text_for(self, operator: str | None = None) -> str:
        if operator is None or operator == "clean":
            return self.text
        if operator not in TEXT_OPERATORS:
            raise ValueError(f"unknown text operator: {operator}")
        return self.text_perturbations[operator]["perturbed_text"]

    def image_path_for(self, operator: str | None = None, severity: int | None = None) -> Path:
        if operator is None or operator == "clean":
            return self.image_path
        if operator not in IMAGE_OPERATORS:
            raise ValueError(f"unknown image operator: {operator}")
        if severity not in SEVERITIES:
            raise ValueError(f"severity must be one of {SEVERITIES}")
        rel = self.image_perturbations[operator][f"sev{severity}"]
        return self._bench.resolve_path(rel)


class SnapBench:
    """Official GitHub layout loader (``benchmark/snap_bench.json``)."""

    def __init__(
        self,
        root: str | os.PathLike | None = None,
        *,
        json_path: str | os.PathLike | None = None,
        images_dir: str | os.PathLike | None = None,
    ) -> None:
        self.root = Path(root) if root is not None else _default_root()
        env_images = os.environ.get("BENCH_IMAGES_DIR")
        self.images_dir = Path(images_dir) if images_dir is not None else (
            Path(env_images) if env_images else self.root / "bench_images"
        )
        self.json_path = Path(json_path) if json_path is not None else (
            self.root / "benchmark" / "snap_bench.json"
        )
        raw = json.loads(self.json_path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise ValueError(f"expected a list of queries in {self.json_path}")

        self.gallery, key_to_id = self._build_gallery(raw)
        self.queries = [self._build_query(record, key_to_id) for record in raw]
        self._by_id = {q.query_id: q for q in self.queries}

    def __len__(self) -> int:
        return len(self.queries)

    def __getitem__(self, query_id: str) -> Query:
        return self._by_id[query_id]

    def resolve_path(self, rel: str | os.PathLike) -> Path:
        path = Path(rel)
        if path.is_absolute():
            return path
        if str(path).startswith("bench_images/"):
            return self.root / path
        return self.images_dir / path

    def gallery_ids(self) -> list[str]:
        return [item.item_id for item in self.gallery]

    def qrels(self) -> dict[str, list[str]]:
        return {q.query_id: list(q.positive_ids) for q in self.queries}

    def _build_gallery(self, raw: list[dict]) -> tuple[list[GalleryItem], dict[tuple[str, str], str]]:
        keys: set[tuple[str, str]] = set()
        for record in raw:
            for item in record["gallery"]["gt"] + record["gallery"]["hard_negative"]:
                keys.add((item["local"], item["caption"]))
        ordered = sorted(keys)
        gallery: list[GalleryItem] = []
        key_to_id: dict[tuple[str, str], str] = {}
        for idx, (local, caption) in enumerate(ordered):
            item_id = f"gallery_{idx:05d}"
            key_to_id[(local, caption)] = item_id
            gallery.append(
                GalleryItem(
                    item_id=item_id,
                    local=local,
                    caption=caption,
                    image_path=self.images_dir / "gallery" / local,
                )
            )
        return gallery, key_to_id

    def _build_query(self, record: dict, key_to_id: dict[tuple[str, str], str]) -> Query:
        def _ids(items: list[dict]) -> list[str]:
            return [key_to_id[(it["local"], it["caption"])] for it in items]

        image_local = record["query_image_local"]
        return Query(
            query_id=record["query_id"],
            text=record["query_text"],
            entity=record["query_entity"],
            domain=record["domain"],
            image_local=image_local,
            image_path=self.images_dir / "query" / image_local,
            positive_ids=_ids(record["gallery"]["gt"]),
            hard_negative_ids=_ids(record["gallery"]["hard_negative"]),
            text_perturbations=record["text_perturbations"],
            image_perturbations=record["image_perturbations"],
            _bench=self,
        )


def load_hf(name: str = "yefd/SnapBench"):
    """Load the Hugging Face release. Requires ``pip install datasets``."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "Hugging Face loading needs the `datasets` package: pip install datasets"
        ) from exc
    queries = load_dataset(name, "queries", split="test")
    gallery = load_dataset(name, "gallery", split="test")
    return queries, gallery
