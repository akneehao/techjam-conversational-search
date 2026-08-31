"""In-memory category-tree indexing over the catalog's `categories` field.

Every product's `categories` is already an ordered root-to-leaf path (e.g.
["Clothing, Shoes & Jewelry", "Women", "Jewelry", "Earrings", "Hoop"]) --
there is no true multi-parent DAG in this catalog (unlike the ontology this
methodology was ported from), so no graph library or BFS is needed. A
product's full "subtree" (itself + any strictly-deeper descendants that
share its path as a prefix) falls straight out of `by_prefix`.
"""

from __future__ import annotations

from collections import defaultdict
from typing import NamedTuple

import numpy as np


class CategoryIndex(NamedTuple):
    path_by_asin: dict[str, tuple[str, ...]]
    siblings_by_path: dict[tuple[str, ...], set[str]]   # exact full-path match
    by_prefix: dict[tuple[str, ...], set[str]]            # every realized prefix, all depths
    prefix_centroid: dict[tuple[str, ...], np.ndarray]    # L2-normalized mean vector per prefix
    id_row: dict[str, int]                                 # asin -> row in doc_vecs


def build_category_index(
    catalog: dict[str, dict],
    doc_vecs: np.ndarray,
    dense_ids: list[str],
) -> CategoryIndex:
    id_row = {asin: i for i, asin in enumerate(dense_ids)}

    path_by_asin: dict[str, tuple[str, ...]] = {}
    for asin, product in catalog.items():
        if asin not in id_row:
            continue
        path = tuple(str(c) for c in (product.get("categories") or []) if str(c).strip())
        if path:
            path_by_asin[asin] = path

    siblings_by_path: dict[tuple[str, ...], set[str]] = defaultdict(set)
    by_prefix: dict[tuple[str, ...], set[str]] = defaultdict(set)
    for asin, path in path_by_asin.items():
        siblings_by_path[path].add(asin)
        for depth in range(1, len(path) + 1):
            by_prefix[path[:depth]].add(asin)

    prefix_centroid: dict[tuple[str, ...], np.ndarray] = {}
    for prefix, asins in by_prefix.items():
        rows = [id_row[a] for a in asins]
        mean = doc_vecs[rows].mean(axis=0)
        norm = float(np.linalg.norm(mean))
        prefix_centroid[prefix] = (mean / norm if norm > 0 else mean).astype("float32")

    return CategoryIndex(
        path_by_asin=path_by_asin,
        siblings_by_path=dict(siblings_by_path),
        by_prefix=dict(by_prefix),
        prefix_centroid=prefix_centroid,
        id_row=id_row,
    )


def descendants_of(cat_index: CategoryIndex, path: tuple[str, ...]) -> set[str]:
    """Products whose path is a strict extension of `path` (true descendants,
    not the same-depth siblings). Falls back to the empty set when nothing is
    deeper -- callers apply the same-path-sibling fallback themselves."""
    subtree = cat_index.by_prefix.get(path, set())
    same_depth = cat_index.siblings_by_path.get(path, set())
    return subtree - same_depth


def immediate_children_of(cat_index: CategoryIndex, path: tuple[str, ...]) -> set[str]:
    """Products exactly one category level deeper than `path`."""
    target_depth = len(path) + 1
    subtree = cat_index.by_prefix.get(path, set())
    return {a for a in subtree if len(cat_index.path_by_asin[a]) == target_depth}
