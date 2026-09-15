"""Pure helpers for balanced concept-exposure assignment."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any


def validate_balanced_targets(
    concepts: Sequence[Mapping[str, Any]], target_concepts: Sequence[str]
) -> None:
    """Require six valid targets with every color and shape represented once."""
    concept_by_id = {str(concept["id"]): concept for concept in concepts}
    if len(target_concepts) != 6 or len(set(target_concepts)) != 6:
        raise ValueError("Exactly six unique target concepts are required")
    unknown = set(target_concepts) - set(concept_by_id)
    if unknown:
        raise ValueError(f"Unknown target concepts: {sorted(unknown)}")
    targets = [concept_by_id[concept_id] for concept_id in target_concepts]
    color_counts = Counter(str(concept["color"]) for concept in targets)
    shape_counts = Counter(str(concept["shape"]) for concept in targets)
    if set(color_counts.values()) != {1} or set(shape_counts.values()) != {1}:
        raise ValueError("Targets must contain every color and every shape exactly once")


def build_count_plan(
    concept_ids: Sequence[str],
    target_concepts: Sequence[str],
    n_full: int,
    exposure_level: float,
) -> dict[str, int]:
    """Return an exactly balanced, constant-budget count for every concept."""
    if n_full <= 0:
        raise ValueError("n_full must be positive")
    if not 0.0 <= exposure_level <= 1.0:
        raise ValueError("exposure_level must be within [0, 1]")
    target_set = set(target_concepts)
    non_targets = [concept_id for concept_id in concept_ids if concept_id not in target_set]
    if not target_set or not non_targets:
        raise ValueError("Both target and non-target concepts are required")

    target_count = round(n_full * exposure_level)
    removed = len(target_set) * (n_full - target_count)
    added, remainder = divmod(removed, len(non_targets))
    if remainder:
        raise ValueError(
            "Removed target budget cannot be redistributed equally across non-targets"
        )
    return {
        concept_id: target_count if concept_id in target_set else n_full + added
        for concept_id in concept_ids
    }
