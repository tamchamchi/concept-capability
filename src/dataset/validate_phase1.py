"""Validate the Phase 1 concept, support, and diversity configuration."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "dataset" / "configs"


def load_yaml_compatible_json(filename: str) -> dict[str, Any]:
    """Load a config. JSON is intentionally used because it is valid YAML 1.2."""
    with (CONFIG_DIR / filename).open(encoding="utf-8") as config_file:
        return json.load(config_file)


def validate_concepts(config: dict[str, Any]) -> tuple[list[str], list[str]]:
    colors = config["colors"]
    shapes = config["shapes"]
    concepts = config["concepts"]

    assert len(colors) == 6, "There must be exactly 6 colors"
    assert len(shapes) == 6, "There must be exactly 6 shapes"
    assert len(concepts) == 36, "There must be exactly 36 concepts"
    assert [item["id"] for item in colors] == list(range(6))
    assert [item["id"] for item in shapes] == list(range(6))

    color_names = [item["name"] for item in colors]
    shape_names = [item["name"] for item in shapes]
    assert len(set(color_names)) == len(color_names)
    assert len(set(shape_names)) == len(shape_names)

    expected_ids = {
        f"{color}_{shape}" for color in color_names for shape in shape_names
    }
    actual_ids = {item["id"] for item in concepts}
    assert actual_ids == expected_ids, "Concepts must be the full color x shape product"
    assert len(actual_ids) == len(concepts), "Concept IDs must be unique"

    for color in colors:
        rgb = color["rgb"]
        assert len(rgb) == 3 and all(0 <= channel <= 255 for channel in rgb)

    # Warn through a hard lower bound if two base colors become difficult to separate.
    minimum_rgb_distance = min(
        math.dist(left["rgb"], right["rgb"])
        for index, left in enumerate(colors)
        for right in colors[index + 1 :]
    )
    assert minimum_rgb_distance >= 60.0, "Base colors are too close in RGB space"
    return color_names, shape_names


def validate_support(
    config: dict[str, Any], color_names: list[str], shape_names: list[str]
) -> None:
    levels = config["support_levels"]
    matrix = config["matrix"]

    assert levels == [0, 1, 5, 20, 100, 500]
    assert config["shape_order"] == shape_names
    assert list(matrix) == color_names
    assert all(len(row) == len(shape_names) for row in matrix.values())
    assert all(sorted(row) == levels for row in matrix.values())

    for column_index in range(len(shape_names)):
        column = [matrix[color][column_index] for color in color_names]
        assert sorted(column) == levels, "Every shape must see every support level"

    counts = Counter(value for row in matrix.values() for value in row)
    assert counts == Counter({level: 6 for level in levels})
    assert sum(value for row in matrix.values() for value in row) == 3756


def validate_renderer(config: dict[str, Any], *, base_profile: bool = False) -> None:
    assert config["image_size"] == 32
    distribution = config["instance_distribution"]
    assert distribution["scope"] in {
        "global_for_all_concepts_and_support_levels",
        "global_for_all_concepts_and_exposure_regimes",
    }

    for axis in ("center_x", "center_y"):
        bounds = distribution["position"][axis]
        assert 0 <= bounds["min"] < bounds["max"] < config["image_size"]

    size = distribution["size"]
    if size["distribution"] == "constant":
        assert size["value"] > 0
    else:
        assert 0 < size["min"] < size["max"]
    assert set(distribution["rotation_degrees"]) == {
        "circle", "square", "triangle", "diamond", "cross", "star"
    }

    diversity = config["diversity_constraint"]
    assert diversity["enabled"] is True
    assert diversity["comparison_scope"] == "within_concept"
    assert math.isclose(sum(diversity["weights"].values()), 1.0)
    assert 0 < diversity["minimum_distance"] < 1

    if base_profile:
        assert config["profile"] == "concept_generalization_base_c0"
        assert size["distribution"] == "constant"
        assert size["value"] == 8.0
        assert all(
            rotation == {"distribution": "constant", "value": 0.0}
            for rotation in distribution["rotation_degrees"].values()
        )
        assert distribution["color_jitter"]["enabled"] is False
        assert distribution["brightness"]["enabled"] is False
        assert diversity["weights"] == {
            "position": 1.0,
            "size": 0.0,
            "rotation": 0.0,
            "color": 0.0,
        }


def main() -> None:
    concepts = load_yaml_compatible_json("concepts.yaml")
    support = load_yaml_compatible_json("support_matrix.yaml")
    renderer = load_yaml_compatible_json("renderer.yaml")
    base_renderer = load_yaml_compatible_json("renderer_base.yaml")

    color_names, shape_names = validate_concepts(concepts)
    validate_support(support, color_names, shape_names)
    validate_renderer(renderer)
    validate_renderer(base_renderer, base_profile=True)
    print("Phase 1 configuration is valid: 6 colors, 6 shapes, 36 concepts.")
    print("Support is balanced: 6 concepts per level, 3,756 training samples.")
    print("Base renderer is valid: fixed size/rotation/color; random position only.")


if __name__ == "__main__":
    main()
