"""Diversity metrics and deterministic rejection sampling helpers."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


def _normalized_delta(left: float, right: float, minimum: float, maximum: float) -> float:
    width = maximum - minimum
    return 0.0 if width == 0 else min(abs(left - right) / width, 1.0)


def normalized_parameter_distance(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    shape_name: str,
    renderer_config: Mapping[str, Any],
) -> float:
    """Compute the configured weighted distance between two rendered instances."""
    distribution = renderer_config["instance_distribution"]
    weights = renderer_config["diversity_constraint"]["weights"]

    position_config = distribution["position"]
    x_delta = _normalized_delta(
        float(left["center_x"]),
        float(right["center_x"]),
        position_config["center_x"]["min"],
        position_config["center_x"]["max"],
    )
    y_delta = _normalized_delta(
        float(left["center_y"]),
        float(right["center_y"]),
        position_config["center_y"]["min"],
        position_config["center_y"]["max"],
    )
    position_distance = math.hypot(x_delta, y_delta) / math.sqrt(2.0)

    size_config = distribution["size"]
    if size_config["distribution"] == "constant":
        size_distance = 0.0
    else:
        size_distance = _normalized_delta(
            float(left["size"]),
            float(right["size"]),
            size_config["min"],
            size_config["max"],
        )

    rotation_config = distribution["rotation_degrees"][shape_name]
    if rotation_config["distribution"] == "constant":
        rotation_distance = 0.0
    else:
        rotation_distance = _normalized_delta(
            float(left["rotation"]),
            float(right["rotation"]),
            rotation_config["min"],
            rotation_config["max"],
        )

    jitter_config = distribution["color_jitter"]
    jitter_width = jitter_config["max_delta"] - jitter_config["min_delta"]
    jitter_deltas = [
        _normalized_delta(
            float(left[f"color_jitter_{channel}"]),
            float(right[f"color_jitter_{channel}"]),
            jitter_config["min_delta"],
            jitter_config["max_delta"],
        )
        for channel in "rgb"
    ]
    jitter_distance = math.sqrt(sum(value * value for value in jitter_deltas)) / math.sqrt(3.0)
    if jitter_width == 0:
        jitter_distance = 0.0

    brightness_config = distribution["brightness"]
    brightness_distance = _normalized_delta(
        float(left["brightness"]),
        float(right["brightness"]),
        brightness_config["min"],
        brightness_config["max"],
    )
    color_distance = (jitter_distance + brightness_distance) / 2.0

    return (
        weights["position"] * position_distance
        + weights["size"] * size_distance
        + weights["rotation"] * rotation_distance
        + weights["color"] * color_distance
    )


def minimum_recent_distance(
    candidate: Mapping[str, Any],
    accepted: Sequence[Mapping[str, Any]],
    shape_name: str,
    renderer_config: Mapping[str, Any],
) -> float:
    """Return distance to the closest configured recent sample."""
    if not accepted:
        return math.inf
    window = int(renderer_config["diversity_constraint"]["recent_sample_window"])
    return min(
        normalized_parameter_distance(candidate, previous, shape_name, renderer_config)
        for previous in accepted[-window:]
    )
