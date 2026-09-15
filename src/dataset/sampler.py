"""Deterministic sampling of within-concept instance parameters."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from .colors import RGB, sample_color
from .shapes import shape_bounds


def _sample_rotation(
    shape_name: str,
    rng: np.random.Generator,
    distribution_config: Mapping[str, Any],
) -> float:
    rotation_config = distribution_config["rotation_degrees"][shape_name]
    if rotation_config["distribution"] == "constant":
        return float(rotation_config["value"])
    if rotation_config["distribution"] == "uniform":
        return float(rng.uniform(rotation_config["min"], rotation_config["max"]))
    raise ValueError(f"Unsupported rotation distribution for {shape_name}")


def _sample_size(
    rng: np.random.Generator,
    distribution_config: Mapping[str, Any],
) -> float:
    size_config = distribution_config["size"]
    if size_config["distribution"] == "constant":
        return float(size_config["value"])
    if size_config["distribution"] == "uniform":
        return float(rng.uniform(size_config["min"], size_config["max"]))
    raise ValueError("Unsupported size distribution")


def _feasible_center_range(
    configured_minimum: float,
    configured_maximum: float,
    lower_shape_bound: float,
    upper_shape_bound: float,
    image_size: int,
    padding: float,
) -> tuple[float, float]:
    minimum = max(configured_minimum, padding - lower_shape_bound)
    maximum = min(configured_maximum, image_size - 1 - padding - upper_shape_bound)
    if minimum > maximum:
        raise ValueError(
            "No valid center can keep this shape inside the canvas; "
            "reduce size/padding or widen the canvas"
        )
    return minimum, maximum


def sample_instance_parameters(
    shape_name: str,
    base_rgb: RGB,
    renderer_seed: int,
    renderer_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Sample one deterministic nuisance-parameter tuple."""
    if not isinstance(renderer_seed, (int, np.integer)) or renderer_seed < 0:
        raise ValueError("renderer_seed must be a non-negative integer")

    rng = np.random.default_rng(int(renderer_seed))
    distribution = renderer_config["instance_distribution"]

    size = _sample_size(rng, distribution)
    rotation = _sample_rotation(shape_name, rng, distribution)

    left, top, right, bottom = shape_bounds(shape_name, size, rotation)
    position = distribution["position"]
    padding = float(position["canvas_padding"])
    image_size = int(renderer_config["image_size"])
    min_x, max_x = _feasible_center_range(
        position["center_x"]["min"],
        position["center_x"]["max"],
        left,
        right,
        image_size,
        padding,
    )
    min_y, max_y = _feasible_center_range(
        position["center_y"]["min"],
        position["center_y"]["max"],
        top,
        bottom,
        image_size,
        padding,
    )
    center_x = float(rng.uniform(min_x, max_x))
    center_y = float(rng.uniform(min_y, max_y))

    final_rgb, color_jitter, brightness = sample_color(base_rgb, rng, distribution)
    return {
        "center_x": center_x,
        "center_y": center_y,
        "size": size,
        "rotation": rotation,
        "color_jitter_r": color_jitter[0],
        "color_jitter_g": color_jitter[1],
        "color_jitter_b": color_jitter[2],
        "brightness": brightness,
        "final_rgb": final_rgb,
    }
