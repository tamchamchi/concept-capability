"""Color palette handling and deterministic color augmentation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


RGB = tuple[int, int, int]


def build_palette(concepts_config: Mapping[str, Any]) -> dict[str, RGB]:
    """Return the configured color name -> RGB mapping."""
    palette = {
        item["name"]: tuple(int(channel) for channel in item["rgb"])
        for item in concepts_config["colors"]
    }
    if any(len(rgb) != 3 for rgb in palette.values()):
        raise ValueError("Every color must contain exactly three RGB channels")
    return palette


def sample_color(
    base_rgb: RGB,
    rng: np.random.Generator,
    distribution_config: Mapping[str, Any],
) -> tuple[RGB, tuple[int, int, int], float]:
    """Apply per-channel jitter followed by brightness scaling."""
    jitter_config = distribution_config["color_jitter"]
    if jitter_config["enabled"]:
        jitter_array = rng.integers(
            jitter_config["min_delta"],
            jitter_config["max_delta"] + 1,
            size=3,
        )
    else:
        jitter_array = np.zeros(3, dtype=np.int64)

    brightness_config = distribution_config["brightness"]
    if brightness_config["enabled"]:
        brightness = float(
            rng.uniform(brightness_config["min"], brightness_config["max"])
        )
    else:
        brightness = 1.0

    clip_min, clip_max = jitter_config["clip_rgb_to"]
    augmented = (np.asarray(base_rgb, dtype=np.float64) + jitter_array) * brightness
    final_array = np.clip(np.rint(augmented), clip_min, clip_max).astype(np.uint8)

    final_rgb = tuple(int(channel) for channel in final_array)
    jitter = tuple(int(channel) for channel in jitter_array)
    return final_rgb, jitter, brightness
