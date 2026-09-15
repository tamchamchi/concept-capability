"""Renderer-aware rule baseline for independent evaluator cross-checks."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from src.dataset.renderer import DEFAULT_CONFIG_DIR, load_config
from src.dataset.shapes import draw_shape_mask


def _normalized_crop(mask: np.ndarray) -> np.ndarray:
    foreground_y, foreground_x = np.nonzero(mask > 20)
    if len(foreground_x) == 0:
        return np.zeros((32, 32), dtype=np.float32)
    crop = mask[
        foreground_y.min() : foreground_y.max() + 1,
        foreground_x.min() : foreground_x.max() + 1,
    ]
    resized = Image.fromarray(crop.astype(np.uint8)).resize(
        (32, 32), Image.Resampling.BILINEAR
    )
    return np.asarray(resized, dtype=np.float32) / 255.0


class RuleBasedEvaluator:
    def __init__(
        self,
        concepts_config: Mapping[str, Any] | None = None,
        renderer_config: Mapping[str, Any] | None = None,
    ) -> None:
        concepts = concepts_config or load_config(DEFAULT_CONFIG_DIR / "concepts.yaml")
        renderer = renderer_config or load_config(
            DEFAULT_CONFIG_DIR / "renderer_base.yaml"
        )
        self.colors = {
            item["name"]: tuple(int(channel) for channel in item["rgb"])
            for item in concepts["colors"]
        }
        self.color_ids = {item["name"]: int(item["id"]) for item in concepts["colors"]}
        self.shape_ids = {item["name"]: int(item["id"]) for item in concepts["shapes"]}
        size = float(renderer["instance_distribution"]["size"]["value"])
        supersampling = int(renderer["supersampling_factor"])
        downsampling = str(renderer["downsampling_filter"])
        padding = float(renderer["instance_distribution"]["position"]["canvas_padding"])
        self.templates = {}
        self.template_areas = {}
        for shape_name in self.shape_ids:
            mask = np.asarray(
                draw_shape_mask(
                    shape_name,
                    32,
                    (16.0, 16.0),
                    size,
                    0.0,
                    supersampling,
                    downsampling,
                    padding,
                )
            )
            self.templates[shape_name] = _normalized_crop(mask)
            self.template_areas[shape_name] = float(mask.sum() / 255.0)

    def _predict_color(self, pixels: np.ndarray) -> str:
        intensity = pixels.max(axis=2)
        foreground = pixels[intensity > max(20, int(intensity.max() * 0.65))]
        if len(foreground) == 0:
            raise ValueError("Image contains no detectable foreground")
        observed = tuple(float(value) for value in np.median(foreground, axis=0))
        return min(
            self.colors,
            key=lambda name: math.dist(observed, self.colors[name]),
        )

    def _predict_shape(self, pixels: np.ndarray, color_name: str) -> str:
        dominant_channel = max(self.colors[color_name])
        intensity = pixels.max(axis=2).astype(np.float32)
        estimated_mask = np.clip(intensity / dominant_channel, 0.0, 1.0)
        area = float(estimated_mask.sum())
        ordered = ["square", "circle", "cross", "diamond"]
        lower_groups = ["circle", "cross", "diamond", "star"]
        for shape_name, lower_shape in zip(ordered, lower_groups):
            threshold = (
                self.template_areas[shape_name]
                + self.template_areas[lower_shape]
            ) / 2.0
            if area > threshold:
                return shape_name
        query = _normalized_crop((estimated_mask * 255.0).astype(np.uint8))
        return min(
            ("triangle", "star"),
            key=lambda name: float(np.square(query - self.templates[name]).mean()),
        )

    def predict(self, image: Image.Image | np.ndarray) -> dict[str, Any]:
        pixels = np.asarray(image.convert("RGB") if isinstance(image, Image.Image) else image)
        if pixels.shape != (32, 32, 3):
            raise ValueError("Evaluator expects a 32x32 RGB image")
        color_name = self._predict_color(pixels)
        shape_name = self._predict_shape(pixels, color_name)
        return {
            "color_name": color_name,
            "color_id": self.color_ids[color_name],
            "shape_name": shape_name,
            "shape_id": self.shape_ids[shape_name],
            "concept_id": f"{color_name}_{shape_name}",
        }

    def predict_path(self, image_path: Path) -> dict[str, Any]:
        with Image.open(image_path) as image:
            return self.predict(image)
