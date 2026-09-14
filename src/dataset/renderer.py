"""Deterministic renderer for the Shapes32-Recoverability dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from PIL import Image

from .colors import build_palette
from .sampler import sample_instance_parameters
from .shapes import SUPPORTED_SHAPES, draw_shape_mask


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_DIR = PROJECT_ROOT / "dataset" / "configs"


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a JSON-compatible YAML configuration file."""
    with Path(path).open(encoding="utf-8") as config_file:
        return json.load(config_file)


def _resolve_config(
    config: Mapping[str, Any] | str | Path | None,
    default_filename: str,
) -> Mapping[str, Any]:
    if config is None:
        return load_config(DEFAULT_CONFIG_DIR / default_filename)
    if isinstance(config, (str, Path)):
        return load_config(config)
    return config


def _support_level(
    color_name: str,
    shape_name: str,
    support_config: Mapping[str, Any],
) -> int:
    shape_index = support_config["shape_order"].index(shape_name)
    return int(support_config["matrix"][color_name][shape_index])


def render_sample(
    color_name: str,
    shape_name: str,
    renderer_seed: int,
    config: Mapping[str, Any] | str | Path | None = None,
    concepts_config: Mapping[str, Any] | str | Path | None = None,
    support_config: Mapping[str, Any] | str | Path | None = None,
) -> tuple[Image.Image, dict[str, Any]]:
    """Render one RGB sample and return its complete renderer metadata.

    Identical arguments and configuration always yield identical pixels and metadata.
    """
    renderer_config = _resolve_config(config, "renderer.yaml")
    concept_spec = _resolve_config(concepts_config, "concepts.yaml")
    support_spec = _resolve_config(support_config, "support_matrix.yaml")

    palette = build_palette(concept_spec)
    if color_name not in palette:
        raise ValueError(f"Unknown color: {color_name!r}")
    if shape_name not in SUPPORTED_SHAPES:
        raise ValueError(f"Unknown shape: {shape_name!r}")

    shape_specs = {item["name"]: item for item in concept_spec["shapes"]}
    color_specs = {item["name"]: item for item in concept_spec["colors"]}
    if shape_name not in shape_specs:
        raise ValueError(f"Shape {shape_name!r} is absent from concepts config")

    base_rgb = palette[color_name]
    parameters = sample_instance_parameters(
        shape_name,
        base_rgb,
        renderer_seed,
        renderer_config,
    )
    mask = draw_shape_mask(
        shape_name=shape_name,
        image_size=int(renderer_config["image_size"]),
        center=(parameters["center_x"], parameters["center_y"]),
        size=parameters["size"],
        rotation_degrees=parameters["rotation"],
        supersampling_factor=int(renderer_config["supersampling_factor"]),
        downsampling_filter=renderer_config["downsampling_filter"],
        canvas_padding=float(
            renderer_config["instance_distribution"]["position"]["canvas_padding"]
        ),
    )

    image_size = int(renderer_config["image_size"])
    background_rgb = tuple(int(value) for value in renderer_config["background_rgb"])
    background = Image.new("RGB", (image_size, image_size), background_rgb)
    foreground = Image.new("RGB", (image_size, image_size), parameters["final_rgb"])
    image = Image.composite(foreground, background, mask)
    image_hash = hashlib.sha256(image.tobytes()).hexdigest()

    metadata = {
        "concept_id": f"{color_name}_{shape_name}",
        "color_name": color_name,
        "color_id": int(color_specs[color_name]["id"]),
        "shape_name": shape_name,
        "shape_id": int(shape_specs[shape_name]["id"]),
        "support_level": _support_level(color_name, shape_name, support_spec),
        "center_x": parameters["center_x"],
        "center_y": parameters["center_y"],
        "size": parameters["size"],
        "rotation": parameters["rotation"],
        "base_rgb_r": base_rgb[0],
        "base_rgb_g": base_rgb[1],
        "base_rgb_b": base_rgb[2],
        "color_jitter_r": parameters["color_jitter_r"],
        "color_jitter_g": parameters["color_jitter_g"],
        "color_jitter_b": parameters["color_jitter_b"],
        "final_rgb_r": parameters["final_rgb"][0],
        "final_rgb_g": parameters["final_rgb"][1],
        "final_rgb_b": parameters["final_rgb"][2],
        "brightness": parameters["brightness"],
        "renderer_seed": int(renderer_seed),
        "image_sha256": image_hash,
        "dataset_version": concept_spec["dataset_version"],
    }
    return image, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Render one synthetic concept sample")
    parser.add_argument("color")
    parser.add_argument("shape")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    image, metadata = render_sample(args.color, args.shape, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
