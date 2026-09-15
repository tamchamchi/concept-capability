"""Generate and audit the Phase 3 renderer-validation dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from .colors import build_palette
from .diversity import minimum_recent_distance
from .renderer import DEFAULT_CONFIG_DIR, DEFAULT_RENDERER_CONFIG, load_config, render_sample


METADATA_FIELDS = [
    "sample_id", "split", "concept_id", "color_name", "color_id", "shape_name",
    "shape_id", "support_level", "center_x", "center_y", "size", "rotation",
    "base_rgb_r", "base_rgb_g", "base_rgb_b", "color_jitter_r",
    "color_jitter_g", "color_jitter_b", "final_rgb_r", "final_rgb_g",
    "final_rgb_b", "brightness", "renderer_seed", "image_sha256", "image_path",
    "dataset_version", "minimum_parameter_distance",
]


def _seed_stream(dataset_seed: int, concept_id: str):
    digest = hashlib.sha256(f"{dataset_seed}:{concept_id}".encode()).digest()
    concept_seed = int.from_bytes(digest[:8], "little")
    rng = np.random.default_rng(concept_seed)
    while True:
        yield int(rng.integers(0, np.iinfo(np.int64).max, dtype=np.int64))


def _nearest_color_name(rgb: tuple[int, int, int], palette: dict[str, tuple[int, int, int]]) -> str:
    return min(palette, key=lambda name: math.dist(rgb, palette[name]))


def _make_sheet(
    samples: list[tuple[Image.Image, dict[str, Any]]],
    columns: int,
    scale: int,
    title: str,
) -> Image.Image:
    rows = math.ceil(len(samples) / columns)
    tile_size = samples[0][0].width * scale
    label_height = 15
    sheet = Image.new("RGB", (columns * tile_size, label_height + rows * tile_size), (24, 24, 24))
    draw = ImageDraw.Draw(sheet)
    draw.text((3, 2), title, fill=(255, 255, 255))
    for index, (image, _) in enumerate(samples):
        x = (index % columns) * tile_size
        y = label_height + (index // columns) * tile_size
        sheet.paste(image.resize((tile_size, tile_size), Image.Resampling.NEAREST), (x, y))
    return sheet


def _make_all_concepts_sheet(
    representative_samples: dict[str, tuple[Image.Image, dict[str, Any]]],
    colors: list[str],
    shapes: list[str],
    scale: int = 4,
) -> Image.Image:
    tile_size = 32 * scale
    top_margin, left_margin = 24, 52
    sheet = Image.new(
        "RGB",
        (left_margin + len(shapes) * tile_size, top_margin + len(colors) * tile_size),
        (24, 24, 24),
    )
    draw = ImageDraw.Draw(sheet)
    for column, shape in enumerate(shapes):
        draw.text((left_margin + column * tile_size + 3, 5), shape, fill=(255, 255, 255))
    for row, color in enumerate(colors):
        draw.text((3, top_margin + row * tile_size + 5), color, fill=(255, 255, 255))
        for column, shape in enumerate(shapes):
            image = representative_samples[f"{color}_{shape}"][0]
            enlarged = image.resize((tile_size, tile_size), Image.Resampling.NEAREST)
            sheet.paste(enlarged, (left_margin + column * tile_size, top_margin + row * tile_size))
    return sheet


def generate(
    output_root: Path,
    samples_per_concept: int = 500,
    renderer_config_path: Path = DEFAULT_RENDERER_CONFIG,
) -> dict[str, Any]:
    concepts_config = load_config(DEFAULT_CONFIG_DIR / "concepts.yaml")
    renderer_config = load_config(renderer_config_path)
    support_config = load_config(DEFAULT_CONFIG_DIR / "support_matrix.yaml")
    palette = build_palette(concepts_config)
    diversity = renderer_config["diversity_constraint"]
    threshold = float(diversity["minimum_distance"])
    max_attempts = int(diversity["max_resample_attempts"])

    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {output_root}. "
            "Choose a new path or move the existing validation run first."
        )
    images_root = output_root / "images"
    visualizations_root = output_root / "visualizations"
    images_root.mkdir(parents=True)
    visualizations_root.mkdir(parents=True)

    all_metadata: list[dict[str, Any]] = []
    representative_samples: dict[str, tuple[Image.Image, dict[str, Any]]] = {}
    audit: dict[str, Any] = {
        "samples_per_concept": samples_per_concept,
        "expected_total": len(concepts_config["concepts"]) * samples_per_concept,
        "diversity_threshold": threshold,
        "concepts": {},
    }

    for concept in concepts_config["concepts"]:
        concept_id = concept["id"]
        concept_dir = images_root / concept_id
        concept_dir.mkdir()
        seed_stream = _seed_stream(concepts_config["dataset_seed"], concept_id)
        accepted_metadata: list[dict[str, Any]] = []
        accepted_samples: list[tuple[Image.Image, dict[str, Any]]] = []
        accepted_hashes: set[str] = set()
        attempts = 0
        similarity_rejections = 0
        duplicate_rejections = 0

        while len(accepted_samples) < samples_per_concept:
            attempts += 1
            if attempts > samples_per_concept * max_attempts:
                raise RuntimeError(f"Could not meet diversity threshold for {concept_id}")
            renderer_seed = next(seed_stream)
            image, metadata = render_sample(
                concept["color"], concept["shape"], renderer_seed,
                renderer_config, concepts_config, support_config,
            )
            distance = minimum_recent_distance(
                metadata, accepted_metadata, concept["shape"], renderer_config
            )
            if distance < threshold:
                similarity_rejections += 1
                continue
            if metadata["image_sha256"] in accepted_hashes:
                duplicate_rejections += 1
                continue

            sample_index = len(accepted_samples)
            sample_id = f"{concept_id}_{sample_index:04d}"
            relative_path = Path("images") / concept_id / f"{sample_id}.png"
            image.save(output_root / relative_path)
            metadata.update(
                {
                    "sample_id": sample_id,
                    "split": "renderer_validation",
                    "image_path": relative_path.as_posix(),
                    "minimum_parameter_distance": None if math.isinf(distance) else distance,
                }
            )
            accepted_metadata.append(metadata)
            accepted_samples.append((image, metadata))
            accepted_hashes.add(metadata["image_sha256"])
            all_metadata.append(metadata)

        representative_samples[concept_id] = accepted_samples[samples_per_concept // 2]
        _make_sheet(accepted_samples, columns=10, scale=2, title=concept_id).save(
            visualizations_root / f"{concept_id}_diversity.png"
        )
        distances = [
            item["minimum_parameter_distance"]
            for item in accepted_metadata
            if item["minimum_parameter_distance"] is not None
        ]
        audit["concepts"][concept_id] = {
            "count": len(accepted_samples),
            "candidate_attempts": attempts,
            "rejected_for_similarity": similarity_rejections,
            "rejected_for_duplicate": duplicate_rejections,
            "minimum_accepted_distance": min(distances),
            "center_x_std": statistics.pstdev(item["center_x"] for item in accepted_metadata),
            "center_y_std": statistics.pstdev(item["center_y"] for item in accepted_metadata),
            "size_std": statistics.pstdev(item["size"] for item in accepted_metadata),
            "rotation_std": statistics.pstdev(item["rotation"] for item in accepted_metadata),
        }

    colors = [item["name"] for item in concepts_config["colors"]]
    shapes = [item["name"] for item in concepts_config["shapes"]]
    _make_all_concepts_sheet(representative_samples, colors, shapes).save(
        visualizations_root / "all_concepts_grid.png"
    )

    with (output_root / "metadata.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=METADATA_FIELDS)
        writer.writeheader()
        writer.writerows(all_metadata)

    hashes = [item["image_sha256"] for item in all_metadata]
    duplicate_hashes = [value for value, count in Counter(hashes).items() if count > 1]
    color_mismatches = []
    border_violations = []
    for metadata in all_metadata:
        rgb = tuple(metadata[f"final_rgb_{channel}"] for channel in "rgb")
        if _nearest_color_name(rgb, palette) != metadata["color_name"]:
            color_mismatches.append(metadata["sample_id"])
        image_array = np.asarray(Image.open(output_root / metadata["image_path"]))
        if any(
            int(edge.sum()) != 0
            for edge in (image_array[0], image_array[-1], image_array[:, 0], image_array[:, -1])
        ):
            border_violations.append(metadata["sample_id"])

    audit["actual_total"] = len(all_metadata)
    audit["duplicate_hash_count"] = len(duplicate_hashes)
    audit["color_mismatch_count"] = len(color_mismatches)
    audit["border_violation_count"] = len(border_violations)
    audit["passed"] = (
        audit["actual_total"] == audit["expected_total"]
        and not duplicate_hashes
        and not color_mismatches
        and not border_violations
    )
    with (output_root / "audit.json").open("w", encoding="utf-8") as file:
        json.dump(audit, file, indent=2)
    return audit


def main() -> None:
    storage_config = load_config(DEFAULT_CONFIG_DIR / "storage.yaml")
    default_output = Path(storage_config["data_root"]) / storage_config["directories"]["renderer_validation"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=default_output)
    parser.add_argument("--samples-per-concept", type=int, default=500)
    parser.add_argument(
        "--renderer-config",
        type=Path,
        default=DEFAULT_RENDERER_CONFIG,
        help="Renderer policy to validate (defaults to the V2 base renderer).",
    )
    args = parser.parse_args()
    if args.samples_per_concept < 2:
        raise ValueError("samples-per-concept must be at least 2")

    audit = generate(args.output_root, args.samples_per_concept, args.renderer_config)
    print(json.dumps({key: value for key, value in audit.items() if key != "concepts"}, indent=2))
    if not audit["passed"]:
        raise SystemExit("Renderer validation failed; inspect audit.json")


if __name__ == "__main__":
    main()
