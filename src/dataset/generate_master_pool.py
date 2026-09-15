"""Generate and validate the exposure-independent concept master pool."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from .diversity import minimum_recent_distance
from .renderer import (
    DEFAULT_CONFIG_DIR,
    DEFAULT_RENDERER_CONFIG,
    load_config,
    render_sample,
)


MASTER_METADATA_FIELDS = [
    "sample_id",
    "split",
    "concept_id",
    "color_name",
    "color_id",
    "shape_name",
    "shape_id",
    "center_x",
    "center_y",
    "size",
    "rotation",
    "base_rgb_r",
    "base_rgb_g",
    "base_rgb_b",
    "color_jitter_r",
    "color_jitter_g",
    "color_jitter_b",
    "final_rgb_r",
    "final_rgb_g",
    "final_rgb_b",
    "brightness",
    "renderer_seed",
    "image_sha256",
    "image_path",
    "dataset_version",
    "minimum_parameter_distance",
]


def _seed_stream(dataset_seed: int, concept_id: str) -> Iterator[int]:
    """Yield deterministic seeds in a namespace distinct from validation data."""
    digest = hashlib.sha256(
        f"{dataset_seed}:master:{concept_id}".encode("utf-8")
    ).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    while True:
        yield int(rng.integers(0, np.iinfo(np.int64).max, dtype=np.int64))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_metadata(output_root: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    csv_path = output_root / "metadata.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=MASTER_METADATA_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    dataframe = pd.DataFrame(rows, columns=MASTER_METADATA_FIELDS)
    dataframe.to_parquet(output_root / "metadata.parquet", index=False)


def _write_config_snapshot(
    output_root: Path,
    concepts_config: Mapping[str, Any],
    renderer_config: Mapping[str, Any],
) -> None:
    config_root = output_root / "configs"
    config_root.mkdir()
    for filename, config in (
        ("concepts.json", concepts_config),
        ("renderer_base.json", renderer_config),
    ):
        with (config_root / filename).open("w", encoding="utf-8") as file:
            json.dump(config, file, indent=2)
            file.write("\n")


def _load_metadata(metadata_path: Path) -> list[dict[str, Any]]:
    if metadata_path.suffix == ".parquet":
        return pd.read_parquet(metadata_path).to_dict(orient="records")
    with metadata_path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def validate_master_pool(
    output_root: Path,
    samples_per_concept: int = 500,
) -> dict[str, Any]:
    """Validate counts, hashes, pixels, semantics, and the frozen distribution."""
    concepts_config = load_config(output_root / "configs" / "concepts.json")
    renderer_config = load_config(output_root / "configs" / "renderer_base.json")
    rows = _load_metadata(output_root / "metadata.parquet")
    concepts = {item["id"]: item for item in concepts_config["concepts"]}
    palette = {
        item["name"]: tuple(item["rgb"]) for item in concepts_config["colors"]
    }
    expected_total = len(concepts) * samples_per_concept

    concept_counts = Counter(str(row["concept_id"]) for row in rows)
    image_hashes = [str(row["image_sha256"]) for row in rows]
    renderer_seeds = [int(row["renderer_seed"]) for row in rows]
    sample_ids = [str(row["sample_id"]) for row in rows]
    failures: dict[str, list[str]] = {
        "missing_images": [],
        "pixel_hash_mismatches": [],
        "invalid_images": [],
        "border_violations": [],
        "semantic_mismatches": [],
        "parameter_violations": [],
    }

    x_values: list[float] = []
    y_values: list[float] = []
    per_concept_positions: dict[str, tuple[list[float], list[float]]] = {
        concept_id: ([], []) for concept_id in concepts
    }
    position = renderer_config["instance_distribution"]["position"]
    x_min, x_max = position["center_x"]["min"], position["center_x"]["max"]
    y_min, y_max = position["center_y"]["min"], position["center_y"]["max"]

    for row in rows:
        sample_id = str(row["sample_id"])
        concept_id = str(row["concept_id"])
        image_path = output_root / str(row["image_path"])
        if not image_path.is_file():
            failures["missing_images"].append(sample_id)
            continue

        try:
            with Image.open(image_path) as image_file:
                image = image_file.convert("RGB")
                pixels = np.asarray(image)
        except Exception:
            failures["invalid_images"].append(sample_id)
            continue

        if image.size != (32, 32) or image_file.mode != "RGB":
            failures["invalid_images"].append(sample_id)
        if hashlib.sha256(pixels.tobytes()).hexdigest() != row["image_sha256"]:
            failures["pixel_hash_mismatches"].append(sample_id)
        if any(
            int(edge.sum()) != 0
            for edge in (pixels[0], pixels[-1], pixels[:, 0], pixels[:, -1])
        ):
            failures["border_violations"].append(sample_id)

        concept = concepts.get(concept_id)
        expected_rgb = palette.get(str(row["color_name"]))
        actual_rgb = tuple(int(row[f"final_rgb_{channel}"]) for channel in "rgb")
        if (
            concept is None
            or row["split"] != "master"
            or concept["color"] != row["color_name"]
            or concept["shape"] != row["shape_name"]
            or actual_rgb != expected_rgb
        ):
            failures["semantic_mismatches"].append(sample_id)

        center_x, center_y = float(row["center_x"]), float(row["center_y"])
        x_values.append(center_x)
        y_values.append(center_y)
        if concept_id in per_concept_positions:
            per_concept_positions[concept_id][0].append(center_x)
            per_concept_positions[concept_id][1].append(center_y)
        if not (
            x_min <= center_x <= x_max
            and y_min <= center_y <= y_max
            and float(row["size"]) == 8.0
            and float(row["rotation"]) == 0.0
            and float(row["brightness"]) == 1.0
            and all(int(row[f"color_jitter_{channel}"]) == 0 for channel in "rgb")
        ):
            failures["parameter_violations"].append(sample_id)

    distribution_failures = []
    for concept_id, (concept_x, concept_y) in per_concept_positions.items():
        if len(concept_x) < 100:
            continue
        if not (
            15.25 <= statistics.mean(concept_x) <= 16.75
            and 15.25 <= statistics.mean(concept_y) <= 16.75
            and 2.75 <= statistics.pstdev(concept_x) <= 4.10
            and 2.75 <= statistics.pstdev(concept_y) <= 4.10
        ):
            distribution_failures.append(concept_id)

    expected_counts = {concept_id: samples_per_concept for concept_id in concepts}
    summary: dict[str, Any] = {
        "samples_per_concept": samples_per_concept,
        "expected_total": expected_total,
        "actual_total": len(rows),
        "concept_count": len(concept_counts),
        "concept_counts_valid": dict(concept_counts) == expected_counts,
        "duplicate_sample_id_count": len(sample_ids) - len(set(sample_ids)),
        "duplicate_renderer_seed_count": len(renderer_seeds) - len(set(renderer_seeds)),
        "duplicate_image_hash_count": len(image_hashes) - len(set(image_hashes)),
        "position_summary": {
            "center_x_min": min(x_values) if x_values else None,
            "center_x_max": max(x_values) if x_values else None,
            "center_x_mean": statistics.mean(x_values) if x_values else None,
            "center_x_std": statistics.pstdev(x_values) if x_values else None,
            "center_y_min": min(y_values) if y_values else None,
            "center_y_max": max(y_values) if y_values else None,
            "center_y_mean": statistics.mean(y_values) if y_values else None,
            "center_y_std": statistics.pstdev(y_values) if y_values else None,
        },
        "distribution_failure_concepts": distribution_failures,
        "failures": {name: values[:100] for name, values in failures.items()},
        "failure_counts": {name: len(values) for name, values in failures.items()},
    }
    summary["passed"] = (
        summary["actual_total"] == expected_total
        and summary["concept_count"] == len(concepts)
        and summary["concept_counts_valid"]
        and summary["duplicate_sample_id_count"] == 0
        and summary["duplicate_renderer_seed_count"] == 0
        and summary["duplicate_image_hash_count"] == 0
        and not distribution_failures
        and not any(summary["failure_counts"].values())
    )
    return summary


def generate(
    output_root: Path,
    samples_per_concept: int = 500,
    renderer_config_path: Path = DEFAULT_RENDERER_CONFIG,
) -> dict[str, Any]:
    """Generate the deterministic master pool and return its validation audit."""
    if samples_per_concept < 2:
        raise ValueError("samples_per_concept must be at least 2")
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {output_root}. "
            "Choose a new path or move the existing master pool first."
        )

    concepts_config = load_config(DEFAULT_CONFIG_DIR / "concepts.yaml")
    renderer_config = load_config(renderer_config_path)
    diversity = renderer_config["diversity_constraint"]
    threshold = float(diversity["minimum_distance"])
    max_attempts = int(diversity["max_resample_attempts"])
    images_root = output_root / "images"
    images_root.mkdir(parents=True)
    _write_config_snapshot(output_root, concepts_config, renderer_config)

    all_rows: list[dict[str, Any]] = []
    global_hashes: set[str] = set()
    global_seeds: set[int] = set()
    generation_stats: dict[str, Any] = {}

    for concept in concepts_config["concepts"]:
        concept_id = concept["id"]
        concept_root = images_root / concept_id
        concept_root.mkdir()
        seeds = _seed_stream(concepts_config["dataset_seed"], concept_id)
        accepted: list[dict[str, Any]] = []
        similarity_rejections = 0
        duplicate_rejections = 0
        seed_collision_rejections = 0
        attempts = 0

        while len(accepted) < samples_per_concept:
            attempts += 1
            if attempts > samples_per_concept * max_attempts:
                raise RuntimeError(f"Could not fill master pool for {concept_id}")
            renderer_seed = next(seeds)
            if renderer_seed in global_seeds:
                seed_collision_rejections += 1
                continue
            image, metadata = render_sample(
                concept["color"],
                concept["shape"],
                renderer_seed,
                config=renderer_config,
                concepts_config=concepts_config,
            )
            distance = minimum_recent_distance(
                metadata, accepted, concept["shape"], renderer_config
            )
            if distance < threshold:
                similarity_rejections += 1
                continue
            if metadata["image_sha256"] in global_hashes:
                duplicate_rejections += 1
                continue

            sample_index = len(accepted)
            sample_id = f"master_{concept_id}_{sample_index:04d}"
            relative_path = Path("images") / concept_id / f"{sample_id}.png"
            image.save(output_root / relative_path)
            metadata.pop("support_level", None)
            metadata.update(
                {
                    "sample_id": sample_id,
                    "split": "master",
                    "image_path": relative_path.as_posix(),
                    "minimum_parameter_distance": (
                        None if math.isinf(distance) else distance
                    ),
                }
            )
            accepted.append(metadata)
            all_rows.append(metadata)
            global_hashes.add(metadata["image_sha256"])
            global_seeds.add(renderer_seed)

        generation_stats[concept_id] = {
            "count": len(accepted),
            "candidate_attempts": attempts,
            "rejected_for_similarity": similarity_rejections,
            "rejected_for_duplicate": duplicate_rejections,
            "rejected_for_seed_collision": seed_collision_rejections,
        }

    _write_metadata(output_root, all_rows)
    audit = validate_master_pool(output_root, samples_per_concept)
    audit["generation"] = generation_stats
    with (output_root / "audit.json").open("w", encoding="utf-8") as file:
        json.dump(audit, file, indent=2)
        file.write("\n")

    manifest = {
        "dataset_name": concepts_config["dataset_name"],
        "dataset_version": concepts_config["dataset_version"],
        "split": "master",
        "exposure_independent": True,
        "samples_per_concept": samples_per_concept,
        "sample_count": len(all_rows),
        "concept_count": len(concepts_config["concepts"]),
        "dataset_seed": concepts_config["dataset_seed"],
        "renderer_profile": renderer_config["profile"],
        "combined_pixel_sha256": hashlib.sha256(
            "\n".join(row["image_sha256"] for row in all_rows).encode("ascii")
        ).hexdigest(),
        "metadata_csv_sha256": _sha256_file(output_root / "metadata.csv"),
        "metadata_parquet_sha256": _sha256_file(output_root / "metadata.parquet"),
        "validation_passed": audit["passed"],
    }
    with (output_root / "manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)
        file.write("\n")
    return audit


def main() -> None:
    storage_config = load_config(DEFAULT_CONFIG_DIR / "storage.yaml")
    default_output = (
        Path(storage_config["data_root"])
        / storage_config["directories"]["master"]
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=default_output)
    parser.add_argument("--samples-per-concept", type=int, default=500)
    parser.add_argument(
        "--renderer-config", type=Path, default=DEFAULT_RENDERER_CONFIG
    )
    args = parser.parse_args()
    audit = generate(
        args.output_root,
        samples_per_concept=args.samples_per_concept,
        renderer_config_path=args.renderer_config,
    )
    print(
        json.dumps(
            {key: value for key, value in audit.items() if key != "generation"},
            indent=2,
        )
    )
    if not audit["passed"]:
        raise SystemExit("Master-pool validation failed; inspect audit.json")


if __name__ == "__main__":
    main()
