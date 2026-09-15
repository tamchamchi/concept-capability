"""Build balanced, leakage-free evaluator splits from the master pool."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .renderer import DEFAULT_CONFIG_DIR, load_config


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _generator_sample_ids(regimes_root: Path) -> set[str]:
    sample_ids: set[str] = set()
    for metadata_path in sorted(regimes_root.glob("E*/metadata.parquet")):
        frame = pd.read_parquet(metadata_path, columns=["sample_id"])
        sample_ids.update(frame["sample_id"].astype(str))
    if not sample_ids:
        raise FileNotFoundError(f"No generator regimes found in {regimes_root}")
    return sample_ids


def validate_evaluator_splits(
    output_root: Path,
    master_root: Path,
    regimes_root: Path,
) -> dict[str, Any]:
    config = load_config(DEFAULT_CONFIG_DIR / "evaluator.yaml")
    concepts = load_config(DEFAULT_CONFIG_DIR / "concepts.yaml")["concepts"]
    expected_concepts = {str(concept["id"]) for concept in concepts}
    generator_ids = _generator_sample_ids(regimes_root)
    split_ids: dict[str, set[str]] = {}
    split_audits: dict[str, Any] = {}

    for split_name, split_config in config["splits"].items():
        rows = pd.read_parquet(output_root / split_name / "metadata.parquet")
        ids = set(rows["sample_id"].astype(str))
        counts = rows.groupby("concept_id").size().to_dict()
        expected_count = int(split_config["samples_per_concept"])
        color_counts = rows.groupby("color_name").size().to_dict()
        shape_counts = rows.groupby("shape_name").size().to_dict()
        missing_images = [
            sample_id
            for sample_id, image_path in zip(rows["sample_id"], rows["image_path"])
            if not (master_root.parent / str(image_path)).is_file()
        ]
        audit = {
            "sample_count": len(rows),
            "samples_per_concept": expected_count,
            "concept_count": rows["concept_id"].nunique(),
            "balanced_concepts": (
                set(counts) == expected_concepts
                and set(counts.values()) == {expected_count}
            ),
            "balanced_colors": len(set(color_counts.values())) == 1,
            "balanced_shapes": len(set(shape_counts.values())) == 1,
            "duplicate_sample_id_count": len(rows) - rows["sample_id"].nunique(),
            "duplicate_image_hash_count": len(rows) - rows["image_sha256"].nunique(),
            "generator_overlap_count": len(ids & generator_ids),
            "missing_image_count": len(missing_images),
        }
        audit["passed"] = (
            audit["sample_count"] == expected_count * len(expected_concepts)
            and audit["balanced_concepts"]
            and audit["balanced_colors"]
            and audit["balanced_shapes"]
            and audit["duplicate_sample_id_count"] == 0
            and audit["duplicate_image_hash_count"] == 0
            and audit["generator_overlap_count"] == 0
            and audit["missing_image_count"] == 0
        )
        split_ids[split_name] = ids
        split_audits[split_name] = audit

    cross_split_overlaps = {
        "train_val": len(split_ids["train"] & split_ids["val"]),
        "train_test": len(split_ids["train"] & split_ids["test"]),
        "val_test": len(split_ids["val"] & split_ids["test"]),
    }
    result: dict[str, Any] = {
        "splits": split_audits,
        "cross_split_overlaps": cross_split_overlaps,
    }
    result["passed"] = (
        all(audit["passed"] for audit in split_audits.values())
        and not any(cross_split_overlaps.values())
    )
    return result


def build(
    output_root: Path,
    master_root: Path,
    regimes_root: Path,
) -> dict[str, Any]:
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {output_root}. "
            "Choose a new path or move the existing evaluator artifacts first."
        )
    config = load_config(DEFAULT_CONFIG_DIR / "evaluator.yaml")
    concepts_config = load_config(DEFAULT_CONFIG_DIR / "concepts.yaml")
    master = pd.read_parquet(master_root / "metadata.parquet")
    master = master.sort_values(["concept_id", "sample_id"])
    output_root.mkdir(parents=True)

    for filename, snapshot in (
        ("evaluator_config.json", config),
        ("concepts.json", concepts_config),
    ):
        with (output_root / filename).open("w", encoding="utf-8") as file:
            json.dump(snapshot, file, indent=2)
            file.write("\n")

    for split_name, split_config in config["splits"].items():
        start, stop = int(split_config["start"]), int(split_config["stop"])
        parts = []
        for _, concept_rows in master.groupby("concept_id", sort=True):
            if len(concept_rows) < stop:
                raise ValueError(
                    f"Master pool needs at least {stop} samples per concept"
                )
            part = concept_rows.iloc[start:stop].copy()
            part["split"] = f"evaluator_{split_name}"
            part["image_path"] = "master/" + part["image_path"].astype(str)
            parts.append(part)
        rows = pd.concat(parts, ignore_index=True)
        rows = rows.sample(
            frac=1.0,
            random_state=int(config["dataset_seed"]) + start,
        ).reset_index(drop=True)
        split_root = output_root / split_name
        split_root.mkdir()
        rows.to_csv(split_root / "metadata.csv", index=False)
        rows.to_parquet(split_root / "metadata.parquet", index=False)
        manifest = {
            "split": f"evaluator_{split_name}",
            "sample_count": len(rows),
            "samples_per_concept": int(split_config["samples_per_concept"]),
            "master_index_range": [start, stop],
            "image_storage": "metadata_only_view_of_master",
            "metadata_csv_sha256": _sha256_file(split_root / "metadata.csv"),
            "metadata_parquet_sha256": _sha256_file(
                split_root / "metadata.parquet"
            ),
        }
        with (split_root / "manifest.json").open("w", encoding="utf-8") as file:
            json.dump(manifest, file, indent=2)
            file.write("\n")

    audit = validate_evaluator_splits(output_root, master_root, regimes_root)
    with (output_root / "split_audit.json").open("w", encoding="utf-8") as file:
        json.dump(audit, file, indent=2)
        file.write("\n")
    return audit


def main() -> None:
    storage = load_config(DEFAULT_CONFIG_DIR / "storage.yaml")
    data_root = Path(storage["data_root"])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--master-root", type=Path,
        default=data_root / storage["directories"]["master"],
    )
    parser.add_argument(
        "--regimes-root", type=Path,
        default=data_root / storage["directories"]["concept_generalization"],
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=data_root / storage["directories"]["evaluator"],
    )
    args = parser.parse_args()
    audit = build(args.output_root, args.master_root, args.regimes_root)
    print(json.dumps(audit, indent=2))
    if not audit["passed"]:
        raise SystemExit("Evaluator split validation failed")


if __name__ == "__main__":
    main()
