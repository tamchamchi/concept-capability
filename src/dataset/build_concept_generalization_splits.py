"""Build the six constant-budget concept-generalization training regimes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .exposure_assignment import build_count_plan, validate_balanced_targets
from .renderer import DEFAULT_CONFIG_DIR, load_config


REGIME_NAMES = ("E100", "E075", "E050", "E025", "E005", "E000")
REGIME_METADATA_FIELDS = [
    "sample_id",
    "split",
    "regime",
    "exposure_level",
    "exposure_percent",
    "is_target",
    "concept_train_count",
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


def _regime_config_path(regime: str) -> Path:
    return DEFAULT_CONFIG_DIR / f"concept_generalization_{regime}.yaml"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _shuffle_seed(dataset_seed: int, regime: str) -> int:
    digest = hashlib.sha256(
        f"{dataset_seed}:concept-generalization:{regime}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "little")


def _write_rows(regime_root: Path, rows: pd.DataFrame) -> None:
    rows.to_parquet(regime_root / "metadata.parquet", index=False)
    rows.to_csv(
        regime_root / "metadata.csv",
        index=False,
        columns=REGIME_METADATA_FIELDS,
        quoting=csv.QUOTE_MINIMAL,
    )


def _position_summary(rows: pd.DataFrame) -> dict[str, float]:
    return {
        "center_x_min": float(rows["center_x"].min()),
        "center_x_max": float(rows["center_x"].max()),
        "center_x_mean": float(rows["center_x"].mean()),
        "center_x_std": float(rows["center_x"].std(ddof=0)),
        "center_y_min": float(rows["center_y"].min()),
        "center_y_max": float(rows["center_y"].max()),
        "center_y_mean": float(rows["center_y"].mean()),
        "center_y_std": float(rows["center_y"].std(ddof=0)),
    }


def _validate_regime(
    rows: pd.DataFrame,
    regime_config: Mapping[str, Any],
    concept_specs: Mapping[str, Mapping[str, Any]],
    target_concepts: Sequence[str],
    primitive_min_other_samples: int,
    master_by_id: Mapping[str, tuple[str, str]],
    data_root: Path,
) -> dict[str, Any]:
    regime = str(regime_config["regime"])
    target_set = set(target_concepts)
    concept_ids = list(concept_specs)
    n_full = int(load_config(DEFAULT_CONFIG_DIR / "target_concepts.yaml")["n_full"])
    expected_plan = build_count_plan(
        concept_ids,
        target_concepts,
        n_full,
        float(regime_config["exposure_level"]),
    )
    counts = Counter(str(value) for value in rows["concept_id"])
    expected_total = int(regime_config["total_train_size"])
    expected_target_count = round(
        n_full * float(regime_config["exposure_level"])
    )
    expected_removed = len(target_set) * (n_full - expected_target_count)
    expected_added = expected_removed // (len(concept_ids) - len(target_set))
    redistribution = regime_config["non_target_redistribution"]
    config_valid = (
        int(regime_config["target_count"]) == expected_target_count
        and int(regime_config["non_target_count"]) == n_full + expected_added
        and int(redistribution["removed_target_samples"]) == expected_removed
        and int(redistribution["added_per_non_target_concept"]) == expected_added
        and expected_total == sum(expected_plan.values())
    )

    metadata_errors = []
    missing_master_samples = []
    missing_images = []
    for record in rows.to_dict(orient="records"):
        sample_id = str(record["sample_id"])
        concept_id = str(record["concept_id"])
        master_identity = master_by_id.get(sample_id)
        if master_identity != (concept_id, str(record["image_sha256"])):
            missing_master_samples.append(sample_id)
        if not (data_root / str(record["image_path"])).is_file():
            missing_images.append(sample_id)
        expected_target = concept_id in target_set
        if (
            record["split"] != "generator_train"
            or record["regime"] != regime
            or bool(record["is_target"]) != expected_target
            or int(record["concept_train_count"]) != expected_plan[concept_id]
            or float(record["exposure_level"])
            != float(regime_config["exposure_level"])
        ):
            metadata_errors.append(sample_id)

    primitive_coverage: dict[str, dict[str, int]] = {}
    primitive_coverage_valid = True
    for target_id in target_concepts:
        target = concept_specs[target_id]
        color_other = sum(
            count
            for concept_id, count in counts.items()
            if concept_id != target_id
            and concept_specs[concept_id]["color"] == target["color"]
        )
        shape_other = sum(
            count
            for concept_id, count in counts.items()
            if concept_id != target_id
            and concept_specs[concept_id]["shape"] == target["shape"]
        )
        primitive_coverage[target_id] = {
            "color_in_other_concepts": color_other,
            "shape_in_other_concepts": shape_other,
        }
        primitive_coverage_valid &= (
            color_other >= primitive_min_other_samples
            and shape_other >= primitive_min_other_samples
        )

    position = _position_summary(rows)
    color_totals = Counter()
    shape_totals = Counter()
    for concept_id, count in counts.items():
        color_totals[str(concept_specs[concept_id]["color"])] += count
        shape_totals[str(concept_specs[concept_id]["shape"])] += count
    primitive_marginals_balanced = (
        len(set(color_totals.values())) == 1
        and len(set(shape_totals.values())) == 1
    )
    distribution_valid = (
        10.0 <= position["center_x_min"] <= position["center_x_max"] <= 22.0
        and 10.0 <= position["center_y_min"] <= position["center_y_max"] <= 22.0
        and 15.5 <= position["center_x_mean"] <= 16.5
        and 15.5 <= position["center_y_mean"] <= 16.5
        and 3.0 <= position["center_x_std"] <= 4.1
        and 3.0 <= position["center_y_std"] <= 4.1
        and set(rows["size"].astype(float)) == {8.0}
        and set(rows["rotation"].astype(float)) == {0.0}
        and set(rows["brightness"].astype(float)) == {1.0}
        and all(set(rows[f"color_jitter_{channel}"].astype(int)) == {0} for channel in "rgb")
    )

    audit: dict[str, Any] = {
        "regime": regime,
        "exposure_level": float(regime_config["exposure_level"]),
        "expected_total": expected_total,
        "actual_total": len(rows),
        "target_count": int(regime_config["target_count"]),
        "non_target_count": int(regime_config["non_target_count"]),
        "config_valid": config_valid,
        "concept_counts": dict(sorted(counts.items())),
        "concept_counts_valid": counts == Counter(expected_plan),
        "color_totals": dict(sorted(color_totals.items())),
        "shape_totals": dict(sorted(shape_totals.items())),
        "primitive_marginals_balanced": primitive_marginals_balanced,
        "duplicate_sample_id_count": len(rows) - rows["sample_id"].nunique(),
        "duplicate_image_hash_count": len(rows) - rows["image_sha256"].nunique(),
        "metadata_error_count": len(metadata_errors),
        "missing_master_sample_count": len(missing_master_samples),
        "missing_image_count": len(missing_images),
        "primitive_coverage": primitive_coverage,
        "primitive_coverage_valid": primitive_coverage_valid,
        "renderer_distribution": position,
        "renderer_distribution_valid": distribution_valid,
    }
    audit["passed"] = (
        audit["actual_total"] == expected_total
        and config_valid
        and audit["concept_counts_valid"]
        and primitive_marginals_balanced
        and audit["duplicate_sample_id_count"] == 0
        and audit["duplicate_image_hash_count"] == 0
        and audit["metadata_error_count"] == 0
        and audit["missing_master_sample_count"] == 0
        and audit["missing_image_count"] == 0
        and primitive_coverage_valid
        and distribution_valid
    )
    return audit


def _validate_nested_selection(
    rows_by_regime: Mapping[str, pd.DataFrame], target_concepts: Sequence[str]
) -> list[str]:
    failures = []
    ascending = ("E000", "E005", "E025", "E050", "E075", "E100")
    target_set = set(target_concepts)
    all_concepts = set(rows_by_regime["E100"]["concept_id"])
    for concept_id in sorted(all_concepts):
        selections = {
            regime: set(
                rows.loc[rows["concept_id"] == concept_id, "sample_id"].astype(str)
            )
            for regime, rows in rows_by_regime.items()
        }
        for lower, higher in zip(ascending, ascending[1:]):
            if concept_id in target_set:
                valid = selections[lower] <= selections[higher]
            else:
                valid = selections[higher] <= selections[lower]
            if not valid:
                failures.append(f"{concept_id}:{lower}:{higher}")
    return failures


def validate_exposure_regimes(
    output_root: Path,
    master_root: Path,
) -> dict[str, Any]:
    """Validate all regimes together, including nesting and constant budget."""
    concepts_config = load_config(DEFAULT_CONFIG_DIR / "concepts.yaml")
    target_config = load_config(DEFAULT_CONFIG_DIR / "target_concepts.yaml")
    concept_specs = {
        str(concept["id"]): concept for concept in concepts_config["concepts"]
    }
    targets = list(target_config["target_concepts"])
    validate_balanced_targets(concepts_config["concepts"], targets)
    master = pd.read_parquet(master_root / "metadata.parquet")
    master_by_id = {
        str(row.sample_id): (str(row.concept_id), str(row.image_sha256))
        for row in master.itertuples(index=False)
    }

    regime_audits = {}
    rows_by_regime = {}
    for regime in REGIME_NAMES:
        rows = pd.read_parquet(output_root / regime / "metadata.parquet")
        rows_by_regime[regime] = rows
        regime_config = load_config(_regime_config_path(regime))
        regime_audits[regime] = _validate_regime(
            rows,
            regime_config,
            concept_specs,
            targets,
            int(target_config["primitive_min_other_samples"]),
            master_by_id,
            master_root.parent,
        )

    totals = {regime: len(rows) for regime, rows in rows_by_regime.items()}
    nested_failures = _validate_nested_selection(rows_by_regime, targets)
    audit: dict[str, Any] = {
        "target_concepts": targets,
        "targets_balanced": True,
        "n_full": int(target_config["n_full"]),
        "total_train_size": int(target_config["total_train_size"]),
        "regime_totals": totals,
        "constant_total_train_size": len(set(totals.values())) == 1,
        "nested_selection_failure_count": len(nested_failures),
        "nested_selection_failures": nested_failures,
        "regimes": regime_audits,
    }
    audit["passed"] = (
        audit["constant_total_train_size"]
        and not nested_failures
        and all(item["passed"] for item in regime_audits.values())
    )
    return audit


def build(
    output_root: Path,
    master_root: Path,
) -> dict[str, Any]:
    """Build all configured regimes as metadata-only views of the master pool."""
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {output_root}. "
            "Choose a new path or move existing regimes first."
        )
    concepts_config = load_config(DEFAULT_CONFIG_DIR / "concepts.yaml")
    target_config = load_config(DEFAULT_CONFIG_DIR / "target_concepts.yaml")
    targets = list(target_config["target_concepts"])
    validate_balanced_targets(concepts_config["concepts"], targets)
    concept_ids = [str(concept["id"]) for concept in concepts_config["concepts"]]
    target_set = set(targets)
    master = pd.read_parquet(master_root / "metadata.parquet")
    master = master.sort_values(["concept_id", "sample_id"])
    output_root.mkdir(parents=True)
    for filename, config in (
        ("concepts.json", concepts_config),
        ("target_concepts.json", target_config),
    ):
        with (output_root / filename).open("w", encoding="utf-8") as file:
            json.dump(config, file, indent=2)
            file.write("\n")

    for regime in REGIME_NAMES:
        config = load_config(_regime_config_path(regime))
        count_plan = build_count_plan(
            concept_ids,
            targets,
            int(target_config["n_full"]),
            float(config["exposure_level"]),
        )
        if sum(count_plan.values()) != int(config["total_train_size"]):
            raise ValueError(f"Invalid total budget in {regime}")
        if any(
            count > int((master["concept_id"] == concept_id).sum())
            for concept_id, count in count_plan.items()
        ):
            raise ValueError(f"Master pool is too small for {regime}")

        selected_parts = []
        for concept_id in concept_ids:
            part = master.loc[master["concept_id"] == concept_id].head(
                count_plan[concept_id]
            ).copy()
            part["split"] = "generator_train"
            part["regime"] = regime
            part["exposure_level"] = float(config["exposure_level"])
            part["exposure_percent"] = int(config["exposure_percent"])
            part["is_target"] = concept_id in target_set
            part["concept_train_count"] = count_plan[concept_id]
            part["image_path"] = "master/" + part["image_path"].astype(str)
            selected_parts.append(part)
        selected = pd.concat(selected_parts, ignore_index=True)
        rng = np.random.default_rng(
            _shuffle_seed(int(config["dataset_seed"]), regime)
        )
        selected = selected.iloc[rng.permutation(len(selected))].reset_index(drop=True)
        selected = selected[REGIME_METADATA_FIELDS]

        regime_root = output_root / regime
        regime_root.mkdir()
        _write_rows(regime_root, selected)
        with (regime_root / "config.json").open("w", encoding="utf-8") as file:
            json.dump(config, file, indent=2)
            file.write("\n")
        manifest = {
            "regime": regime,
            "sample_count": len(selected),
            "image_storage": "metadata_only_view_of_master",
            "image_path_base": str(master_root.parent),
            "metadata_csv_sha256": _sha256_file(regime_root / "metadata.csv"),
            "metadata_parquet_sha256": _sha256_file(
                regime_root / "metadata.parquet"
            ),
            "selection_sha256": hashlib.sha256(
                "\n".join(sorted(selected["sample_id"].astype(str))).encode("ascii")
            ).hexdigest(),
        }
        with (regime_root / "manifest.json").open("w", encoding="utf-8") as file:
            json.dump(manifest, file, indent=2)
            file.write("\n")

    audit = validate_exposure_regimes(output_root, master_root)
    with (output_root / "audit.json").open("w", encoding="utf-8") as file:
        json.dump(audit, file, indent=2)
        file.write("\n")
    return audit


def main() -> None:
    storage = load_config(DEFAULT_CONFIG_DIR / "storage.yaml")
    data_root = Path(storage["data_root"])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--master-root",
        type=Path,
        default=data_root / storage["directories"]["master"],
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=data_root / storage["directories"]["concept_generalization"],
    )
    args = parser.parse_args()
    audit = build(args.output_root, args.master_root)
    print(json.dumps(audit, indent=2))
    if not audit["passed"]:
        raise SystemExit("Exposure-regime validation failed; inspect audit.json")


if __name__ == "__main__":
    main()
