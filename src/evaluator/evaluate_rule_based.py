"""Evaluate the renderer-aware rule baseline on an evaluator split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.dataset.renderer import DEFAULT_CONFIG_DIR, load_config

from .rule_based import RuleBasedEvaluator


def evaluate(metadata_path: Path, data_root: Path) -> dict[str, float | int]:
    rows = pd.read_parquet(metadata_path)
    evaluator = RuleBasedEvaluator()
    color_correct = 0
    shape_correct = 0
    joint_correct = 0
    for row in rows.itertuples(index=False):
        prediction = evaluator.predict_path(data_root / row.image_path)
        color_match = prediction["color_id"] == row.color_id
        shape_match = prediction["shape_id"] == row.shape_id
        color_correct += color_match
        shape_correct += shape_match
        joint_correct += color_match and shape_match
    total = len(rows)
    return {
        "sample_count": total,
        "color_accuracy": color_correct / total,
        "shape_accuracy": shape_correct / total,
        "joint_accuracy": joint_correct / total,
    }


def main() -> None:
    storage = load_config(DEFAULT_CONFIG_DIR / "storage.yaml")
    data_root = Path(storage["data_root"])
    evaluator_root = data_root / storage["directories"]["evaluator"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=data_root)
    parser.add_argument(
        "--metadata",
        type=Path,
        default=evaluator_root / "test" / "metadata.parquet",
    )
    parser.add_argument(
        "--output", type=Path, default=evaluator_root / "rule_based_metrics.json"
    )
    args = parser.parse_args()
    metrics = evaluate(args.metadata, args.data_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
        file.write("\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
