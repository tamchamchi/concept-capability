#!/usr/bin/env python3
"""Run data generation, evaluator training, and validation end to end."""

from __future__ import annotations

import argparse
import json
import sys
import tarfile
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset.build_concept_generalization_splits import (  # noqa: E402
    build as build_regimes,
    validate_exposure_regimes,
)
from src.dataset.build_evaluator_splits import (  # noqa: E402
    build as build_evaluator_splits,
    validate_evaluator_splits,
)
from src.dataset.generate_master_pool import (  # noqa: E402
    generate as generate_master,
    validate_master_pool,
)
from src.dataset.renderer import DEFAULT_CONFIG_DIR  # noqa: E402
from src.evaluator.evaluate_rule_based import evaluate as evaluate_rule_based  # noqa: E402
from src.evaluator.train import train as train_evaluator  # noqa: E402


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _require_accelerator(device: str) -> None:
    if device != "cuda":
        return
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is unavailable. In Colab choose "
            "Runtime > Change runtime type > T4 GPU, then run again."
        )


def _archive_data(data_root: Path, archive_path: Path) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(data_root, arcname=data_root.name)


def run(args: argparse.Namespace) -> dict[str, Any]:
    data_root = args.data_root.resolve()
    master_root = data_root / "master"
    regimes_root = data_root / "concept_generalization"
    evaluator_root = data_root / "evaluator"
    model_root = evaluator_root / "model"
    report: dict[str, Any] = {
        "data_root": str(data_root),
        "device_requested": args.device,
        "steps": {},
    }

    if (master_root / "metadata.parquet").is_file():
        master_audit = validate_master_pool(master_root, samples_per_concept=500)
        master_action = "reused"
    else:
        master_audit = generate_master(master_root, samples_per_concept=500)
        master_action = "generated"
    if not master_audit["passed"]:
        raise RuntimeError("Master-pool validation failed")
    report["steps"]["master"] = {
        "action": master_action,
        "passed": True,
        "sample_count": master_audit["actual_total"],
    }

    if all(
        (regimes_root / regime / "metadata.parquet").is_file()
        for regime in ("E100", "E075", "E050", "E025", "E005", "E000")
    ):
        regime_audit = validate_exposure_regimes(regimes_root, master_root)
        regime_action = "reused"
    else:
        regime_audit = build_regimes(regimes_root, master_root)
        regime_action = "generated"
    if not regime_audit["passed"]:
        raise RuntimeError("Exposure-regime validation failed")
    report["steps"]["exposure_regimes"] = {
        "action": regime_action,
        "passed": True,
        "regime_totals": regime_audit["regime_totals"],
    }

    if all(
        (evaluator_root / split / "metadata.parquet").is_file()
        for split in ("train", "val", "test")
    ):
        split_audit = validate_evaluator_splits(
            evaluator_root, master_root, regimes_root
        )
        split_action = "reused"
    else:
        split_audit = build_evaluator_splits(
            evaluator_root, master_root, regimes_root
        )
        split_action = "generated"
    if not split_audit["passed"]:
        raise RuntimeError("Evaluator split validation failed")
    report["steps"]["evaluator_splits"] = {
        "action": split_action,
        "passed": True,
        "split_counts": {
            name: audit["sample_count"]
            for name, audit in split_audit["splits"].items()
        },
    }

    rule_metrics = evaluate_rule_based(
        evaluator_root / "test" / "metadata.parquet", data_root
    )
    with (evaluator_root / "rule_based_metrics.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(rule_metrics, file, indent=2)
        file.write("\n")
    report["steps"]["rule_based_evaluator"] = rule_metrics

    if not args.validate_only:
        _require_accelerator(args.device)
        metrics_path = model_root / "metrics.json"
        if metrics_path.is_file() and not args.retrain:
            learned_metrics = _load_json(metrics_path)
            training_action = "reused"
        else:
            learned_metrics = train_evaluator(
                evaluator_root=evaluator_root,
                data_root=data_root,
                output_root=model_root,
                config_path=DEFAULT_CONFIG_DIR / "evaluator.yaml",
                device_name=args.device,
                epochs=args.epochs,
                batch_size=args.batch_size,
            )
            training_action = "trained"
        if not learned_metrics["accepted"]:
            raise RuntimeError("Learned evaluator failed its acceptance thresholds")
        report["steps"]["learned_evaluator"] = {
            "action": training_action,
            **learned_metrics,
        }
    else:
        report["steps"]["learned_evaluator"] = {"action": "skipped"}

    report["passed"] = True
    report_path = args.report or evaluator_root / "pipeline_report.json"
    if args.archive is not None:
        report["archive"] = str(args.archive)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
        file.write("\n")
    if args.archive is not None:
        _archive_data(data_root, args.archive)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "colab_evaluator_data",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Create/reuse data and run audits without CNN training.",
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Train again even when accepted metrics already exist.",
    )
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--archive",
        type=Path,
        help="Optional .tar.gz destination, such as a mounted Google Drive path.",
    )
    args = parser.parse_args()
    report = run(args)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
