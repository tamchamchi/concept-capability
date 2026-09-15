"""Train and evaluate the dual-head Shape × Color classifier."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.dataset.renderer import DEFAULT_CONFIG_DIR, load_config

from .data import EvaluatorDataset
from .model import ShapeColorEvaluator


def _write_history(output_root: Path, history: list[dict[str, Any]]) -> None:
    with (output_root / "history.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def _write_training_plot(
    output_root: Path,
    history: list[dict[str, Any]],
    joint_threshold: float,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/concept-capability-matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    epochs = [row["epoch"] for row in history]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in history], marker="o")
    axes[0].set(title="Evaluator training loss", xlabel="Epoch", ylabel="Loss")
    axes[0].grid(alpha=0.25)
    for field, label in (
        ("val_color_accuracy", "Color"),
        ("val_shape_accuracy", "Shape"),
        ("val_joint_accuracy", "Joint"),
    ):
        axes[1].plot(epochs, [row[field] for row in history], marker="o", label=label)
    axes[1].axhline(
        joint_threshold, color="black", linestyle="--", linewidth=1, label="Threshold"
    )
    axes[1].set(
        title="Validation accuracy",
        xlabel="Epoch",
        ylabel="Accuracy",
        ylim=(0.0, 1.02),
    )
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(output_root / "training_curves.png", dpi=160)
    plt.close(figure)


def _set_seed(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


@torch.no_grad()
def evaluate(
    model: ShapeColorEvaluator,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    color_correct = 0
    shape_correct = 0
    joint_correct = 0
    total = 0
    color_confusion = torch.zeros((6, 6), dtype=torch.int64)
    shape_confusion = torch.zeros((6, 6), dtype=torch.int64)
    for images, color_ids, shape_ids in loader:
        images = images.to(device)
        color_ids = color_ids.to(device)
        shape_ids = shape_ids.to(device)
        color_logits, shape_logits = model(images)
        color_predictions = color_logits.argmax(dim=1)
        shape_predictions = shape_logits.argmax(dim=1)
        color_matches = color_predictions == color_ids
        shape_matches = shape_predictions == shape_ids
        color_correct += int(color_matches.sum())
        shape_correct += int(shape_matches.sum())
        joint_correct += int((color_matches & shape_matches).sum())
        total += len(images)
        for truth, prediction in zip(color_ids.cpu(), color_predictions.cpu()):
            color_confusion[int(truth), int(prediction)] += 1
        for truth, prediction in zip(shape_ids.cpu(), shape_predictions.cpu()):
            shape_confusion[int(truth), int(prediction)] += 1
    return {
        "sample_count": total,
        "color_accuracy": color_correct / total,
        "shape_accuracy": shape_correct / total,
        "joint_accuracy": joint_correct / total,
        "color_confusion_matrix": color_confusion.tolist(),
        "shape_confusion_matrix": shape_confusion.tolist(),
    }


def train(
    evaluator_root: Path,
    data_root: Path,
    output_root: Path,
    config_path: Path,
    device_name: str = "auto",
    epochs: int | None = None,
    batch_size: int | None = None,
) -> dict[str, Any]:
    config = copy.deepcopy(load_config(config_path))
    training = config["training"]
    if epochs is not None:
        training["epochs"] = epochs
    if batch_size is not None:
        training["batch_size"] = batch_size
    seed = int(training["seed"])
    _set_seed(seed)
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    if device.type == "cpu":
        torch.set_num_threads(min(8, torch.get_num_threads()))
    output_root.mkdir(parents=True, exist_ok=True)
    concepts = load_config(DEFAULT_CONFIG_DIR / "concepts.yaml")
    color_names = [item["name"] for item in concepts["colors"]]
    shape_names = [item["name"] for item in concepts["shapes"]]

    train_dataset = EvaluatorDataset(
        evaluator_root / "train" / "metadata.parquet",
        data_root,
        augment=True,
        gaussian_noise_std=float(training["gaussian_noise_std"]),
        brightness_jitter=float(training["brightness_jitter"]),
    )
    val_dataset = EvaluatorDataset(
        evaluator_root / "val" / "metadata.parquet", data_root
    )
    test_dataset = EvaluatorDataset(
        evaluator_root / "test" / "metadata.parquet", data_root
    )
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(training["batch_size"]),
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(training["batch_size"]),
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=int(training["batch_size"]),
        shuffle=False,
        num_workers=0,
    )

    model = ShapeColorEvaluator().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    criterion = nn.CrossEntropyLoss()
    history = []
    best_joint_accuracy = -1.0
    best_state = None
    accepted_epoch_streak = 0

    for epoch in range(1, int(training["epochs"]) + 1):
        model.train()
        total_loss = 0.0
        sample_count = 0
        for images, color_ids, shape_ids in train_loader:
            images = images.to(device)
            color_ids = color_ids.to(device)
            shape_ids = shape_ids.to(device)
            optimizer.zero_grad(set_to_none=True)
            color_logits, shape_logits = model(images)
            loss = criterion(color_logits, color_ids) + criterion(
                shape_logits, shape_ids
            )
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(images)
            sample_count += len(images)

        validation = evaluate(model, val_loader, device)
        epoch_result = {
            "epoch": epoch,
            "train_loss": total_loss / sample_count,
            "val_color_accuracy": validation["color_accuracy"],
            "val_shape_accuracy": validation["shape_accuracy"],
            "val_joint_accuracy": validation["joint_accuracy"],
        }
        history.append(epoch_result)
        print(json.dumps(epoch_result), flush=True)
        if validation["joint_accuracy"] > best_joint_accuracy:
            best_joint_accuracy = validation["joint_accuracy"]
            best_state = copy.deepcopy(model.state_dict())
            torch.save(
                {
                    "model_state_dict": best_state,
                    "model_config": config["model"],
                    "color_names": color_names,
                    "shape_names": shape_names,
                    "best_epoch": epoch,
                    "best_val_joint_accuracy": best_joint_accuracy,
                },
                output_root / "evaluator.pt",
            )
        _write_history(output_root, history)
        _write_training_plot(
            output_root,
            history,
            float(config["acceptance"]["minimum_joint_accuracy"]),
        )
        if validation["joint_accuracy"] >= float(
            config["acceptance"]["minimum_joint_accuracy"]
        ):
            accepted_epoch_streak += 1
        else:
            accepted_epoch_streak = 0
        if (
            epoch >= int(training["minimum_epochs"])
            and accepted_epoch_streak >= int(training["early_stop_patience"])
        ):
            break

    assert best_state is not None
    model.load_state_dict(best_state)
    validation = evaluate(model, val_loader, device)
    test = evaluate(model, test_loader, device)
    acceptance = config["acceptance"]
    accepted = (
        test["color_accuracy"] >= float(acceptance["minimum_color_accuracy"])
        and test["shape_accuracy"] >= float(acceptance["minimum_shape_accuracy"])
        and test["joint_accuracy"] >= float(acceptance["minimum_joint_accuracy"])
    )
    results = {
        "model": config["name"],
        "device": str(device),
        "seed": seed,
        "best_epoch": max(history, key=lambda row: row["val_joint_accuracy"])["epoch"],
        "validation": validation,
        "test": test,
        "acceptance": acceptance,
        "accepted": accepted,
    }

    torch.save(
        {
            "model_state_dict": best_state,
            "model_config": config["model"],
            "color_names": color_names,
            "shape_names": shape_names,
            "best_epoch": results["best_epoch"],
            "best_val_joint_accuracy": validation["joint_accuracy"],
        },
        output_root / "evaluator.pt",
    )
    with (output_root / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)
        file.write("\n")
    _write_history(output_root, history)
    _write_training_plot(
        output_root,
        history,
        float(config["acceptance"]["minimum_joint_accuracy"]),
    )
    return results


def main() -> None:
    storage = load_config(DEFAULT_CONFIG_DIR / "storage.yaml")
    data_root = Path(storage["data_root"])
    evaluator_root = data_root / storage["directories"]["evaluator"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=data_root)
    parser.add_argument("--evaluator-root", type=Path, default=evaluator_root)
    parser.add_argument(
        "--output-root", type=Path, default=evaluator_root / "model"
    )
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG_DIR / "evaluator.yaml"
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    args = parser.parse_args()
    results = train(
        args.evaluator_root,
        args.data_root,
        args.output_root,
        args.config,
        args.device,
        args.epochs,
        args.batch_size,
    )
    print(json.dumps(results, indent=2))
    if not results["accepted"]:
        raise SystemExit("Evaluator did not meet configured acceptance thresholds")


if __name__ == "__main__":
    main()
