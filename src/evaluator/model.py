"""Small dual-head CNN for independent shape and color predictions."""

from __future__ import annotations

import torch
from torch import nn


class ShapeColorEvaluator(nn.Module):
    def __init__(self, color_classes: int = 6, shape_classes: int = 6) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.shared = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
        )
        self.color_head = nn.Linear(128, color_classes)
        self.shape_head = nn.Linear(128, shape_classes)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        representation = self.shared(self.features(images))
        return self.color_head(representation), self.shape_head(representation)

    @staticmethod
    def target_score(
        color_logits: torch.Tensor,
        shape_logits: torch.Tensor,
        color_ids: torch.Tensor,
        shape_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Return log P(target color) + log P(target shape) for seed ranking."""
        color_log_probability = color_logits.log_softmax(dim=1)
        shape_log_probability = shape_logits.log_softmax(dim=1)
        row_indices = torch.arange(color_logits.shape[0], device=color_logits.device)
        return (
            color_log_probability[row_indices, color_ids]
            + shape_log_probability[row_indices, shape_ids]
        )
