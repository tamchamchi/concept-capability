"""In-memory evaluator dataset without a torchvision dependency."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class EvaluatorDataset(Dataset):
    def __init__(
        self,
        metadata_path: Path,
        data_root: Path,
        augment: bool = False,
        gaussian_noise_std: float = 0.0,
        brightness_jitter: float = 0.0,
    ) -> None:
        self.metadata = pd.read_parquet(metadata_path)
        images = []
        for image_path in self.metadata["image_path"]:
            with Image.open(data_root / str(image_path)) as image:
                images.append(np.asarray(image.convert("RGB"), dtype=np.uint8).copy())
        image_array = np.stack(images)
        self.images = torch.from_numpy(image_array).permute(0, 3, 1, 2).float() / 255.0
        self.color_ids = torch.tensor(
            self.metadata["color_id"].to_numpy(), dtype=torch.long
        )
        self.shape_ids = torch.tensor(
            self.metadata["shape_id"].to_numpy(), dtype=torch.long
        )
        self.augment = augment
        self.gaussian_noise_std = gaussian_noise_std
        self.brightness_jitter = brightness_jitter

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image = self.images[index]
        if self.augment:
            if self.brightness_jitter > 0:
                factor = 1.0 + (
                    2.0 * torch.rand(()) - 1.0
                ) * self.brightness_jitter
                image = image * factor
            if self.gaussian_noise_std > 0:
                image = image + torch.randn_like(image) * self.gaussian_noise_std
            image = image.clamp(0.0, 1.0)
        return image, self.color_ids[index], self.shape_ids[index]
