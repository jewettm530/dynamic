"""Shared paired ED/ES dataset for controlled Stage 1 experiments.

This dataset intentionally reproduces the exact sample definition used by the
Stage 1 multi-task weighting experiments. All controlled Step 3 models therefore
receive the same videos, labeled ED/ES frames, masks, and EF targets.
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset

from echonet.datasets.echo import Echo


class EchoStage1PairedDataset(Dataset):
    """Return paired ED/ES frames, LV masks, and one continuous EF target."""

    def __init__(
        self,
        root: str,
        split: str,
        mean: float = 0.0,
        std: float = 1.0,
    ) -> None:
        self.dataset = Echo(
            root=root,
            split=split,
            target_type=[
                "Filename",
                "LargeFrame",
                "SmallFrame",
                "LargeTrace",
                "SmallTrace",
                "EF",
            ],
            mean=mean,
            std=std,
            length=16,
            period=2,
            clips=1,
            pad=None,
            noise=None,
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        _, targets = self.dataset[index]

        (
            filename,
            ed_frame,
            es_frame,
            ed_mask,
            es_mask,
            ef,
        ) = targets

        ed_frame = torch.as_tensor(
            ed_frame,
            dtype=torch.float32,
        )

        es_frame = torch.as_tensor(
            es_frame,
            dtype=torch.float32,
        )

        ed_mask = torch.as_tensor(
            ed_mask,
            dtype=torch.float32,
        ).unsqueeze(0)

        es_mask = torch.as_tensor(
            es_mask,
            dtype=torch.float32,
        ).unsqueeze(0)

        return {
            "filename": filename,
            "ed_image": ed_frame,
            "es_image": es_frame,
            "ed_mask": (ed_mask > 0.5).float(),
            "es_mask": (es_mask > 0.5).float(),
            "ef": torch.tensor(
                float(ef) / 100.0,
                dtype=torch.float32,
            ),
        }
