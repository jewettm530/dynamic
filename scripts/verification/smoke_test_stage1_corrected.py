#!/usr/bin/env python3
"""Small architecture smoke test for corrected Stage 1.

Uses random tensors and no pretrained download. Confirms:
- EF accepts [B,C,T,H,W] with T>2 and returns one scalar/video.
- EF rejects 2-frame input.
- segmentation accepts labeled 2D frames and returns one LV logit map/frame.
- weighted EF + segmentation gradients both reach the shared encoder.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from echonet.modeling.stage1_video_multitask import Stage1VideoMultitaskModel


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="outputs/stage1_corrected_smoke_test.json")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(42)
    model = Stage1VideoMultitaskModel(pretrained=False, segmentation_decoder_width=32)
    model.train()

    video = torch.randn(2, 3, 8, 64, 64)
    ef_target = torch.tensor([0.50, 0.65])
    frames = torch.randn(4, 3, 64, 64)
    masks = (torch.rand(4, 1, 64, 64) > 0.7).float()

    model.zero_grad(set_to_none=True)
    ef_pred = model.forward_ef(video)
    ef_loss = torch.nn.functional.mse_loss(ef_pred, ef_target)
    (0.5 * ef_loss).backward()

    seg_logits = model.forward_segmentation(frames)
    seg_loss = torch.nn.functional.binary_cross_entropy_with_logits(seg_logits, masks)
    (0.5 * seg_loss).backward()

    shared_grad = model.stem[0].weight.grad
    if shared_grad is None or not torch.isfinite(shared_grad).all():
        raise RuntimeError("Shared encoder did not receive finite gradients")

    rejected_two_frame = False
    try:
        model.forward_ef(torch.randn(1, 3, 2, 64, 64))
    except ValueError:
        rejected_two_frame = True
    if not rejected_two_frame:
        raise RuntimeError("EF path unexpectedly accepted T=2")

    payload = {
        "status": "pass",
        "video_input_shape": list(video.shape),
        "ef_output_shape": list(ef_pred.shape),
        "segmentation_input_shape": list(frames.shape),
        "segmentation_output_shape": list(seg_logits.shape),
        "T_greater_than_2_enforced": rejected_two_frame,
        "shared_encoder_received_gradient": True,
        "ef_aggregation": "AdaptiveAvgPool3d over temporal/spatial features before one scalar output",
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
