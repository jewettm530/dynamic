"""Evaluation helpers for corrected Stage 1."""

from __future__ import annotations

from pathlib import Path

import torch

from echonet.modeling.stage1_video_multitask import Stage1VideoMultitaskModel
from echonet.utils.stage1_metrics import (
    dice_score,
    ef_fraction_to_percent,
    hd95_pixels,
    regression_metrics,
    summarize_segmentation,
)


def load_checkpoint(path: str | Path, device):
    path = Path(path)
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # older PyTorch
        return torch.load(path, map_location=device)


def build_model(checkpoint, device):
    cfg = checkpoint.get("config", {})
    width = int(cfg.get("segmentation_decoder_width", 128))
    model = Stage1VideoMultitaskModel(
        pretrained=False,
        segmentation_decoder_width=width,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model


def video_settings(checkpoint):
    cfg = checkpoint.get("config", {})
    ef_input = cfg.get("ef_input", {})
    training = cfg.get("training", {})
    return {
        "frames": int(ef_input.get("T", 32)),
        "period": int(ef_input.get("period", 2)),
        "training_sampling": ef_input.get("training_sampling"),
        "validation_sampling": ef_input.get("validation_sampling"),
        "split": cfg.get("split"),
        "preprocessing": cfg.get("preprocessing"),
        "epochs": int(training.get("epochs", -1)),
        "batch_size": int(training.get("batch_size", -1)),
        "optimizer": training.get("optimizer"),
        "learning_rate": training.get("learning_rate"),
        "momentum": training.get("momentum"),
        "weight_decay": training.get("weight_decay"),
        "lr_step_period": training.get("lr_step_period"),
        "checkpoint_rule": cfg.get("checkpoint_rule"),
    }


def assert_b1_b3_matched(b1_checkpoint, b3_checkpoint):
    a = video_settings(b1_checkpoint)
    b = video_settings(b3_checkpoint)
    if a != b:
        raise RuntimeError(
            f"B1/B3 matched-comparison settings differ:\nB1={a}\nB3={b}"
        )


def evaluate_ef(model, loader, device):
    targets, predictions, names = [], [], []
    with torch.no_grad():
        for batch in loader:
            video = batch["video"].to(
                device, dtype=torch.float32, non_blocking=True
            )
            ef = batch["ef"].to(
                device, dtype=torch.float32, non_blocking=True
            ).reshape(-1)
            pred = model.forward_ef(video).reshape(-1)
            targets.extend(ef_fraction_to_percent(ef.cpu().numpy()).tolist())
            predictions.extend(
                ef_fraction_to_percent(pred.cpu().numpy()).tolist()
            )
            names.extend(list(batch["filename"]))
    metrics = regression_metrics(targets, predictions)
    metrics["n_videos"] = len(targets)
    rows = [
        {
            "filename": f,
            "ef_target_percent": y,
            "ef_prediction_percent": p,
        }
        for f, y, p in zip(names, targets, predictions)
    ]
    return metrics, rows


def evaluate_segmentation(model, loader, device):
    ed_dice, es_dice, ed_hd95, es_hd95 = [], [], [], []
    rows = []
    with torch.no_grad():
        for batch in loader:
            ed = batch["ed_image"].to(
                device, dtype=torch.float32, non_blocking=True
            )
            es = batch["es_image"].to(
                device, dtype=torch.float32, non_blocking=True
            )
            ed_mask = batch["ed_mask"].to(
                device, dtype=torch.float32, non_blocking=True
            )
            es_mask = batch["es_mask"].to(
                device, dtype=torch.float32, non_blocking=True
            )
            b = ed.shape[0]
            frames = torch.cat([ed, es], dim=0)
            masks = torch.cat([ed_mask, es_mask], dim=0)
            logits = model.forward_segmentation(frames)
            probs = torch.sigmoid(logits).cpu().numpy()[:, 0]
            truth = masks.cpu().numpy()[:, 0]
            for i, filename in enumerate(list(batch["filename"])):
                row = {"filename": filename}
                for phase, p, t in (
                    ("ed", probs[i], truth[i]),
                    ("es", probs[b + i], truth[b + i]),
                ):
                    pred, true = p >= 0.5, t >= 0.5
                    d = dice_score(pred, true)
                    h = hd95_pixels(pred, true)
                    row[f"dice_{phase}"] = d
                    row[f"hd95_{phase}"] = h
                    if phase == "ed":
                        ed_dice.append(d)
                        ed_hd95.append(h)
                    else:
                        es_dice.append(d)
                        es_hd95.append(h)
                row["mean_dice"] = (row["dice_ed"] + row["dice_es"]) / 2.0
                row["mean_hd95"] = (row["hd95_ed"] + row["hd95_es"]) / 2.0
                rows.append(row)
    metrics = summarize_segmentation(ed_dice, es_dice, ed_hd95, es_hd95)
    metrics["n_videos"] = len(rows)
    return metrics, rows
