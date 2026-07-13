import os
import sys
import random
import argparse
import time

sys.path.insert(0, DATA_DIR)

import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt

import echonet
from echonet.paths import (
    DATA_DIR,
    VISUALIZATIONS_OUTPUT_DIR,
    VIDEOS_DIR,
    EF_PROCESSING_VISUALIZATIONS_DIR,
    VIDEO_CHECKPOINTS_DIR,
)


def strip_module_prefix(state_dict):
    """Remove 'module.' prefix from DataParallel checkpoints."""
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            new_state_dict[key[7:]] = value
        else:
            new_state_dict[key] = value
    return new_state_dict


def load_ef_model(checkpoint_path, device):
    """Load trained R2Plus1D-18 EF prediction model."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = torchvision.models.video.r2plus1d_18()
    model.fc = torch.nn.Linear(model.fc.in_features, 1)

    state_dict = strip_module_prefix(checkpoint["state_dict"])
    model.load_state_dict(state_dict, strict=False)

    model.to(device)
    model.eval()

    return model


def prepare_frame_for_display(frame):
    """
    Convert frame from C x H x W to H x W.
    EchoNet frames are stored as 3-channel grayscale images.
    """
    frame = frame.transpose(1, 2, 0)

    if frame.shape[-1] == 3:
        frame = frame[:, :, 0]

    frame = np.clip(frame, 0, 1)
    return frame


def visualize_single_video(
    dataset,
    model,
    device,
    idx,
    output_dir,
    num_display_frames=8,
):
    """Randomly selected video -> sampled frames -> EF model -> predicted EF figure."""

    video, true_ef = dataset[idx]

    input_tensor = torch.from_numpy(video).unsqueeze(0).float().to(device)

    with torch.no_grad():
        pred_ef = model(input_tensor).item()

    abs_error = abs(pred_ef - float(true_ef))

    channels, frames, height, width = video.shape

    frame_indices = np.linspace(
        0,
        frames - 1,
        num_display_frames,
        dtype=int
    )

    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    axes = axes.flatten()

    for ax, frame_idx in zip(axes, frame_indices):
        frame = video[:, frame_idx, :, :]
        frame = prepare_frame_for_display(frame)

        ax.imshow(frame, cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"Sampled frame {frame_idx}")
        ax.axis("off")

    fig.suptitle(
        "EchoNet EF Prediction Processing Visualization\n"
        f"Dataset index: {idx} | "
        f"Input tensor: (1, {channels}, {frames}, {height}, {width}) | "
        f"True EF: {float(true_ef):.1f}% | "
        f"Predicted EF: {pred_ef:.1f}% | "
        f"Absolute error: {abs_error:.1f}%",
        fontsize=12
    )

    fig.text(
        0.5,
        0.03,
        "Processing steps: Random test video → 32-frame sampled clip "
        "→ normalized tensor input → R2Plus1D-18 model → predicted EF percentage.",
        ha="center",
        fontsize=10
    )

    plt.tight_layout(rect=[0, 0.06, 1, 0.90])

    output_path = os.path.join(
        output_dir,
        f"ef_processing_index_{idx}.png"
    )
    plt.savefig(output_path, dpi=150)
    plt.close(fig)

    return {
        "index": idx,
        "true_ef": float(true_ef),
        "pred_ef": pred_ef,
        "abs_error": abs_error,
        "output_path": output_path,
        "input_shape": (1, channels, frames, height, width),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Visualize how randomly selected EchoNet videos are processed for EF prediction."
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(VIDEO_CHECKPOINTS_DIR / "best.pt"),
        help="Path to trained EF model checkpoint."
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(EF_PROCESSING_VISUALIZATIONS_DIR),
        help="Directory to save visualization images."
    )

    parser.add_argument(
        "--num_videos",
        type=int,
        default=3,
        help="Number of random videos to visualize."
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducible video selection."
    )

    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to sample from."
    )

    parser.add_argument(
        "--length",
        type=int,
        default=32,
        help="Number of frames used as model input."
    )

    parser.add_argument(
        "--period",
        type=int,
        default=2,
        help="Frame sampling period."
    )

    parser.add_argument(
        "--num_display_frames",
        type=int,
        default=8,
        help="Number of sampled frames to show in each figure."
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run prediction on."
    )

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(
        "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    model = load_ef_model(args.checkpoint, device)
    print("EF model loaded successfully.")

    mean = np.array([0.0, 0.0, 0.0])
    std = np.array([1.0, 1.0, 1.0])

    dataset = echonet.datasets.Echo(
        split=args.split,
        target_type="EF",
        mean=mean,
        std=std,
        length=args.length,
        period=args.period,
    )

    print(f"Loaded EchoNet split: {args.split}")
    print(f"Dataset size: {len(dataset)}")

    selected_indices = random.sample(
        range(len(dataset)),
        k=min(args.num_videos, len(dataset))
    )

    summary_path = os.path.join(args.output_dir, "ef_processing_summary.csv")

    start = time.time()

    with open(summary_path, "w") as f:
        f.write("index,true_ef,predicted_ef,absolute_error,input_shape,output_path\n")

        for idx in selected_indices:
            result = visualize_single_video(
                dataset=dataset,
                model=model,
                device=device,
                idx=idx,
                output_dir=args.output_dir,
                num_display_frames=args.num_display_frames,
            )

            f.write(
                f"{result['index']},"
                f"{result['true_ef']:.4f},"
                f"{result['pred_ef']:.4f},"
                f"{result['abs_error']:.4f},"
                f"\"{result['input_shape']}\","
                f"{result['output_path']}\n"
            )

            print(
                f"Saved {result['output_path']} | "
                f"True EF={result['true_ef']:.1f}% | "
                f"Pred EF={result['pred_ef']:.1f}% | "
                f"Error={result['abs_error']:.1f}%"
            )

    print(f"\nSaved summary CSV: {summary_path}")
    print(f"Finished in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()