#!/usr/bin/env python3

"""Code to generate plots for Extended Data Fig. 3."""

import argparse
import os
import matplotlib
import matplotlib.pyplot as plt

def latexify():
    plt.rcParams.update({
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "figure.dpi": 300,
        "savefig.dpi": 300,
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", nargs="?", default="output")
    parser.add_argument("fig", nargs="?", default=os.path.join("figure", "loss"))
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument("--period", type=int, default=2)
    args = parser.parse_args()

    latexify()
    os.makedirs(args.fig, exist_ok=True)
    fig = plt.figure(figsize=(7, 5))
    gs = matplotlib.gridspec.GridSpec(ncols=3, nrows=2, figure=fig, width_ratios=[2.75, 2.75, 1.50])

    try:
        colors = list(matplotlib.colors.TABLEAU_COLORS.values())
    except AttributeError:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # EF loss curves
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1], sharey=ax0)
    for pretrained in [True]:
        for idx, model in enumerate(["r2plus1d_18", "r3d_18", "mc3_18"]):
            loss = load(os.path.join(args.dir, "video", f"{model}_{args.frames}_{args.period}_{'pretrained' if pretrained else 'random'}", "log.csv"))
            if loss is None:
                continue
            ax0.plot(range(1, 1 + len(loss["train"])), loss["train"], "-" if pretrained else "--", color=colors[idx])
            ax1.plot(range(1, 1 + len(loss["val"])), loss["val"], "-" if pretrained else "--", color=colors[idx])

    ax0.text(-0.25, 1.00, "(a)", transform=ax0.transAxes)
    ax1.text(-0.25, 1.00, "(b)", transform=ax1.transAxes)
    ax0.set_xlabel("Epochs")
    ax1.set_xlabel("Epochs")
    ax0.set_xticks([0, 15, 30, 45])
    ax1.set_xticks([0, 15, 30, 45])
    ax0.set_ylabel("Training MSE Loss")
    ax1.set_ylabel("Validation MSE Loss")

    # Segmentation loss curves
    ax0 = fig.add_subplot(gs[1, 0])
    ax1 = fig.add_subplot(gs[1, 1], sharey=ax0)
    pretrained = False
    model = "deeplabv3_resnet50"
    loss = load(os.path.join(args.dir, "segmentation", f"{model}_{'pretrained' if pretrained else 'random'}", "log.csv"))
    if loss is not None:
        ax0.plot(range(1, 1 + len(loss["train"])), loss["train"], "--", color=colors[3])
        ax1.plot(range(1, 1 + len(loss["val"])), loss["val"], "--", color=colors[3])

    ax0.text(-0.25, 1.00, "(c)", transform=ax0.transAxes)
    ax1.text(-0.25, 1.00, "(d)", transform=ax1.transAxes)
    ax0.set_ylim([0, 0.13])
    ax0.set_xlabel("Epochs")
    ax1.set_xlabel("Epochs")
    ax0.set_xticks([0, 25, 50])
    ax1.set_xticks([0, 25, 50])
    ax0.set_ylabel("Training Cross Entropy Loss")
    ax1.set_ylabel("Validation Cross Entropy Loss")

    # Legend
    ax = fig.add_subplot(gs[:, 2])
    for model, color in zip(["EchoNet-Dynamic (EF)", "R3D", "MC3", "EchoNet-Dynamic (Seg)"], colors[:4]):
        ax.plot([float("nan")], [float("nan")], "-", color=color, label=model)
    ax.set_title("")
    ax.axis("off")
    ax.legend(loc="center")

    plt.tight_layout()
    plt.savefig(os.path.join(args.fig, "loss.pdf"))
    plt.savefig(os.path.join(args.fig, "loss.eps"))
    plt.savefig(os.path.join(args.fig, "loss.png"))
    plt.close(fig)


def load(filename):
    """Loads losses from specified file."""
    if not os.path.exists(filename):
        print(f"Warning: {filename} not found")
        return None
    losses = {"train": [], "val": []}
    with open(filename, "r") as f:
        for line in f:
            line = line.strip().split(",")
            if len(line) < 3:
                continue
            try:
                epoch = int(line[0])
                split = line[1].strip()
                loss = float(line[2])
            except (ValueError, IndexError):
                continue
            if split not in losses:
                continue
            if epoch == len(losses[split]):
                losses[split].append(loss)
            elif epoch == len(losses[split]) - 1:
                losses[split][-1] = loss
            else:
                # This can happen if epochs are not sequential; ignore
                pass
    return losses if losses["train"] or losses["val"] else None


if __name__ == "__main__":
    main()