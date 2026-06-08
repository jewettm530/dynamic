#!/usr/bin/env python3

"""Code to generate plots for Extended Data Fig. 4."""

import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# ----------------------------------------------------------------------
# Local replacement for echonet.utils.latexify (if missing)
# ----------------------------------------------------------------------
def latexify():
    """Set matplotlib parameters for publication‑ready plots (without requiring LaTeX)."""
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


def main(root=os.path.join("timing", "video"),
         fig_root=os.path.join("figure", "complexity"),
         FRAMES=(1, 8, 16, 32, 64, 96),
         pretrained=True):
    """Generate plots for Extended Data Fig. 4."""
    latexify()
    os.makedirs(fig_root, exist_ok=True)
    fig = plt.figure(figsize=(6.50, 2.50))
    gs = matplotlib.gridspec.GridSpec(1, 3, width_ratios=[2.5, 2.5, 1.50])
    ax = (plt.subplot(gs[0]), plt.subplot(gs[1]), plt.subplot(gs[2]))

    # Use TABLEAU_COLORS if available, else fallback to default cycle
    try:
        colors = list(matplotlib.colors.TABLEAU_COLORS.values())
    except AttributeError:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Legend
    for (model, color) in zip(["EchoNet-Dynamic (EF)", "R3D", "MC3"], colors[:3]):
        ax[2].plot([float("nan")], [float("nan")], "-", color=color, label=model)
    ax[2].set_title("")
    ax[2].axis("off")
    ax[2].legend(loc="center")

    for idx, (model, color) in enumerate(zip(["r2plus1d_18", "r3d_18", "mc3_18"], colors[:3])):
        for split in ["val"]:
            print(model, split)
            data = [load(root, model, frames, 1, pretrained, split) for frames in FRAMES]
            # Skip if any data is None
            if any(d is None for d in data):
                print(f"Warning: Missing data for {model}, skip plotting.")
                continue
            time = np.array([d[0] for d in data])
            n = np.array([d[1] for d in data])
            mem_allocated = np.array([d[2] for d in data])
            batch_size = np.array([d[4] for d in data])

            ax[0].plot(FRAMES, time / n, "-" if pretrained else "--", marker=".", color=color)
            print("Time:\n" + "\n".join(f"{f:8d}: {t/n_i:f}" for f, t, n_i in zip(FRAMES, time, n)))

            ax[1].plot(FRAMES, mem_allocated / batch_size / 1e9, "-" if pretrained else "--", marker=".", color=color)
            print("Memory:\n" + "\n".join(f"{f:8d}: {m/b/1e9:f}" for f, m, b in zip(FRAMES, mem_allocated, batch_size)))
            print()

    ax[0].set_xticks(FRAMES)
    ax[0].text(-0.05, 1.10, "(a)", transform=ax[0].transAxes)
    ax[0].set_xlabel("Clip length (frames)")
    ax[0].set_ylabel("Time Per Clip (seconds)")

    ax[1].set_xticks(FRAMES)
    ax[1].text(-0.05, 1.10, "(b)", transform=ax[1].transAxes)
    ax[1].set_xlabel("Clip length (frames)")
    ax[1].set_ylabel("Memory Per Clip (GB)")

    plt.tight_layout()
    plt.savefig(os.path.join(fig_root, "complexity.pdf"))
    plt.savefig(os.path.join(fig_root, "complexity.eps"))
    plt.close(fig)


def load(root, model, frames, period, pretrained, split):
    """Loads runtime and memory usage for specified hyperparameter choice."""
    fname = os.path.join(root, f"{model}_{frames}_{period}_{'pretrained' if pretrained else 'random'}", "log.csv")
    if not os.path.exists(fname):
        print(f"Warning: {fname} not found.")
        return None
    with open(fname, "r") as f:
        for line in f:
            line = line.strip().split(",")
            if len(line) < 4:
                continue
            if line[1] == split:
                # Expected format: epoch,split,time,n,mem_allocated,mem_cached,batch_size
                try:
                    time = float(line[2])
                    n = int(line[3])
                    mem_allocated = int(line[4])
                    mem_cached = int(line[5])
                    batch_size = int(line[6])
                    return time, n, mem_allocated, mem_cached, batch_size
                except (IndexError, ValueError):
                    print(f"Warning: Malformed line in {fname}: {line}")
                    continue
    print(f"Warning: No data for split={split} in {fname}")
    return None


if __name__ == "__main__":
    main()