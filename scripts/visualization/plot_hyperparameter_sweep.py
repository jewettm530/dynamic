#!/usr/bin/env python3

"""Code to generate plots for Extended Data Fig. 1."""

import os
import matplotlib
import matplotlib.pyplot as plt
import echonet  # only for potential utils, but we define our own latexify here

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


def main(root=os.path.join("output", "video"),
         fig_root=os.path.join("figure", "hyperparameter"),
         FRAMES=(1, 8, 16, 32, 64, 96, None),
         PERIOD=(1, 2, 4, 6, 8)):
    """Generate plots for Extended Data Fig. 1."""
    latexify()
    os.makedirs(fig_root, exist_ok=True)

    MAX = FRAMES[-2]
    START = 1
    TERM0 = 104
    BREAK = 112
    TERM1 = 120
    ALL = 128
    END = 135
    RATIO = (BREAK - START) / (END - BREAK)

    fig = plt.figure(figsize=(3 + 2.5 + 1.5, 2.75))
    outer = matplotlib.gridspec.GridSpec(1, 3, width_ratios=[3, 2.5, 1.50])
    ax = plt.subplot(outer[2])
    ax2 = plt.subplot(outer[1])
    gs = matplotlib.gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0], width_ratios=[RATIO, 1], wspace=0.020)

    # Colors
    try:
        colors = list(matplotlib.colors.TABLEAU_COLORS.values())
    except AttributeError:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Legend
    for (model, color) in zip(["EchoNet-Dynamic (EF)", "R3D", "MC3"], colors[:3]):
        ax.plot([float("nan")], [float("nan")], "-", color=color, label=model)
    ax.plot([float("nan")], [float("nan")], "-", color="k", label="Pretrained")
    ax.plot([float("nan")], [float("nan")], "--", color="k", label="Random")
    ax.set_title("")
    ax.axis("off")
    ax.legend(loc="center")

    # Length sweep
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1], sharey=ax0)
    print("FRAMES")
    for (model, color) in zip(["r2plus1d_18", "r3d_18", "mc3_18"], colors[:3]):
        for pretrained in [True, False]:
            loss = [load(root, model, frames, 1, pretrained) for frames in FRAMES]
            # skip if any loss is None
            if any(l is None for l in loss):
                print(f"Warning: missing data for {model} pretrained={pretrained}")
                continue
            print(model, pretrained)
            print("    ".join(f"{l:.1f}" if l is not None else "None" for l in loss))

            l0 = loss[-2]
            l1 = loss[-1]
            ax0.plot(FRAMES[:-1] + (TERM0,),
                     loss[:-1] + [l0 + (l1 - l0) * (TERM0 - MAX) / (ALL - MAX)],
                     "-" if pretrained else "--", color=color)
            ax1.plot([TERM1, ALL],
                     [l0 + (l1 - l0) * (TERM1 - MAX) / (ALL - MAX)] + [loss[-1]],
                     "-" if pretrained else "--", color=color)
            ax0.scatter([x if x is not None else ALL for x in FRAMES], loss, color=color, s=4)
            ax1.scatter([x if x is not None else ALL for x in FRAMES], loss, color=color, s=4)

    ax0.set_xticks([x if x is not None else ALL for x in FRAMES])
    ax1.set_xticks([x if x is not None else ALL for x in FRAMES])
    ax0.set_xticklabels([str(x) if x is not None else "All" for x in FRAMES])
    ax1.set_xticklabels([str(x) if x is not None else "All" for x in FRAMES])

    ax0.set_xlim(START, BREAK)
    ax1.set_xlim(BREAK, END)

    ax0.spines['right'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.get_yaxis().set_visible(False)

    d = 0.015
    kwargs = dict(transform=ax0.transAxes, color='k', clip_on=False, linewidth=1)
    x0, x1, y0, y1 = ax0.axis()
    scale = (y1 - y0) / (x1 - x0) / 2
    ax0.plot((1 - scale * d, 1 + scale * d), (-d, +d), **kwargs)
    ax0.plot((1 - scale * d, 1 + scale * d), (1 - d, 1 + d), **kwargs)

    kwargs.update(transform=ax1.transAxes)
    x0, x1, y0, y1 = ax1.axis()
    scale = (y1 - y0) / (x1 - x0) / 2
    ax1.plot((-scale * d, scale * d), (-d, +d), **kwargs)
    ax1.plot((-scale * d, scale * d), (1 - d, 1 + d), **kwargs)

    ax0.text(-0.05, 1.10, "(a)", transform=ax0.transAxes)
    ax0.set_xlabel("Clip length (frames)")
    ax0.set_ylabel("Validation Loss")

    # Period sweep
    print("PERIOD")
    for (model, color) in zip(["r2plus1d_18", "r3d_18", "mc3_18"], colors[:3]):
        for pretrained in [True, False]:
            loss = [load(root, model, 64 // period, period, pretrained) for period in PERIOD]
            if any(l is None for l in loss):
                print(f"Warning: missing data for {model} pretrained={pretrained}")
                continue
            print(model, pretrained)
            print("    ".join(f"{l:.1f}" for l in loss))
            ax2.plot(PERIOD, loss, "-" if pretrained else "--", marker=".", color=color)

    ax2.set_xticks(PERIOD)
    ax2.text(-0.05, 1.10, "(b)", transform=ax2.transAxes)
    ax2.set_xlabel("Sampling Period (frames)")
    ax2.set_ylabel("Validation Loss")

    plt.tight_layout()
    plt.savefig(os.path.join(fig_root, "hyperparameter.pdf"))
    plt.savefig(os.path.join(fig_root, "hyperparameter.eps"))
    plt.savefig(os.path.join(fig_root, "hyperparameter.png"))
    plt.close(fig)


def load(root, model, frames, period, pretrained):
    """Loads best validation loss for specified hyperparameter choice."""
    pretrained_str = "pretrained" if pretrained else "random"
    fname = os.path.join(root, f"{model}_{frames}_{period}_{pretrained_str}", "log.csv")
    if not os.path.exists(fname):
        print(f"Warning: {fname} not found.")
        return None
    with open(fname, "r") as f:
        for line in f:
            # Look for "Best validation loss" line (could be colon or space)
            if "Best validation loss" in line:
                # Try to extract the number
                parts = line.split()
                for part in parts:
                    try:
                        return float(part)
                    except ValueError:
                        continue
    print(f"Warning: Could not find best validation loss in {fname}")
    return None


if __name__ == "__main__":
    main()