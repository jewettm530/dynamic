#!/usr/bin/env python3

"""Code to generate plots for Extended Data Fig. 6."""

import os
import pickle
import numpy as np
import torch
import torchvision
import sklearn.metrics
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import echonet.datasets  # for Echo dataset class

# ----------------------------------------------------------------------
# Missing utility functions
# ----------------------------------------------------------------------
def dice_similarity_coefficient(inter, union):
    return 2 * inter / (union + inter + 1e-8)

def get_mean_and_std(dataset):
    """Compute mean and std over all frames of the dataset."""
    mean = 0.
    std = 0.
    n = 0
    for data, _ in dataset:
        # data shape: (C, T, H, W)
        mean += data.mean(axis=(1,2,3)).sum()
        std += data.std(axis=(1,2,3)).sum()
        n += 1
    mean /= n
    std /= n
    return mean, std

def evaluate_video_model(model, dataloader, device):
    """Simple evaluation: collect predictions and targets."""
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs).squeeze()
            y_true.extend(targets.cpu().numpy().flatten())
            y_pred.extend(outputs.cpu().numpy().flatten())
    mse = sklearn.metrics.mean_squared_error(y_true, y_pred)
    r2 = sklearn.metrics.r2_score(y_true, y_pred)
    return mse, np.array(y_pred), np.array(y_true)

def evaluate_segmentation_model(model, dataloader, device):
    """Evaluate segmentation: compute intersection and union per sample."""
    large_inter = []
    large_union = []
    small_inter = []
    small_union = []
    model.eval()
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            # targets is a tuple: (LargeFrame, SmallFrame, LargeTrace, SmallTrace)
            # We only need the traces (index 2 and 3)
            large_trace = targets[2].to(device)
            small_trace = targets[3].to(device)
            outputs = model(inputs)['out']  # shape (B,1,H,W)
            # Threshold at 0.5
            large_pred = (outputs > 0.5).float()
            small_pred = (outputs > 0.5).float()
            # Intersection and union
            large_inter.append((large_pred * large_trace).sum(dim=(1,2,3)).cpu().numpy())
            large_union.append(((large_pred + large_trace) > 0.5).sum(dim=(1,2,3)).cpu().numpy())
            small_inter.append((small_pred * small_trace).sum(dim=(1,2,3)).cpu().numpy())
            small_union.append(((small_pred + small_trace) > 0.5).sum(dim=(1,2,3)).cpu().numpy())
    return (np.concatenate(large_inter), np.concatenate(large_union),
            np.concatenate(small_inter), np.concatenate(small_union))


def main(fig_root=os.path.join("figure", "noise"),
         video_output=os.path.join("output", "video", "r2plus1d_18_32_2_pretrained"),
         seg_output=os.path.join("output", "segmentation", "deeplabv3_resnet50_random"),
         NOISE=(0, 0.1, 0.2, 0.3, 0.4, 0.5)):
    """Generate plots for Extended Data Fig. 6."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(fig_root, exist_ok=True)

    filename = os.path.join(fig_root, "data.pkl")
    try:
        with open(filename, "rb") as f:
            Y, YHAT, INTER, UNION = pickle.load(f)
    except FileNotFoundError:
        # Load trained models
        model_v = torchvision.models.video.r2plus1d_18(pretrained=False)
        model_v.fc = torch.nn.Linear(model_v.fc.in_features, 1)
        if device.type == "cuda":
            model_v = torch.nn.DataParallel(model_v)
        model_v.to(device)
        checkpoint = torch.load(os.path.join(video_output, "checkpoint.pt"))
        model_v.load_state_dict(checkpoint['state_dict'])

        model_s = torchvision.models.segmentation.deeplabv3_resnet50(aux_loss=False, pretrained=False)
        model_s.classifier[-1] = torch.nn.Conv2d(model_s.classifier[-1].in_channels, 1, kernel_size=model_s.classifier[-1].kernel_size)
        if device.type == "cuda":
            model_s = torch.nn.DataParallel(model_s)
        model_s.to(device)
        checkpoint = torch.load(os.path.join(seg_output, "checkpoint.pt"))
        model_s.load_state_dict(checkpoint['state_dict'])

        # Compute mean/std from training set (one time)
        train_dataset = echonet.datasets.Echo(split="train")
        mean, std = get_mean_and_std(train_dataset)

        Y, YHAT, INTER, UNION = [], [], [], []
        for noise in NOISE:
            print(f"Noise level: {noise}")
            # Save example image
            dataset_example = echonet.datasets.Echo(split="test", noise=noise)
            img = dataset_example[0][0][:, 0, :, :].astype(np.uint8).transpose(1,2,0)
            Image.fromarray(img).save(os.path.join(fig_root, f"noise_{int(100*noise)}.tif"))

            # Segmentation evaluation
            seg_dataset = echonet.datasets.Echo(
                split="test", target_type=["LargeFrame","SmallFrame","LargeTrace","SmallTrace"],
                mean=mean, std=std, noise=noise
            )
            seg_loader = torch.utils.data.DataLoader(seg_dataset, batch_size=16, shuffle=False,
                                                     num_workers=4, pin_memory=(device.type=="cuda"))
            large_inter, large_union, small_inter, small_union = evaluate_segmentation_model(model_s, seg_loader, device)
            inter = np.concatenate([large_inter, small_inter]).sum()
            union = np.concatenate([large_union, small_union]).sum()
            dice = dice_similarity_coefficient(inter, union)
            print(f"  Segmentation DSC: {dice:.4f}")
            INTER.append(np.concatenate([large_inter, small_inter]))
            UNION.append(np.concatenate([large_union, small_union]))

            # EF prediction evaluation
            ef_dataset = echonet.datasets.Echo(
                split="test", target_type="EF", mean=mean, std=std,
                length=32, period=2, noise=noise
            )
            ef_loader = torch.utils.data.DataLoader(ef_dataset, batch_size=16, shuffle=False,
                                                    num_workers=4, pin_memory=(device.type=="cuda"))
            mse, yhat, y = evaluate_video_model(model_v, ef_loader, device)
            print(f"  EF MSE: {mse:.4f}, R2: {sklearn.metrics.r2_score(y, yhat):.4f}")
            Y.append(y)
            YHAT.append(yhat)

        with open(filename, "wb") as f:
            pickle.dump((Y, YHAT, INTER, UNION), f)

    # Plotting
    latexify()  # same as defined earlier
    NOISE_PCT = [int(100*n) for n in NOISE]
    fig = plt.figure(figsize=(6.50, 4.75))
    gs = matplotlib.gridspec.GridSpec(3, 1, height_ratios=[2.0, 2.0, 0.75])
    ax = (plt.subplot(gs[0]), plt.subplot(gs[1]), plt.subplot(gs[2]))

    r2 = [sklearn.metrics.r2_score(y, yhat) for (y, yhat) in zip(Y, YHAT)]
    ax[0].plot(NOISE_PCT, r2, color="k", linewidth=1, marker=".")
    ax[0].set_xticks([])
    ax[0].set_ylabel("R$^2$")
    ax[0].axis([min(NOISE_PCT)-5, max(NOISE_PCT)+5, 0, 1])

    dice = [dice_similarity_coefficient(inter.sum(), union.sum()) for (inter, union) in zip(INTER, UNION)]
    ax[1].plot(NOISE_PCT, dice, color="k", linewidth=1, marker=".")
    ax[1].set_xlabel("Pixels Removed (%)")
    ax[1].set_ylabel("DSC")
    ax[1].axis([min(NOISE_PCT)-5, max(NOISE_PCT)+5, 0, 1])

    for noise in NOISE_PCT:
        img_path = os.path.join(fig_root, f"noise_{noise}.tif")
        if os.path.exists(img_path):
            image = plt.imread(img_path)
            imagebox = OffsetImage(image, zoom=0.4)
            ab = AnnotationBbox(imagebox, (noise, 0.0), frameon=False)
            ax[2].add_artist(ab)
    ax[2].axis("off")
    ax[2].axis([min(NOISE_PCT)-5, max(NOISE_PCT)+5, -1, 1])

    plt.tight_layout()
    plt.savefig(os.path.join(fig_root, "noise.pdf"), dpi=1200)
    plt.savefig(os.path.join(fig_root, "noise.eps"), dpi=300)
    plt.savefig(os.path.join(fig_root, "noise.png"), dpi=600)
    plt.close(fig)


if __name__ == "__main__":
    main()