"""Functions for training and running segmentation."""

import math
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import skimage.draw
import torch
import torchvision
import tqdm
import echonet

from echonet.utils.vit_segmentation_model import ViTSegmentationModel


def _make_segmentation_model(modelname, pretrained):
    """Create torchvision segmentation model while avoiding aux classifier during training."""
    
    if modelname == "vit_base_patch16_224":
        return ViTSegmentationModel(
            model_name="vit_base_patch16_224",
            pretrained=pretrained,
            num_classes=1,
        )

    if modelname == "deeplabv3_resnet50":
        if pretrained:
            try:
                from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights
                weights = DeepLabV3_ResNet50_Weights.DEFAULT
                model = torchvision.models.segmentation.deeplabv3_resnet50(
                    weights=weights,
                    aux_loss=True
                )
            except Exception:
                model = torchvision.models.segmentation.deeplabv3_resnet50(
                    pretrained=True,
                    aux_loss=True
                )
        else:
            model = torchvision.models.segmentation.deeplabv3_resnet50(
                weights=None,
                weights_backbone=None,
                aux_loss=False
            )
    else:
        model = torchvision.models.segmentation.__dict__[modelname](
            pretrained=pretrained,
            aux_loss=True if pretrained else False
        )

    # Remove aux classifier so model behaves like original EchoNet aux_loss=False setup
    model.aux_classifier = None

    # Binary LV segmentation: output channel = 1
    model.classifier[-1] = torch.nn.Conv2d(
        model.classifier[-1].in_channels,
        1,
        kernel_size=model.classifier[-1].kernel_size
    )

    return model


def run(
    num_epochs=50,
    modelname="deeplabv3_resnet50",
    pretrained=True,
    weights=None,
    output=None,
    device=None,
    n_train_patients=None,
    num_workers=8,
    batch_size=8,
    seed=0,
    lr=1e-5,
    weight_decay=1e-5,
    lr_step_period=None,
    save_segmentation=False,
    block_size=1024,
    run_test=False,
):
    """Train/test EchoNet LV segmentation model."""

    np.random.seed(seed)
    torch.manual_seed(seed)

    if output is None:
        output = os.path.join(
            "output",
            "segmentation",
            "{}_{}".format(modelname, "pretrained" if pretrained else "random")
        )
    os.makedirs(output, exist_ok=True)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    # Model
    model = _make_segmentation_model(modelname, pretrained)

    if device.type == "cuda":
        model = torch.nn.DataParallel(model)
    model.to(device)

    # Optional checkpoint initialization
    if weights is not None:
        checkpoint = torch.load(weights, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])

    # Optimizer
    optim = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=weight_decay
    )

    if lr_step_period is None:
        lr_step_period = math.inf
    scheduler = torch.optim.lr_scheduler.StepLR(optim, lr_step_period)

    # Dataset statistics
    mean, std = echonet.utils.get_mean_and_std(
        echonet.datasets.Echo(split="train"),
        num_workers=num_workers
    )

    tasks = ["LargeFrame", "SmallFrame", "LargeTrace", "SmallTrace"]
    kwargs = {
        "target_type": tasks,
        "mean": mean,
        "std": std,
    }

    dataset = {}
    dataset["train"] = echonet.datasets.Echo(split="train", **kwargs)

    if n_train_patients is not None and len(dataset["train"]) > n_train_patients:
        indices = np.random.choice(len(dataset["train"]), n_train_patients, replace=False)
        dataset["train"] = torch.utils.data.Subset(dataset["train"], indices)

    dataset["val"] = echonet.datasets.Echo(split="val", **kwargs)

    with open(os.path.join(output, "log.csv"), "a") as f:
        epoch_resume = 0
        bestLoss = float("inf")

        try:
            checkpoint = torch.load(os.path.join(output, "checkpoint.pt"), map_location=device)
            model.load_state_dict(checkpoint["state_dict"])
            optim.load_state_dict(checkpoint["opt_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_dict"])
            epoch_resume = checkpoint["epoch"] + 1
            bestLoss = checkpoint["best_loss"]
            f.write("Resuming from epoch {}\n".format(epoch_resume))
        except FileNotFoundError:
            f.write("Starting run from scratch\n")

        for epoch in range(epoch_resume, num_epochs):
            print("Epoch #{}".format(epoch), flush=True)

            for phase in ["train", "val"]:
                start_time = time.time()

                for i in range(torch.cuda.device_count()):
                    torch.cuda.reset_peak_memory_stats(i)

                ds = dataset[phase]
                dataloader = torch.utils.data.DataLoader(
                    ds,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    shuffle=True,
                    pin_memory=(device.type == "cuda"),
                    drop_last=(phase == "train")
                )

                loss, large_inter, large_union, small_inter, small_union = run_epoch(
                    model,
                    dataloader,
                    phase == "train",
                    optim,
                    device
                )

                overall_dice = 2 * (large_inter.sum() + small_inter.sum()) / (
                    large_union.sum() + large_inter.sum() +
                    small_union.sum() + small_inter.sum()
                )
                large_dice = 2 * large_inter.sum() / (large_union.sum() + large_inter.sum())
                small_dice = 2 * small_inter.sum() / (small_union.sum() + small_inter.sum())

                memory_reserved = (
                    sum(torch.cuda.max_memory_reserved(i) for i in range(torch.cuda.device_count()))
                    if device.type == "cuda" else 0
                )
                memory_allocated = (
                    sum(torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count()))
                    if device.type == "cuda" else 0
                )

                f.write("{},{},{},{},{},{},{},{},{},{},{}\n".format(
                    epoch,
                    phase,
                    loss,
                    overall_dice,
                    large_dice,
                    small_dice,
                    time.time() - start_time,
                    large_inter.size,
                    memory_allocated,
                    memory_reserved,
                    batch_size
                ))
                f.flush()

            scheduler.step()

            save = {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "best_loss": bestLoss,
                "loss": loss,
                "opt_dict": optim.state_dict(),
                "scheduler_dict": scheduler.state_dict(),
            }

            torch.save(save, os.path.join(output, "checkpoint.pt"))

            if loss < bestLoss:
                torch.save(save, os.path.join(output, "best.pt"))
                bestLoss = loss

        if num_epochs != 0:
            checkpoint = torch.load(os.path.join(output, "best.pt"), map_location=device)
            model.load_state_dict(checkpoint["state_dict"])
            f.write("Best validation loss {} from epoch {}\n".format(
                checkpoint["loss"],
                checkpoint["epoch"]
            ))

        if run_test:
            for split in ["val", "test"]:
                dataset_eval = echonet.datasets.Echo(split=split, **kwargs)
                dataloader = torch.utils.data.DataLoader(
                    dataset_eval,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    shuffle=False,
                    pin_memory=(device.type == "cuda")
                )

                loss, large_inter, large_union, small_inter, small_union = run_epoch(
                    model,
                    dataloader,
                    False,
                    None,
                    device
                )

                overall_dice = 2 * (large_inter + small_inter) / (
                    large_union + large_inter + small_union + small_inter
                )
                large_dice = 2 * large_inter / (large_union + large_inter)
                small_dice = 2 * small_inter / (small_union + small_inter)

                for title, dice in [
                    ("overall", overall_dice),
                    ("large", large_dice),
                    ("small", small_dice)
                ]:
                    fig = plt.figure(figsize=(3, 2))
                    plt.hist(dice, bins=np.arange(0, 1 + 1e-6, 0.01))
                    plt.xlabel("DSC")
                    plt.ylabel("Videos")
                    plt.xlim([0, 1])
                    plt.tight_layout()
                    plt.savefig(os.path.join(output, "hist_{}_{}.pdf".format(title, split)))
                    plt.close(fig)

                with open(os.path.join(output, "{}_dice.csv".format(split)), "w") as g:
                    g.write("Filename, Overall, Large, Small\n")
                    for filename, overall, large, small in zip(
                        dataset_eval.fnames,
                        overall_dice,
                        large_dice,
                        small_dice
                    ):
                        g.write("{},{},{},{}\n".format(filename, overall, large, small))

                f.write("{} dice (overall): {:.4f} ({:.4f} - {:.4f})\n".format(
                    split,
                    *echonet.utils.bootstrap(
                        np.concatenate((large_inter, small_inter)),
                        np.concatenate((large_union, small_union)),
                        echonet.utils.dice_similarity_coefficient
                    )
                ))
                f.write("{} dice (large):   {:.4f} ({:.4f} - {:.4f})\n".format(
                    split,
                    *echonet.utils.bootstrap(
                        large_inter,
                        large_union,
                        echonet.utils.dice_similarity_coefficient
                    )
                ))
                f.write("{} dice (small):   {:.4f} ({:.4f} - {:.4f})\n".format(
                    split,
                    *echonet.utils.bootstrap(
                        small_inter,
                        small_union,
                        echonet.utils.dice_similarity_coefficient
                    )
                ))
                f.flush()


def run_epoch(model, dataloader, train, optim, device):
    """Run one epoch of training/evaluation for segmentation."""

    total = 0.0
    n = 0

    pos = 0
    neg = 0
    pos_pix = 0
    neg_pix = 0

    model.train(train)

    large_inter = 0
    large_union = 0
    small_inter = 0
    small_union = 0

    large_inter_list = []
    large_union_list = []
    small_inter_list = []
    small_union_list = []

    with torch.set_grad_enabled(train):
        with tqdm.tqdm(total=len(dataloader)) as pbar:
            for _, (large_frame, small_frame, large_trace, small_trace) in dataloader:
                pos += (large_trace == 1).sum().item()
                pos += (small_trace == 1).sum().item()
                neg += (large_trace == 0).sum().item()
                neg += (small_trace == 0).sum().item()

                pos_pix += (large_trace == 1).sum(0).to("cpu").detach().numpy()
                pos_pix += (small_trace == 1).sum(0).to("cpu").detach().numpy()
                neg_pix += (large_trace == 0).sum(0).to("cpu").detach().numpy()
                neg_pix += (small_trace == 0).sum(0).to("cpu").detach().numpy()

                # Diastole / large frame
                large_frame = large_frame.to(device)
                large_trace = large_trace.to(device)

                y_large = model(large_frame)["out"]

                loss_large = torch.nn.functional.binary_cross_entropy_with_logits(
                    y_large[:, 0, :, :],
                    large_trace,
                    reduction="sum"
                )

                large_pred = y_large[:, 0, :, :].detach().cpu().numpy() > 0.0
                large_true = large_trace[:, :, :].detach().cpu().numpy() > 0.0

                large_inter += np.logical_and(large_pred, large_true).sum()
                large_union += np.logical_or(large_pred, large_true).sum()

                large_inter_list.extend(np.logical_and(large_pred, large_true).sum((1, 2)))
                large_union_list.extend(np.logical_or(large_pred, large_true).sum((1, 2)))

                # Systole / small frame
                small_frame = small_frame.to(device)
                small_trace = small_trace.to(device)

                y_small = model(small_frame)["out"]

                loss_small = torch.nn.functional.binary_cross_entropy_with_logits(
                    y_small[:, 0, :, :],
                    small_trace,
                    reduction="sum"
                )

                small_pred = y_small[:, 0, :, :].detach().cpu().numpy() > 0.0
                small_true = small_trace[:, :, :].detach().cpu().numpy() > 0.0

                small_inter += np.logical_and(small_pred, small_true).sum()
                small_union += np.logical_or(small_pred, small_true).sum()

                small_inter_list.extend(np.logical_and(small_pred, small_true).sum((1, 2)))
                small_union_list.extend(np.logical_or(small_pred, small_true).sum((1, 2)))

                loss = (loss_large + loss_small) / 2

                if train:
                    optim.zero_grad()
                    loss.backward()
                    optim.step()

                total += loss.item()
                n += large_trace.size(0)

                p = pos / (pos + neg)
                p_pix = (pos_pix + 1) / (pos_pix + neg_pix + 2)

                pbar.set_postfix_str(
                    "{:.4f} ({:.4f}) / {:.4f} {:.4f}, {:.4f}, {:.4f}".format(
                        total / n / 112 / 112,
                        loss.item() / large_trace.size(0) / 112 / 112,
                        -p * math.log(p) - (1 - p) * math.log(1 - p),
                        (-p_pix * np.log(p_pix) - (1 - p_pix) * np.log(1 - p_pix)).mean(),
                        2 * large_inter / (large_union + large_inter),
                        2 * small_inter / (small_union + small_inter)
                    )
                )
                pbar.update()

    large_inter_list = np.array(large_inter_list)
    large_union_list = np.array(large_union_list)
    small_inter_list = np.array(small_inter_list)
    small_union_list = np.array(small_union_list)

    return (
        total / n / 112 / 112,
        large_inter_list,
        large_union_list,
        small_inter_list,
        small_union_list,
    )


def _video_collate_fn(x):
    """Collate function for saving videos."""

    video, target = zip(*x)
    i = list(map(lambda t: t.shape[1], video))
    video = torch.as_tensor(np.swapaxes(np.concatenate(video, 1), 0, 1))
    target = zip(*target)

    return video, target, i