"""Functions for training and running segmentation."""

import math
import sklearn
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


def run(num_epochs=50,
        modelname="deeplabv3_resnet50",
        pretrained=False,
        output=None,
        device=None,
        n_train_patients=None,
        num_workers=8,
        batch_size=8,
        seed=0,
        lr_step_period=None,
        save_segmentation=False,
        block_size=1024,
        run_test=False):
    """Trains/tests segmentation model."""

    # Seed RNGs
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Set default output directory
    if output is None:
        output = os.path.join("output", "segmentation", "{}_{}".format(modelname, "pretrained" if pretrained else "random"))
    os.makedirs(output, exist_ok=True)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up model (2D segmentation)
    model = torchvision.models.segmentation.__dict__[modelname](pretrained=pretrained, aux_loss=False)
    # Change output channels to 1 (background/foreground)
    model.classifier[-1] = torch.nn.Conv2d(model.classifier[-1].in_channels, 1, kernel_size=model.classifier[-1].kernel_size)

    if device.type == "cuda":
        model = torch.nn.DataParallel(model)
    model.to(device)

    # Set up optimizer
    optim = torch.optim.SGD(model.parameters(), lr=1e-5, momentum=0.9)
    if lr_step_period is None:
        lr_step_period = math.inf
    scheduler = torch.optim.lr_scheduler.StepLR(optim, lr_step_period)

    # Compute dataset statistics
    mean, std = echonet.utils.get_mean_and_std(echonet.datasets.Echo(split="train"), num_workers=num_workers)

    # Only these four target types
    tasks = ["LargeFrame", "SmallFrame", "LargeTrace", "SmallTrace"]
    kwargs = {
        "target_type": tasks,
        "mean": mean,
        "std": std,
        "length": 1,     # single frame
        "period": 1,
    }

    # Datasets and dataloaders
    train_dataset = echonet.datasets.Echo(split="train", **kwargs)
    if n_train_patients is not None and len(train_dataset) > n_train_patients:
        indices = np.random.choice(len(train_dataset), n_train_patients, replace=False)
        train_dataset = torch.utils.data.Subset(train_dataset, indices)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, num_workers=num_workers,
        shuffle=True, pin_memory=(device.type == "cuda"), drop_last=True)
    val_dataloader = torch.utils.data.DataLoader(
        echonet.datasets.Echo(split="val", **kwargs), batch_size=batch_size,
        num_workers=num_workers, shuffle=False, pin_memory=(device.type == "cuda"))

    dataloaders = {'train': train_dataloader, 'val': val_dataloader}

    # Training loop
    with open(os.path.join(output, "log.csv"), "a") as f:
        epoch_resume = 0
        bestLoss = float("inf")
        try:
            checkpoint = torch.load(os.path.join(output, "checkpoint.pt"))
            model.load_state_dict(checkpoint['state_dict'])
            optim.load_state_dict(checkpoint['opt_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_dict'])
            epoch_resume = checkpoint["epoch"] + 1
            bestLoss = checkpoint["best_loss"]
            f.write("Resuming from epoch {}\n".format(epoch_resume))
        except FileNotFoundError:
            f.write("Starting run from scratch\n")

        for epoch in range(epoch_resume, num_epochs):
            print("Epoch #{}".format(epoch), flush=True)
            for phase in ['train', 'val']:
                start_time = time.time()
                for i in range(torch.cuda.device_count()):
                    torch.cuda.reset_peak_memory_stats(i)

                loss, large_inter, large_union, small_inter, small_union = run_epoch(
                    model, dataloaders[phase], phase == "train", optim, device)

                overall_dice = 2 * (large_inter.sum() + small_inter.sum()) / (large_union.sum() + large_inter.sum() + small_union.sum() + small_inter.sum())
                large_dice = 2 * large_inter.sum() / (large_union.sum() + large_inter.sum())
                small_dice = 2 * small_inter.sum() / (small_union.sum() + small_inter.sum())

                f.write("{},{},{},{},{},{},{},{},{},{},{}\n".format(
                    epoch, phase, loss, overall_dice, large_dice, small_dice,
                    time.time() - start_time, large_inter.size,
                    sum(torch.cuda.max_memory_allocated() for i in range(torch.cuda.device_count())),
                    sum(torch.cuda.max_memory_cached() for i in range(torch.cuda.device_count())),
                    batch_size))
                f.flush()

            scheduler.step()

            # Save checkpoint
            save = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'best_loss': bestLoss,
                'loss': loss,
                'opt_dict': optim.state_dict(),
                'scheduler_dict': scheduler.state_dict(),
            }
            torch.save(save, os.path.join(output, "checkpoint.pt"))
            if loss < bestLoss:
                torch.save(save, os.path.join(output, "best.pt"))
                bestLoss = loss

        # Load best weights and test
        checkpoint = torch.load(os.path.join(output, "best.pt"))
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        f.write("Best validation loss {} from epoch {}\n".format(checkpoint["loss"], checkpoint["epoch"]))

        if run_test:
            for split in ["val", "test"]:
                dataset = echonet.datasets.Echo(split=split, **kwargs)
                dataloader = torch.utils.data.DataLoader(
                    dataset, batch_size=batch_size, num_workers=num_workers,
                    shuffle=False, pin_memory=(device.type == "cuda"))

                loss, large_inter, large_union, small_inter, small_union = run_epoch(
                    model, dataloader, False, None, device)

                overall_dice = 2 * (large_inter + small_inter) / (large_union + large_inter + small_union + small_inter)
                large_dice = 2 * large_inter / (large_union + large_inter)
                small_dice = 2 * small_inter / (small_union + small_inter)

                # Save dice histograms
                for (title, dice) in [("overall", overall_dice), ("large", large_dice), ("small", small_dice)]:
                    fig = plt.figure(figsize=(3, 2))
                    plt.hist(dice, bins=np.arange(0, 1 + 1e-6, 0.01))
                    plt.xlabel("DSC")
                    plt.ylabel("Videos")
                    plt.xlim([0, 1])
                    plt.tight_layout()
                    plt.savefig(os.path.join(output, "hist_{}_{}.pdf".format(title, split)))
                    plt.close(fig)

                # Write per-video dice to CSV
                with open(os.path.join(output, "{}_dice.csv".format(split)), "w") as g:
                    g.write("Filename, Overall, Large, Small\n")
                    for (filename, ov, la, sm) in zip(dataset.fnames, overall_dice, large_dice, small_dice):
                        g.write("{},{},{},{}\n".format(filename, ov, la, sm))

                f.write("{} dice (overall): {:.4f} ({:.4f} - {:.4f})\n".format(
                    split, *echonet.utils.bootstrap(np.concatenate((large_inter, small_inter)),
                                                    np.concatenate((large_union, small_union)),
                                                    echonet.utils.dice_similarity_coefficient)))
                f.write("{} dice (large):   {:.4f} ({:.4f} - {:.4f})\n".format(
                    split, *echonet.utils.bootstrap(large_inter, large_union,
                                                    echonet.utils.dice_similarity_coefficient)))
                f.write("{} dice (small):   {:.4f} ({:.4f} - {:.4f})\n".format(
                    split, *echonet.utils.bootstrap(small_inter, small_union,
                                                    echonet.utils.dice_similarity_coefficient)))
                f.flush()


def run_epoch(model, dataloader, train, optim, device):
    """Run one epoch of training/evaluation for segmentation."""
    model.train(train)

    total_loss = 0.0
    n = 0

    # For computing Dice and IoU
    large_inter_list = []
    large_union_list = []
    small_inter_list = []
    small_union_list = []

    with torch.set_grad_enabled(train):
        with tqdm.tqdm(total=len(dataloader)) as pbar:
            for (_, (large_frame, small_frame, large_trace, small_trace)) in dataloader:
                # Move tensors to device
                large_frame = large_frame.to(device)
                small_frame = small_frame.to(device)
                large_trace = large_trace.to(device)
                small_trace = small_trace.to(device)

                # ----- Diastolic (large) prediction -----
                output_large = model(large_frame)["out"]        # (B,1,H,W)
                # Target: same shape as output (B,1,H,W) but values 0/1
                target_large = large_trace.unsqueeze(1)        # add channel dim
                # Compute loss (binary cross-entropy)
                loss_large = torch.nn.functional.binary_cross_entropy_with_logits(output_large, target_large)

                # ----- Systolic (small) prediction -----
                output_small = model(small_frame)["out"]
                target_small = small_trace.unsqueeze(1)
                loss_small = torch.nn.functional.binary_cross_entropy_with_logits(output_small, target_small)

                loss = (loss_large + loss_small) / 2

                if train:
                    optim.zero_grad()
                    loss.backward()
                    optim.step()

                total_loss += loss.item() * large_frame.size(0)
                n += large_frame.size(0)

                # Compute intersection and union for dice (threshold at 0.5)
                pred_large = (torch.sigmoid(output_large) > 0.5).float()
                pred_small = (torch.sigmoid(output_small) > 0.5).float()

                # For each sample in batch, store inter and union
                with torch.no_grad():
                    inter_l = (pred_large * target_large).sum(dim=(1,2,3)).cpu().numpy()
                    union_l = ((pred_large + target_large) > 0.5).sum(dim=(1,2,3)).cpu().numpy()
                    inter_s = (pred_small * target_small).sum(dim=(1,2,3)).cpu().numpy()
                    union_s = ((pred_small + target_small) > 0.5).sum(dim=(1,2,3)).cpu().numpy()

                large_inter_list.extend(inter_l)
                large_union_list.extend(union_l)
                small_inter_list.extend(inter_s)
                small_union_list.extend(union_s)

                # Progress bar
                pbar.set_postfix_str("loss: {:.4f}".format(total_loss / n))
                pbar.update()

    # Convert to numpy arrays
    large_inter = np.array(large_inter_list)
    large_union = np.array(large_union_list)
    small_inter = np.array(small_inter_list)
    small_union = np.array(small_union_list)

    return total_loss / n, large_inter, large_union, small_inter, small_union


def _video_collate_fn(x):
    """Collate function for saving videos (not used in training)."""
    video, target = zip(*x)
    i = list(map(lambda t: t.shape[1], video))
    video = torch.as_tensor(np.swapaxes(np.concatenate(video, 1), 0, 1))
    target = zip(*target)
    return video, target, i