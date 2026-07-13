import os
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam

from models.multitask_deeplab import MultitaskDeepLabV3
from losses.multitask_loss import MultitaskLoss
from utils.metrics import dice_coefficient, iou_score, classification_accuracy


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()

    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    total_acc = 0.0

    for batch in dataloader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss, loss_dict = criterion(outputs, masks, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_dice += dice_coefficient(outputs["segmentation"], masks)
        total_iou += iou_score(outputs["segmentation"], masks)
        total_acc += classification_accuracy(outputs["classification"], labels)

    n = len(dataloader)

    return {
        "loss": total_loss / n,
        "dice": total_dice / n,
        "iou": total_iou / n,
        "accuracy": total_acc / n
    }


def validate(model, dataloader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    total_acc = 0.0

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(images)
            loss, loss_dict = criterion(outputs, masks, labels)

            total_loss += loss.item()
            total_dice += dice_coefficient(outputs["segmentation"], masks)
            total_iou += iou_score(outputs["segmentation"], masks)
            total_acc += classification_accuracy(outputs["classification"], labels)

    n = len(dataloader)

    return {
        "loss": total_loss / n,
        "dice": total_dice / n,
        "iou": total_iou / n,
        "accuracy": total_acc / n
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_classes = 2
    epochs = 25
    lr = 1e-4

    model = MultitaskDeepLabV3(
        num_classes=num_classes,
        pretrained=True
    ).to(device)

    criterion = MultitaskLoss(
        segmentation_loss="bce_dice",
        seg_weight=1.0,
        class_weight=0.3
    )

    optimizer = Adam(model.parameters(), lr=lr)

    # Replace these with your real dataset objects
    train_dataset = None
    val_dataset = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=4
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=4
    )

    best_dice = 0.0
    os.makedirs("outputs/checkpoints", exist_ok=True)

    for epoch in range(epochs):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device
        )

        val_metrics = validate(
            model,
            val_loader,
            criterion,
            device
        )

        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Train: {train_metrics}")
        print(f"Val:   {val_metrics}")

        if val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]
            torch.save(
                model.state_dict(),
                "outputs/checkpoints/best_multitask_deeplab.pt"
            )
            print("Saved new best model.")


if __name__ == "__main__":
    main()