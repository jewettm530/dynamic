import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

from echonet.datasets.echo import Echo
from echonet.modeling.multitask_deeplab import MultitaskDeepLabV3
from echonet.losses.multitask_loss import MultitaskLoss
from echonet.utils.metrics import (
    dice_coefficient,
    iou_score,
    classification_accuracy,
)
from echonet.paths import (
    DATA_DIR,
    FILE_LIST_PATH,
    MULTITASK_CHECKPOINTS_DIR,
    VIDEOS_DIR,
    VOLUME_TRACINGS_PATH,
)

class EchoMultitaskDataset(Dataset):
    """
    Wraps the existing EchoNet dataset so each item contains:
    image:
        End-diastolic frame, shape [3, H, W].
    mask:
        Binary LV segmentation mask, shape [1, H, W].
    label:
        Binary EF class:
            0 = EF >= threshold
            1 = EF < threshold
    ef:
        Original continuous EF value, retained for analysis.
    """

    def __init__(
        self,
        root: str,
        split: str,
        ef_threshold: float = 40.0,
        mean: float = 0.0,
        std: float = 1.0,
    ):
        self.ef_threshold = ef_threshold

        self.dataset = Echo(
            root=root,
            split=split,
            target_type=["LargeFrame", "LargeTrace", "EF"],
            mean=mean,
            std=std,
            length=16,
            period=2,
            clips=1,
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        _, targets = self.dataset[index]

        frame, mask, ef = targets

        image = torch.as_tensor(frame, dtype=torch.float32)
        mask = torch.as_tensor(mask, dtype=torch.float32)
        ef = torch.as_tensor(ef, dtype=torch.float32)

        # Model expects [C, H, W].
        if image.ndim != 3:
            raise ValueError(
                f"Expected image shape [C, H, W], got {tuple(image.shape)}"
            )

        # BCE segmentation loss expects [1, H, W] per sample.
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)

        # Ensure the target remains binary.
        mask = (mask > 0.5).float()

        # CrossEntropyLoss requires an integer class index.
        label = torch.tensor(
            int(float(ef) < self.ef_threshold),
            dtype=torch.long,
        )

        return {
            "image": image,
            "mask": mask,
            "label": label,
            "ef": ef,
        }

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()

    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    total_acc = 0.0

    for batch in dataloader:
        images = batch["image"].to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )

        masks = batch["mask"].to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )

        labels = batch["label"].to(
            device=device,
            dtype=torch.long,
            non_blocking=True,
        )

        optimizer.zero_grad()

        outputs = model(images)
        loss, loss_dict = criterion(outputs, masks, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_dice += dice_coefficient(
            outputs["segmentation"],
            masks,
        )
        total_iou += iou_score(
            outputs["segmentation"],
            masks,
        )
        total_acc += classification_accuracy(
            outputs["classification"],
            labels,
        )

    n = len(dataloader)

    if n == 0:
        raise RuntimeError("The DataLoader contains no batches.")

    return {
        "loss": total_loss / n,
        "dice": total_dice / n,
        "iou": total_iou / n,
        "accuracy": total_acc / n,
    }

def validate(model, dataloader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    total_acc = 0.0

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(
                device=device,
                dtype=torch.float32,
                non_blocking=True,
            )

            masks = batch["mask"].to(
                device=device,
                dtype=torch.float32,
                non_blocking=True,
            )

            labels = batch["label"].to(
                device=device,
                dtype=torch.long,
                non_blocking=True,
            )

            outputs = model(images)
            loss, loss_dict = criterion(
                outputs,
                masks,
                labels,
            )

            total_loss += loss.item()
            total_dice += dice_coefficient(
                outputs["segmentation"],
                masks,
            )
            total_iou += iou_score(
                outputs["segmentation"],
                masks,
            )
            total_acc += classification_accuracy(
                outputs["classification"],
                labels,
            )

    n = len(dataloader)

    if n == 0:
        raise RuntimeError("The DataLoader contains no batches.")

    return {
        "loss": total_loss / n,
        "dice": total_dice / n,
        "iou": total_iou / n,
        "accuracy": total_acc / n,
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

    data_root = DATA_DIR

    MULTITASK_CHECKPOINTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_checkpoint_path = (
        MULTITASK_CHECKPOINTS_DIR
        / "best_multitask_deeplab.pt"
    )

    if not data_root.exists():
        raise FileNotFoundError(
            f"Dataset directory was not found: {data_root}"
        )

    required_files = [
        FILE_LIST_PATH,
        VOLUME_TRACINGS_PATH,
        VIDEOS_DIR,
    ]

    for required_path in required_files:
        if not required_path.exists():
            raise FileNotFoundError(
                f"Required EchoNet dataset item was not found: {required_path}"
            )

    train_dataset = EchoMultitaskDataset(
        root=str(data_root),
        split="train",
        ef_threshold=40.0,
    )

    val_dataset = EchoMultitaskDataset(
        root=str(data_root),
        split="val",
        ef_threshold=40.0,
    )

    print(f"Training samples:   {len(train_dataset):,}")
    print(f"Validation samples: {len(val_dataset):,}")

    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=4,
        pin_memory=pin_memory,
        persistent_workers=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=4,
        pin_memory=pin_memory,
        persistent_workers=True,
    )

    sample_batch = next(iter(train_loader))

    print("Batch image shape:", sample_batch["image"].shape)
    print("Batch mask shape:", sample_batch["mask"].shape)
    print("Batch label shape:", sample_batch["label"].shape)
    print("Batch EF values:", sample_batch["ef"][:4])

    if sample_batch["image"].ndim != 4:
        raise ValueError(
            "Images must have batch shape [B, 3, H, W]."
        )
    if sample_batch["mask"].ndim != 4:
        raise ValueError(
            "Masks must have batch shape [B, 1, H, W]."
        )

    if sample_batch["image"].shape[1] != 3:
        raise ValueError(
            "Images must have three channels: [B, 3, H, W]."
        )
    if sample_batch["mask"].shape[1] != 1:
        raise ValueError(
            "Masks must have one channel: [B, 1, H, W]."
        )

    best_dice = 0.0

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
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_dice": best_dice,
                    "validation_metrics": val_metrics,
                    "ef_threshold": 40.0,
                },
                best_checkpoint_path,
            )

            print(f"Saved new best model to: {best_checkpoint_path}")


if __name__ == "__main__":
    main()