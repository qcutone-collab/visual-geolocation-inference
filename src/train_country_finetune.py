import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms


ROOT_DIR = Path("data/geoworld")
MODEL_DIR = Path("models")
METRICS_PATH = MODEL_DIR / "country_finetune_metrics.json"
CHECKPOINT_PATH = MODEL_DIR / "country_finetune_best.pt"

SEED = 42
MIN_IMAGES_PER_COUNTRY = 60
TEST_SIZE = 0.2
BATCH_SIZE = 16
EPOCHS = 12
FREEZE_EPOCHS = 2
LR_HEAD = 3e-4
LR_ALL = 8e-5
WEIGHT_DECAY = 1e-4


@dataclass
class Sample:
    path: Path
    class_idx: int


class CountryDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample.path).convert("RGB")
        image = self.transform(image)
        return image, sample.class_idx


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collect_samples():
    class_names = []
    by_class_paths = []

    for country_dir in sorted(ROOT_DIR.iterdir()):
        if not country_dir.is_dir():
            continue

        image_paths = list(country_dir.glob("*.jpg")) + list(country_dir.glob("*.png"))
        if len(image_paths) < MIN_IMAGES_PER_COUNTRY:
            continue

        class_names.append(country_dir.name)
        by_class_paths.append(image_paths)

    return class_names, by_class_paths


def build_split(class_names, by_class_paths):
    all_paths = []
    all_labels = []
    for class_idx, paths in enumerate(by_class_paths):
        for p in paths:
            all_paths.append(p)
            all_labels.append(class_idx)

    train_paths, test_paths, train_labels, test_labels = train_test_split(
        all_paths,
        all_labels,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=all_labels,
    )

    train_samples = [Sample(path=p, class_idx=y) for p, y in zip(train_paths, train_labels)]
    test_samples = [Sample(path=p, class_idx=y) for p, y in zip(test_paths, test_labels)]
    return train_samples, test_samples


def build_model(num_classes, device):
    # Use a lighter backbone on CPU to make training practical.
    if device == "cpu":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    else:
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)
    return model


def topk_accuracy(logits, targets, k=3):
    with torch.no_grad():
        _, pred = logits.topk(k, dim=1)
        correct = pred.eq(targets.view(-1, 1)).sum().item()
        return correct / len(targets)


def evaluate(model, loader, criterion, device):
    model.eval()
    loss_total = 0.0
    top1_total = 0.0
    top3_total = 0.0
    n = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)

            batch_size = len(labels)
            loss_total += float(loss.item()) * batch_size
            top1_total += float((logits.argmax(dim=1) == labels).sum().item())
            top3_total += float(topk_accuracy(logits, labels, k=3) * batch_size)
            n += batch_size

    return {
        "loss": loss_total / max(n, 1),
        "top1": top1_total / max(n, 1),
        "top3": top3_total / max(n, 1),
    }


def train():
    set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    class_names, by_class_paths = collect_samples()
    if not class_names:
        raise RuntimeError("No classes found. Check data/geoworld.")

    train_samples, test_samples = build_split(class_names, by_class_paths)
    print("Classes:", len(class_names))
    print("Train samples:", len(train_samples))
    print("Test samples:", len(test_samples))

    train_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_ds = CountryDataset(train_samples, train_transform)
    test_ds = CountryDataset(test_samples, test_transform)

    sample_counts = np.bincount([s.class_idx for s in train_samples], minlength=len(class_names))
    class_sample_weights = 1.0 / np.maximum(sample_counts, 1)
    sample_weights = np.array([class_sample_weights[s.class_idx] for s in train_samples], dtype=np.float32)
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights),
        num_samples=len(train_samples),
        replacement=True,
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    counts = np.bincount([s.class_idx for s in train_samples], minlength=len(class_names))
    class_weights = torch.tensor((counts.sum() / np.maximum(counts, 1)).astype(np.float32), device=device)

    model = build_model(len(class_names), device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)

    for p in model.parameters():
        p.requires_grad = False
    for p in model.fc.parameters():
        p.requires_grad = True

    optimizer = torch.optim.AdamW(model.fc.parameters(), lr=LR_HEAD, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    best_top1 = -1.0
    history = []
    MODEL_DIR.mkdir(exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        if epoch == FREEZE_EPOCHS + 1:
            for p in model.parameters():
                p.requires_grad = True
            optimizer = torch.optim.AdamW(model.parameters(), lr=LR_ALL, weight_decay=WEIGHT_DECAY)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(EPOCHS - FREEZE_EPOCHS + 1))
            print("Unfroze full backbone.")

        model.train()
        running_loss = 0.0
        running_top1 = 0.0
        seen = 0

        for step, (images, labels) in enumerate(train_loader, start=1):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            bs = len(labels)
            running_loss += float(loss.item()) * bs
            running_top1 += float((logits.argmax(dim=1) == labels).sum().item())
            seen += bs
            if step % 50 == 0:
                print(
                    f"epoch {epoch}/{EPOCHS} step {step}/{len(train_loader)} "
                    f"loss {running_loss / max(seen, 1):.4f} top1 {running_top1 / max(seen, 1):.4f}"
                )

        scheduler.step()
        train_loss = running_loss / max(seen, 1)
        train_top1 = running_top1 / max(seen, 1)

        val_metrics = evaluate(model, test_loader, criterion, device)
        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 5),
            "train_top1": round(train_top1, 5),
            "val_loss": round(val_metrics["loss"], 5),
            "val_top1": round(val_metrics["top1"], 5),
            "val_top3": round(val_metrics["top3"], 5),
        }
        history.append(row)
        print(row)

        if val_metrics["top1"] > best_top1:
            best_top1 = val_metrics["top1"]
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "class_names": class_names,
                    "val_top1": best_top1,
                },
                CHECKPOINT_PATH,
            )

    final = {
        "device": device,
        "classes": len(class_names),
        "train_samples": len(train_samples),
        "test_samples": len(test_samples),
        "best_val_top1": best_top1,
        "history": history,
        "checkpoint": str(CHECKPOINT_PATH),
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2)
    print("Saved metrics:", METRICS_PATH)
    print("Best top1:", round(best_top1, 4))


if __name__ == "__main__":
    train()
