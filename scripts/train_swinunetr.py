"""
train_swinunetr.py
------------------
SwinUNETR encoder'i kullanarak 4-sinif molekuler alt tip siniflandirmasi.

Strateji:
  - MONAI SwinUNETR'in encoder kismini al (decoder'i at)
  - Global average pooling ile feature vector uret
  - 4-sinif linear head ekle
  - WeightedCE loss (class imbalance icin)
  - Train/val dongusunde en iyi modeli kaydet

Kullanim:
  python scripts/train_swinunetr.py `
    --crops_dir "C:/Users/PC/Desktop/data/crops_expert" `
    --split_dir splits/subtype/ `
    --output_dir models/swinunetr/ `
    --epochs 100 `
    --batch_size 4 `
    --lr 1e-4
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import pandas as pd
from monai.networks.nets import SwinUNETR
from monai.transforms import (
    Compose, RandFlipd, RandRotate90d,
    RandScaleIntensityd, RandShiftIntensityd,
    EnsureTyped,
)
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import label_binarize


LABELS = ["Luminal_A", "Luminal_B", "HER2", "TNBC"]
LABEL2IDX = {l: i for i, l in enumerate(LABELS)}


# ── Dataset ────────────────────────────────────────────────────────────────
class CropDataset(Dataset):
    def __init__(self, patient_ids, labels, crops_dir: Path, transform=None):
        self.items = []
        missing = 0
        for pid, lbl in zip(patient_ids, labels):
            f = crops_dir / f"{pid}_image.nii.gz"
            if not f.exists():
                missing += 1
                continue
            self.items.append((f, LABEL2IDX[lbl]))
        if missing:
            print(f"  [!] {missing} hasta icin crop bulunamadi")
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        fpath, label = self.items[idx]
        img = nib.load(str(fpath)).get_fdata(dtype=np.float32)
        # (H,W,D) -> (1,H,W,D)
        img = img[np.newaxis]
        sample = {"image": img, "label": label}
        if self.transform:
            sample = self.transform(sample)
        return torch.tensor(sample["image"], dtype=torch.float32), \
               torch.tensor(sample["label"], dtype=torch.long)


# ── Model ──────────────────────────────────────────────────────────────────
class SwinClassifier(nn.Module):
    def __init__(self, num_classes=4, img_size=96, dropout=0.3):
        super().__init__()
        self.swin = SwinUNETR(
            in_channels=1,
            out_channels=num_classes,   # kullanilmayacak
            feature_size=48,
            use_checkpoint=True,
        )
        # Encoder'in son feature boyutu: 48*16 = 768
        enc_dim = 768
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(enc_dim),
            nn.Dropout(dropout),
            nn.Linear(enc_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        # SwinUNETR'in encoder hidden stateleri
        hidden = self.swin.swinViT(x, self.swin.normalize)
        # Son katman: hidden[-1] shape (B, C, H, W, D)
        feat = self.pool(hidden[-1])
        return self.head(feat)


# ── Train loop ─────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, n = 0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
        correct += (logits.argmax(1) == labels).sum().item()
        n += len(labels)
    return total_loss / n, correct / n


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, n = 0, 0, 0
    all_probs, all_labels = [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss = criterion(logits, labels)
        probs = torch.softmax(logits, dim=1)
        total_loss += loss.item() * len(labels)
        correct += (logits.argmax(1) == labels).sum().item()
        n += len(labels)
        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
    probs_np = np.concatenate(all_probs)
    labels_np = np.concatenate(all_labels)
    # Macro AUC (one-vs-rest)
    try:
        labels_bin = label_binarize(labels_np, classes=list(range(4)))
        auc = roc_auc_score(labels_bin, probs_np, average="macro", multi_class="ovr")
    except Exception:
        auc = 0.0
    return total_loss / n, correct / n, auc, probs_np, labels_np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--crops_dir", required=True)
    p.add_argument("--split_dir", default="splits/subtype/")
    p.add_argument("--output_dir", default="models/swinunetr/")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--img_size", type=int, default=96)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--patience", type=int, default=20,
                   help="Early stopping: bu kadar epoch val AUC iyilesmezse dur")
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    crops_dir = Path(args.crops_dir)
    split_dir = Path(args.split_dir)

    # Split yukle
    with open(split_dir / "subtype_split.json") as f:
        split = json.load(f)

    # CSV'den label'lari al
    train_df = pd.read_csv(split_dir / "train.csv")
    val_df   = pd.read_csv(split_dir / "val.csv")
    test_df  = pd.read_csv(split_dir / "test.csv")

    # Augmentation (sadece train)
    train_transform = Compose([
        RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image"], prob=0.5, spatial_axis=2),
        RandRotate90d(keys=["image"], prob=0.5),
        RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.5),
        RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
        EnsureTyped(keys=["image"]),
    ])

    train_ds = CropDataset(
        train_df["patient_id"].tolist(), train_df["label"].tolist(),
        crops_dir, transform=train_transform)
    val_ds = CropDataset(
        val_df["patient_id"].tolist(), val_df["label"].tolist(),
        crops_dir)
    test_ds = CropDataset(
        test_df["patient_id"].tolist(), test_df["label"].tolist(),
        crops_dir)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers)
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model = SwinClassifier(
        num_classes=4,
        img_size=args.img_size,
        dropout=args.dropout
    ).to(device)

    # Class weights (imbalance icin)
    counts = train_df["label"].value_counts()
    weights = torch.tensor([
        1.0 / counts.get(l, 1) for l in LABELS
    ], dtype=torch.float32)
    weights = weights / weights.sum() * 4  # normalize
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)

    # Log
    log = []
    best_auc = 0.0
    patience_counter = 0

    print(f"\n{'Epoch':>6} {'TrLoss':>8} {'TrAcc':>7} "
          f"{'VlLoss':>8} {'VlAcc':>7} {'VlAUC':>7} {'Time':>6}")
    print("-" * 55)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        vl_loss, vl_acc, vl_auc, _, _ = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        print(f"{epoch:>6} {tr_loss:>8.4f} {tr_acc:>7.3f} "
              f"{vl_loss:>8.4f} {vl_acc:>7.3f} {vl_auc:>7.4f} {elapsed:>5.1f}s")

        log.append({
            "epoch": epoch, "tr_loss": tr_loss, "tr_acc": tr_acc,
            "vl_loss": vl_loss, "vl_acc": vl_acc, "vl_auc": vl_auc
        })

        # En iyi modeli kaydet
        if vl_auc > best_auc:
            best_auc = vl_auc
            patience_counter = 0
            torch.save(model.state_dict(), out / "best_model.pth")
            print(f"  *** Yeni en iyi: AUC {best_auc:.4f} ***")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping: {args.patience} epoch'ta iyilesme yok")
                break

    # Log kaydet
    pd.DataFrame(log).to_csv(out / "training_log.csv", index=False)

    # Test sonuclari
    print(f"\n{'='*55}")
    print("TEST SONUCLARI (en iyi model)")
    print(f"{'='*55}")
    model.load_state_dict(torch.load(out / "best_model.pth", map_location=device))
    _, te_acc, te_auc, probs, labels_np = eval_epoch(
        model, test_loader, criterion, device)

    preds = probs.argmax(1)
    print(f"Accuracy: {te_acc:.4f}")
    print(f"Macro AUC: {te_auc:.4f}")
    print(f"\nPer-class raporu:")
    print(classification_report(labels_np, preds,
                                 target_names=LABELS, digits=3))

    # Per-class AUC
    labels_bin = label_binarize(labels_np, classes=list(range(4)))
    print("Per-class AUC:")
    for i, lbl in enumerate(LABELS):
        try:
            auc_i = roc_auc_score(labels_bin[:, i], probs[:, i])
            print(f"  {lbl}: {auc_i:.4f}")
        except Exception:
            print(f"  {lbl}: N/A")

    # Test sonuclarini kaydet
    result = {
        "best_val_auc": best_auc,
        "test_accuracy": te_acc,
        "test_macro_auc": te_auc,
    }
    with open(out / "test_results.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nModel: {out}/best_model.pth")
    print(f"Log:   {out}/training_log.csv")


if __name__ == "__main__":
    main()