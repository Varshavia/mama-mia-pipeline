"""
train_resnet2d.py
-----------------
Pretrained 2D ResNet18 ile molekuler alt tip siniflandirma.
Her hasta icin tumoru iceren 3 ardisik dilim alinir.
Wang & Hu 2025 yaklasimi.

Kullanim:
  python train_resnet2d.py \
    --crops_dir /workspace/crops_expert \
    --split_dir /workspace/splits/subtype/ \
    --output_dir /workspace/models/resnet2d/ \
    --epochs 50 \
    --batch_size 32 \
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
from torchvision import models
from torchvision.models import ResNet18_Weights
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import label_binarize


LABELS = ["Luminal_A", "Luminal_B", "HER2", "TNBC"]
LABEL2IDX = {l: i for i, l in enumerate(LABELS)}


# ── Dataset ─────────────────────────────────────────────────────────────────
class SliceDataset(Dataset):
    """
    Her hasta icin tümörün merkez dilimini ve yanındaki 2 dilimi alir.
    Giris: (3, H, W) — 3 kanal olarak stack edilmis 3 dilim.
    ResNet18 RGB girisini bekliyor, biz 3 ardisik dilimi RGB gibi veriyoruz.
    """
    def __init__(self, patient_ids, labels, crops_dir: Path,
                 img_size=96, augment=False):
        self.items = []
        missing = 0
        for pid, lbl in zip(patient_ids, labels):
            f = crops_dir / f"{pid}_image.nii.gz"
            mf = crops_dir / f"{pid}_mask.nii.gz"
            if not f.exists():
                missing += 1
                continue
            self.items.append((f, mf, LABEL2IDX[lbl]))
        if missing:
            print(f"  [!] {missing} hasta icin crop bulunamadi")
        self.img_size = img_size
        self.augment = augment

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        fpath, mpath, label = self.items[idx]

        vol = nib.load(str(fpath)).get_fdata(dtype=np.float32)  # (H, W, D)

        # Mask varsa merkez dilimi mask'ten bul, yoksa ortadan al
        if mpath.exists():
            mask = nib.load(str(mpath)).get_fdata()
            nonzero = np.where(mask > 0)
            if len(nonzero[2]) > 0:
                center_d = int(np.median(nonzero[2]))
            else:
                center_d = vol.shape[2] // 2
        else:
            center_d = vol.shape[2] // 2

        D = vol.shape[2]
        d0 = max(0, center_d - 1)
        d1 = center_d
        d2 = min(D - 1, center_d + 1)

        s0 = vol[:, :, d0]
        s1 = vol[:, :, d1]
        s2 = vol[:, :, d2]

        # Her dilimi normalize et
        def norm(s):
            mn, mx = s.min(), s.max()
            if mx - mn > 1e-8:
                return (s - mn) / (mx - mn)
            return s

        s0, s1, s2 = norm(s0), norm(s1), norm(s2)

        # (3, H, W) stack
        img = np.stack([s0, s1, s2], axis=0).astype(np.float32)

        # Augmentation
        if self.augment:
            if np.random.random() > 0.5:
                img = np.flip(img, axis=2).copy()
            if np.random.random() > 0.5:
                img = np.flip(img, axis=1).copy()
            if np.random.random() > 0.5:
                angle = np.random.uniform(-15, 15)
                from scipy.ndimage import rotate
                img = np.stack([
                    rotate(img[i], angle, reshape=False) for i in range(3)
                ])
            # Intensity jitter
            img = img * np.random.uniform(0.9, 1.1)
            img = np.clip(img, 0, 1)

        # ImageNet normalize (ResNet icin)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean[:, None, None]) / std[:, None, None]

        return torch.tensor(img, dtype=torch.float32), \
               torch.tensor(label, dtype=torch.long)


# ── Model ────────────────────────────────────────────────────────────────────
class ResNet2DClassifier(nn.Module):
    def __init__(self, num_classes=4, dropout=0.5, pretrained=True):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)
        # Son FC katmanini degistir
        in_features = backbone.fc.in_features
        backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes)
        )
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)


# ── Train / eval ─────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, n = 0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
    try:
        lb = label_binarize(labels_np, classes=list(range(4)))
        auc = roc_auc_score(lb, probs_np, average="macro", multi_class="ovr")
    except Exception:
        auc = 0.0
    return total_loss / n, correct / n, auc, probs_np, labels_np


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--crops_dir", required=True)
    p.add_argument("--split_dir", default="splits/subtype/")
    p.add_argument("--output_dir", default="models/resnet2d/")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--num_workers", type=int, default=4)
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    crops_dir = Path(args.crops_dir)
    split_dir = Path(args.split_dir)

    train_df = pd.read_csv(split_dir / "train.csv")
    val_df   = pd.read_csv(split_dir / "val.csv")
    test_df  = pd.read_csv(split_dir / "test.csv")

    train_ds = SliceDataset(
        train_df["patient_id"].tolist(), train_df["label"].tolist(),
        crops_dir, augment=True)
    val_ds = SliceDataset(
        val_df["patient_id"].tolist(), val_df["label"].tolist(),
        crops_dir)
    test_ds = SliceDataset(
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

    model = ResNet2DClassifier(
        num_classes=4, dropout=args.dropout, pretrained=True).to(device)

    # Class weights
    counts = train_df["label"].value_counts()
    weights = torch.tensor(
        [1.0 / counts.get(l, 1) for l in LABELS], dtype=torch.float32)
    weights = weights / weights.sum() * 4
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))

    # Sadece son katmanı hızlı eğit, sonra tümünü fine-tune et
    optimizer = torch.optim.AdamW([
        {"params": model.backbone.fc.parameters(), "lr": args.lr * 10},
        {"params": [p for n, p in model.backbone.named_parameters()
                    if "fc" not in n], "lr": args.lr},
    ], weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)

    log = []
    best_auc = 0.0
    patience_counter = 0

    print(f"\n{'Epoch':>6} {'TrLoss':>8} {'TrAcc':>7} "
          f"{'VlLoss':>8} {'VlAcc':>7} {'VlAUC':>7} {'Time':>6}")
    print("-" * 55)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(
            model, train_loader, optimizer, criterion, device)
        vl_loss, vl_acc, vl_auc, _, _ = eval_epoch(
            model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0

        print(f"{epoch:>6} {tr_loss:>8.4f} {tr_acc:>7.3f} "
              f"{vl_loss:>8.4f} {vl_acc:>7.3f} {vl_auc:>7.4f} {elapsed:>5.1f}s")

        log.append({"epoch": epoch, "tr_loss": tr_loss, "tr_acc": tr_acc,
                    "vl_loss": vl_loss, "vl_acc": vl_acc, "vl_auc": vl_auc})

        if vl_auc > best_auc:
            best_auc = vl_auc
            patience_counter = 0
            torch.save(model.state_dict(), out / "best_model.pth")
            print(f"  *** Yeni en iyi: AUC {best_auc:.4f} ***")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping ({args.patience} epoch)")
                break

    pd.DataFrame(log).to_csv(out / "training_log.csv", index=False)

    # Test
    print(f"\n{'='*55}")
    print("TEST SONUCLARI")
    print(f"{'='*55}")
    model.load_state_dict(torch.load(out / "best_model.pth",
                                      map_location=device,
                                      weights_only=True))
    _, te_acc, te_auc, probs, labels_np = eval_epoch(
        model, test_loader, criterion, device)
    preds = probs.argmax(1)

    print(f"Accuracy:  {te_acc:.4f}")
    print(f"Macro AUC: {te_auc:.4f}")
    print(f"\n{classification_report(labels_np, preds, target_names=LABELS, digits=3)}")

    lb = label_binarize(labels_np, classes=list(range(4)))
    print("Per-class AUC:")
    for i, lbl in enumerate(LABELS):
        try:
            auc_i = roc_auc_score(lb[:, i], probs[:, i])
            print(f"  {lbl}: {auc_i:.4f}")
        except Exception:
            print(f"  {lbl}: N/A")

    # Probları kaydet (fusion için)
    np.save(out / "test_probs.npy", probs)
    np.save(out / "test_labels.npy", labels_np)

    # Val problarını da kaydet
    _, _, _, val_probs, val_labels = eval_epoch(
        model, val_loader, criterion, device)
    np.save(out / "val_probs.npy", val_probs)
    np.save(out / "val_labels.npy", val_labels)

    json.dump({"best_val_auc": best_auc, "test_acc": te_acc,
               "test_auc": te_auc},
              open(out / "test_results.json", "w"), indent=2)

    print(f"\nModel: {out}/best_model.pth")
    print(f"Probs kaydedildi: {out}/test_probs.npy")


if __name__ == "__main__":
    main()