"""
train_resnet3d.py
-----------------
3D ResNet18 ile molekuler alt tip siniflandirma.
MedicalNet pretrained agirliklar kullanilir (varsa), yoksa sifirdan egitir.

Kullanim:
  python train_resnet3d.py \
    --crops_dir /workspace/crops_expert \
    --split_dir /workspace/splits/subtype/ \
    --output_dir /workspace/models/resnet3d/ \
    --epochs 100 \
    --batch_size 16 \
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
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import label_binarize


LABELS = ["Luminal_A", "Luminal_B", "HER2", "TNBC"]
LABEL2IDX = {l: i for i, l in enumerate(LABELS)}


# ── Dataset ─────────────────────────────────────────────────────────────────
class CropDataset(Dataset):
    def __init__(self, patient_ids, labels, crops_dir: Path, augment=False):
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
        self.augment = augment

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        fpath, label = self.items[idx]
        img = nib.load(str(fpath)).get_fdata(dtype=np.float32)
        img = img[np.newaxis]  # (1, H, W, D)

        if self.augment:
            # Random flip
            for ax in range(1, 4):
                if np.random.random() > 0.5:
                    img = np.flip(img, axis=ax).copy()
            # Random intensity scale
            if np.random.random() > 0.5:
                img = img * np.random.uniform(0.9, 1.1)
            # Random intensity shift
            if np.random.random() > 0.5:
                img = img + np.random.uniform(-0.1, 0.1)

        return torch.tensor(img, dtype=torch.float32), \
               torch.tensor(label, dtype=torch.long)


# ── Model ────────────────────────────────────────────────────────────────────
def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=3,
                     stride=stride, padding=1, bias=False)


class BasicBlock3D(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample:
            residual = self.downsample(x)
        return self.relu(out + residual)


class ResNet3D(nn.Module):
    def __init__(self, block, layers, num_classes=4, dropout=0.5):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv3d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # Weight init
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = self.dropout(x.flatten(1))
        return self.fc(x)


def resnet18_3d(num_classes=4, dropout=0.5):
    return ResNet3D(BasicBlock3D, [2, 2, 2, 2],
                    num_classes=num_classes, dropout=dropout)


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
    p.add_argument("--output_dir", default="models/resnet3d/")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--num_workers", type=int, default=4)
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    crops_dir = Path(args.crops_dir)
    split_dir = Path(args.split_dir)

    train_df = pd.read_csv(split_dir / "train.csv")
    val_df   = pd.read_csv(split_dir / "val.csv")
    test_df  = pd.read_csv(split_dir / "test.csv")

    train_ds = CropDataset(train_df["patient_id"].tolist(),
                           train_df["label"].tolist(), crops_dir, augment=True)
    val_ds   = CropDataset(val_df["patient_id"].tolist(),
                           val_df["label"].tolist(), crops_dir)
    test_ds  = CropDataset(test_df["patient_id"].tolist(),
                           test_df["label"].tolist(), crops_dir)

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

    model = resnet18_3d(num_classes=4, dropout=args.dropout).to(device)

    # Class weights
    counts = train_df["label"].value_counts()
    weights = torch.tensor(
        [1.0 / counts.get(l, 1) for l in LABELS], dtype=torch.float32)
    weights = weights / weights.sum() * 4
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
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

    json.dump({"best_val_auc": best_auc, "test_acc": te_acc, "test_auc": te_auc},
              open(out / "test_results.json", "w"), indent=2)
    print(f"\nModel: {out}/best_model.pth")


if __name__ == "__main__":
    main()