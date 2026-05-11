"""
train_resnet3d_binary.py
------------------------
3D ResNet18 ile TNBC vs non-TNBC ikili siniflandirma.
4-siniftan farkli: sadece 2 sinif, daha az overfitting riski.

Kullanim:
  python train_resnet3d_binary.py \
    --crops_dir /workspace/crops_expert \
    --split_dir /workspace/splits/subtype/ \
    --output_dir /workspace/models/resnet3d_binary/ \
    --epochs 100 \
    --batch_size 16 \
    --lr 3e-5
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
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix

LABELS_4 = ["Luminal_A", "Luminal_B", "HER2", "TNBC"]


# ── Dataset ──────────────────────────────────────────────────────────────────
class BinaryCropDataset(Dataset):
    def __init__(self, patient_ids, labels, crops_dir: Path, augment=False):
        self.items = []
        missing = 0
        for pid, lbl in zip(patient_ids, labels):
            f = crops_dir / f"{pid}_image.nii.gz"
            if not f.exists():
                missing += 1
                continue
            binary_label = 1 if lbl == "TNBC" else 0
            self.items.append((f, binary_label))
        if missing:
            print(f"  [!] {missing} hasta icin crop bulunamadi")
        self.augment = augment

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        fpath, label = self.items[idx]
        img = nib.load(str(fpath)).get_fdata(dtype=np.float32)[np.newaxis]

        if self.augment:
            for ax in range(1, 4):
                if np.random.random() > 0.5:
                    img = np.flip(img, axis=ax).copy()
            if np.random.random() > 0.5:
                img = img * np.random.uniform(0.9, 1.1)
            if np.random.random() > 0.5:
                img = img + np.random.uniform(-0.1, 0.1)

        return torch.tensor(img, dtype=torch.float32), \
               torch.tensor(label, dtype=torch.float32)


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


class ResNet3DBinary(nn.Module):
    def __init__(self, block, layers, dropout=0.6):
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
        self.fc = nn.Linear(512, 1)  # Binary: tek çıkış

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes))
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x); x = self.layer2(x)
        x = self.layer3(x); x = self.layer4(x)
        x = self.dropout(self.avgpool(x).flatten(1))
        return self.fc(x).squeeze(-1)


# ── Train / eval ─────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, n = 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
        n += len(labels)
    return total_loss / n


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        probs = torch.sigmoid(model(imgs))
        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.numpy())
    probs_np = np.concatenate(all_probs)
    labels_np = np.concatenate(all_labels)
    auc = roc_auc_score(labels_np, probs_np)
    return auc, probs_np, labels_np


def find_best_threshold(probs, labels):
    fpr, tpr, thresholds = roc_curve(labels, probs)
    youden = tpr - fpr
    best_idx = np.argmax(youden)
    return thresholds[best_idx], tpr[best_idx], 1 - fpr[best_idx]


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--crops_dir", required=True)
    p.add_argument("--split_dir", default="splits/subtype/")
    p.add_argument("--output_dir", default="models/resnet3d_binary/")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--dropout", type=float, default=0.6)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--num_workers", type=int, default=4)
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    crops_dir = Path(args.crops_dir)
    split_dir = Path(args.split_dir)

    train_df = pd.read_csv(split_dir / "train.csv")
    val_df   = pd.read_csv(split_dir / "val.csv")
    test_df  = pd.read_csv(split_dir / "test.csv")

    train_ds = BinaryCropDataset(train_df["patient_id"].tolist(),
                                  train_df["label"].tolist(), crops_dir, augment=True)
    val_ds   = BinaryCropDataset(val_df["patient_id"].tolist(),
                                  val_df["label"].tolist(), crops_dir)
    test_ds  = BinaryCropDataset(test_df["patient_id"].tolist(),
                                  test_df["label"].tolist(), crops_dir)

    print(f"Train: {len(train_ds)} | TNBC: {sum(1 for _, l in train_ds.items if l == 1)}")
    print(f"Val:   {len(val_ds)}   | TNBC: {sum(1 for _, l in val_ds.items if l == 1)}")
    print(f"Test:  {len(test_ds)}  | TNBC: {sum(1 for _, l in test_ds.items if l == 1)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers)
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | GPU: {torch.cuda.get_device_name(0) if device.type=='cuda' else 'N/A'}")

    model = ResNet3DBinary(BasicBlock3D, [2, 2, 2, 2], dropout=args.dropout).to(device)

    n_tnbc = sum(1 for _, l in train_ds.items if l == 1)
    n_other = len(train_ds) - n_tnbc
    pos_weight = torch.tensor([n_other / n_tnbc], dtype=torch.float32).to(device)
    print(f"pos_weight: {pos_weight.item():.2f}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(model.parameters(),
                                   lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)

    log = []
    best_auc = 0.0
    patience_counter = 0

    print(f"\n{'Epoch':>6} {'TrLoss':>8} {'VlAUC':>8} {'Time':>6}")
    print("-" * 35)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        vl_auc, _, _ = eval_epoch(model, val_loader, device)
        scheduler.step()
        elapsed = time.time() - t0

        print(f"{epoch:>6} {tr_loss:>8.4f} {vl_auc:>8.4f} {elapsed:>5.1f}s")
        log.append({"epoch": epoch, "tr_loss": tr_loss, "vl_auc": vl_auc})

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
    print("TEST SONUCLARI — TNBC vs non-TNBC")
    print(f"{'='*55}")
    model.load_state_dict(torch.load(out / "best_model.pth",
                                      map_location=device, weights_only=True))
    te_auc, probs, labels = eval_epoch(model, test_loader, device)
    best_thr, sens, spec = find_best_threshold(probs, labels)
    preds = (probs >= best_thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0

    print(f"Test AUC:    {te_auc:.4f}")
    print(f"Threshold:   {best_thr:.3f}")
    print(f"Sensitivity: {sens:.4f}")
    print(f"Specificity: {spec:.4f}")
    print(f"PPV:         {ppv:.4f}")
    print(f"NPV:         {npv:.4f}")
    print(f"TP={tp} FP={fp} FN={fn} TN={tn}")
    print(f"\nWang & Hu TNBC AUC: 0.800")
    print(f"Bizim:              {te_auc:.4f}")

    np.save(out / "test_probs.npy", probs)
    np.save(out / "test_labels.npy", labels)
    json.dump({"best_val_auc": best_auc, "test_auc": te_auc,
               "sensitivity": float(sens), "specificity": float(spec),
               "ppv": float(ppv), "npv": float(npv)},
              open(out / "results.json", "w"), indent=2)


if __name__ == "__main__":
    main()