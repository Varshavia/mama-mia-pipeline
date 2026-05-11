"""
train_lstm_binary.py
--------------------
TNBC vs non-TNBC ikili siniflandirma icin sifirdan LSTM egitimi.
4-sinifli modelden farkli: sadece TNBC mi degil mi sorusuna odaklanir.

Kullanim:
  python scripts/train_lstm_binary.py `
    --dce_csv "C:/Users/PC/Desktop/data/dce_curves.csv" `
    --output_dir models/lstm_binary/ `
    --epochs 150
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, classification_report, roc_curve
from sklearn.preprocessing import label_binarize


LABELS_4 = ["Luminal_A", "Luminal_B", "HER2", "TNBC"]


class DCEBinaryDataset(Dataset):
    def __init__(self, df):
        self.phase_cols = sorted([c for c in df.columns if c.startswith("phase_")])
        self.sequences, self.labels, self.pids = [], [], []
        for _, row in df.iterrows():
            seq = [0.0 if pd.isna(row[c]) else float(row[c]) for c in self.phase_cols]
            self.sequences.append(seq)
            # Binary label: TNBC=1, diger=0
            self.labels.append(1 if row["label"] == "TNBC" else 0)
            self.pids.append(row["patient_id"])

    def __len__(self): return len(self.sequences)

    def __getitem__(self, idx):
        x = torch.tensor(self.sequences[idx], dtype=torch.float32).unsqueeze(-1)
        return x, torch.tensor(self.labels[idx], dtype=torch.float32)


class BinaryLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True,
                            bidirectional=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        lstm_out = hidden_size * 2
        self.attention = nn.Sequential(
            nn.Linear(lstm_out, 64), nn.Tanh(), nn.Linear(64, 1))
        self.head = nn.Sequential(
            nn.LayerNorm(lstm_out),
            nn.Dropout(dropout),
            nn.Linear(lstm_out, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)  # Binary: tek çıkış
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        attn_w = torch.softmax(self.attention(out), dim=1)
        context = (out * attn_w).sum(dim=1)
        return self.head(context).squeeze(-1)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, n = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * len(y)
        n += len(y)
    return total_loss / n


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    for x, y in loader:
        x = x.to(device)
        probs = torch.sigmoid(model(x))
        all_probs.append(probs.cpu().numpy())
        all_labels.append(y.numpy())
    probs_np = np.concatenate(all_probs)
    labels_np = np.concatenate(all_labels)
    auc = roc_auc_score(labels_np, probs_np)
    return auc, probs_np, labels_np


def find_best_threshold(probs, labels):
    fpr, tpr, thresholds = roc_curve(labels, probs)
    youden = tpr - fpr
    best_idx = np.argmax(youden)
    return thresholds[best_idx], tpr[best_idx], 1 - fpr[best_idx]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dce_csv", required=True)
    p.add_argument("--output_dir", default="models/lstm_binary/")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden_size", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.4)
    p.add_argument("--patience", type=int, default=30)
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.dce_csv)
    df = df[df["label"].isin(LABELS_4)].copy()

    train_df = df[df["split"] == "train"]
    val_df   = df[df["split"] == "val"]
    test_df  = df[df["split"] == "test"]

    print(f"Train: {len(train_df)} | TNBC: {(train_df['label']=='TNBC').sum()}")
    print(f"Val:   {len(val_df)}   | TNBC: {(val_df['label']=='TNBC').sum()}")
    print(f"Test:  {len(test_df)}  | TNBC: {(test_df['label']=='TNBC').sum()}")

    train_ds = DCEBinaryDataset(train_df)
    val_ds   = DCEBinaryDataset(val_df)
    test_ds  = DCEBinaryDataset(test_df)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = BinaryLSTM(hidden_size=args.hidden_size, dropout=args.dropout).to(device)

    # TNBC az — class weight ile dengele
    n_tnbc = (train_df["label"] == "TNBC").sum()
    n_other = len(train_df) - n_tnbc
    pos_weight = torch.tensor([n_other / n_tnbc], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5)

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

    from sklearn.metrics import confusion_matrix
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0

    print(f"Test AUC:    {te_auc:.4f}")
    print(f"Threshold:   {best_thr:.3f} (Youden index)")
    print(f"Sensitivity: {sens:.4f}  (TNBC'yi TNBC deme)")
    print(f"Specificity: {spec:.4f}  (non-TNBC'yi non-TNBC deme)")
    print(f"PPV:         {ppv:.4f}")
    print(f"NPV:         {npv:.4f}")
    print(f"\nKarışıklık Matrisi:")
    print(f"  TP={tp} FP={fp}")
    print(f"  FN={fn} TN={tn}")
    print(f"\nWang & Hu TNBC AUC: 0.800")
    print(f"Bizim:              {te_auc:.4f}")

    np.save(out / "test_probs.npy", probs)
    np.save(out / "test_labels.npy", labels)
    json.dump({
        "best_val_auc": best_auc, "test_auc": te_auc,
        "sensitivity": float(sens), "specificity": float(spec),
        "ppv": float(ppv), "npv": float(npv),
        "threshold": float(best_thr)
    }, open(out / "results.json", "w"), indent=2)
    print(f"\nModel: {out}/best_model.pth")


if __name__ == "__main__":
    main()