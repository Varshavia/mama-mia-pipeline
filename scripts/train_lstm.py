"""
train_lstm.py
-------------
DCE kinetik egrisini LSTM ile siniflandirma.

Input: her hasta icin [faz0, faz1, ..., fazN] normalize intensite degerleri
Model: Bidirectional LSTM + attention + 4-sinif head

Kullanim:
  python scripts/train_lstm.py `
    --dce_csv "C:/Users/PC/Desktop/data/dce_curves.csv" `
    --output_dir models/lstm/ `
    --epochs 100
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
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import label_binarize


LABELS = ["Luminal_A", "Luminal_B", "HER2", "TNBC"]
LABEL2IDX = {l: i for i, l in enumerate(LABELS)}


# ── Dataset ─────────────────────────────────────────────────────────────────
class DCEDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.phase_cols = sorted([c for c in df.columns if c.startswith("phase_")])
        self.sequences = []
        self.labels = []
        for _, row in df.iterrows():
            seq = []
            for col in self.phase_cols:
                val = row[col]
                seq.append(0.0 if pd.isna(val) else float(val))
            self.sequences.append(seq)
            self.labels.append(LABEL2IDX[row["label"]])

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        # (seq_len, 1) — tek feature: normalize intensite
        x = torch.tensor(self.sequences[idx], dtype=torch.float32).unsqueeze(-1)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


# ── Model ────────────────────────────────────────────────────────────────────
class DCELSTMClassifier(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_layers=2,
                 num_classes=4, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # Bidirectional -> hidden_size * 2
        lstm_out = hidden_size * 2

        # Temporal attention
        self.attention = nn.Sequential(
            nn.Linear(lstm_out, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        self.head = nn.Sequential(
            nn.LayerNorm(lstm_out),
            nn.Dropout(dropout),
            nn.Linear(lstm_out, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        # x: (B, seq_len, 1)
        out, _ = self.lstm(x)  # (B, seq_len, hidden*2)

        # Attention weights
        attn_w = self.attention(out)          # (B, seq_len, 1)
        attn_w = torch.softmax(attn_w, dim=1)
        context = (out * attn_w).sum(dim=1)  # (B, hidden*2)

        return self.head(context)


# ── Train / eval loops ───────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, n = 0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * len(y)
        correct += (logits.argmax(1) == y).sum().item()
        n += len(y)
    return total_loss / n, correct / n


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, n = 0, 0, 0
    all_probs, all_labels = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        probs = torch.softmax(logits, dim=1)
        total_loss += loss.item() * len(y)
        correct += (logits.argmax(1) == y).sum().item()
        n += len(y)
        all_probs.append(probs.cpu().numpy())
        all_labels.append(y.cpu().numpy())
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
    p.add_argument("--dce_csv", required=True)
    p.add_argument("--output_dir", default="models/lstm/")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden_size", type=int, default=128)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--patience", type=int, default=20)
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.dce_csv)
    print(f"Toplam: {len(df)} hasta")
    print(f"Label dagilimi:\n{df['label'].value_counts().to_string()}")

    train_df = df[df["split"] == "train"]
    val_df   = df[df["split"] == "val"]
    test_df  = df[df["split"] == "test"]

    train_ds = DCEDataset(train_df)
    val_ds   = DCEDataset(val_df)
    test_ds  = DCEDataset(test_df)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    print(f"Sekans uzunlugu: {len(train_ds[0][0])} faz")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = DCELSTMClassifier(
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    # Class weights
    counts = train_df["label"].value_counts()
    weights = torch.tensor(
        [1.0 / counts.get(l, 1) for l in LABELS], dtype=torch.float32)
    weights = weights / weights.sum() * 4
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5)

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
    model.load_state_dict(torch.load(out / "best_model.pth", map_location=device))
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