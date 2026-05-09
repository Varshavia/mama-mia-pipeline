"""
fusion_late.py
--------------
LSTM + Radiomics late fusion.
Her modelin test olasılıklarını alır, ağırlıklı ortalama yapar.

Kullanım:
  python scripts/fusion_late.py `
    --dce_csv "C:/Users/PC/Desktop/data/dce_curves.csv" `
    --radiomics_csv "C:/Users/PC/Desktop/data/radiomics_features.csv" `
    --lstm_model models/lstm/best_model.pth `
    --output_dir models/fusion/
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import label_binarize, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier


LABELS = ["Luminal_A", "Luminal_B", "HER2", "TNBC"]
LABEL2IDX = {l: i for i, l in enumerate(LABELS)}
IDX2LABEL = {i: l for l, i in LABEL2IDX.items()}


# ── LSTM Modeli (train_lstm.py ile aynı) ────────────────────────────────────
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
        lstm_out = hidden_size * 2
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
        out, _ = self.lstm(x)
        attn_w = self.attention(out)
        attn_w = torch.softmax(attn_w, dim=1)
        context = (out * attn_w).sum(dim=1)
        return self.head(context)


class DCEDataset(Dataset):
    def __init__(self, df):
        self.phase_cols = sorted([c for c in df.columns if c.startswith("phase_")])
        self.sequences = []
        self.labels = []
        self.pids = []
        for _, row in df.iterrows():
            seq = [0.0 if pd.isna(row[c]) else float(row[c]) for c in self.phase_cols]
            self.sequences.append(seq)
            self.labels.append(LABEL2IDX[row["label"]])
            self.pids.append(row["patient_id"])

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        x = torch.tensor(self.sequences[idx], dtype=torch.float32).unsqueeze(-1)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


# ── LSTM inference ───────────────────────────────────────────────────────────
@torch.no_grad()
def get_lstm_probs(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        all_probs.append(probs.cpu().numpy())
        all_labels.append(y.numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels)


# ── Radiomics RF inference ───────────────────────────────────────────────────
def get_rf_probs(rad_df, split):
    feat_cols = [c for c in rad_df.select_dtypes(include=["number"]).columns]
    train_df = rad_df[rad_df["split"] == "train"]
    split_df = rad_df[rad_df["split"] == split]

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_train = scaler.fit_transform(imputer.fit_transform(
        train_df[feat_cols].values))
    X_split = scaler.transform(imputer.transform(split_df[feat_cols].values))
    y_train = train_df["label"].map(LABEL2IDX).values

    counts = train_df["label"].value_counts()
    rf = RandomForestClassifier(n_estimators=500, max_depth=8,
                                 random_state=42, n_jobs=-1,
                                 class_weight="balanced")
    rf.fit(X_train, y_train)
    probs = rf.predict_proba(X_split)
    labels = split_df["label"].map(LABEL2IDX).values
    pids = split_df["patient_id"].values
    return probs, labels, pids


# ── Değerlendirme ────────────────────────────────────────────────────────────
def evaluate(probs, labels, name=""):
    lb = label_binarize(labels, classes=list(range(4)))
    macro_auc = roc_auc_score(lb, probs, average="macro", multi_class="ovr")
    preds = probs.argmax(1)
    print(f"\n{'='*55}")
    print(f"{name}")
    print(f"{'='*55}")
    print(f"Macro AUC: {macro_auc:.4f}")
    print(f"\nPer-class AUC:")
    for i, lbl in enumerate(LABELS):
        try:
            auc_i = roc_auc_score(lb[:, i], probs[:, i])
            print(f"  {lbl}: {auc_i:.4f}")
        except Exception:
            pass
    pred_labels = [IDX2LABEL[p] for p in preds]
    true_labels = [IDX2LABEL[t] for t in labels]
    print(f"\n{classification_report(true_labels, pred_labels, digits=3)}")
    return macro_auc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dce_csv", required=True)
    p.add_argument("--radiomics_csv", required=True)
    p.add_argument("--lstm_model", required=True)
    p.add_argument("--output_dir", default="models/fusion/")
    p.add_argument("--lstm_weight", type=float, default=0.5,
                   help="LSTM agirlik (radiomics = 1 - lstm_weight)")
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── DCE / LSTM ───────────────────────────────────────────────────────────
    dce_df = pd.read_csv(args.dce_csv)
    rad_df = pd.read_csv(args.radiomics_csv)

    # LSTM modelini yükle
    lstm_model = DCELSTMClassifier().to(device)
    lstm_model.load_state_dict(torch.load(
        args.lstm_model, map_location=device, weights_only=True))

    results = {}

    for split in ["val", "test"]:
        print(f"\n\n{'#'*60}")
        print(f"SPLIT: {split.upper()}")
        print(f"{'#'*60}")

        # LSTM olasılıkları
        split_dce = dce_df[dce_df["split"] == split].copy()
        # sadece 4 sınıf olan hastaları al
        split_dce = split_dce[split_dce["label"].isin(LABELS)]
        lstm_ds = DCEDataset(split_dce)
        lstm_loader = DataLoader(lstm_ds, batch_size=64, shuffle=False)
        lstm_probs, lstm_labels = get_lstm_probs(lstm_model, lstm_loader, device)
        lstm_pids = split_dce["patient_id"].values

        # Radiomics olasılıkları
        rf_probs, rf_labels, rf_pids = get_rf_probs(rad_df, split)

        # Ortak hasta ID'leri bul
        lstm_pid_set = set(lstm_pids)
        rf_pid_set = set(rf_pids)
        common_pids = sorted(lstm_pid_set & rf_pid_set)
        print(f"Ortak hasta: {len(common_pids)}")

        # Index eşleştir
        lstm_idx = {pid: i for i, pid in enumerate(lstm_pids)}
        rf_idx = {pid: i for i, pid in enumerate(rf_pids)}

        common_lstm_probs = np.array([lstm_probs[lstm_idx[p]] for p in common_pids])
        common_rf_probs   = np.array([rf_probs[rf_idx[p]] for p in common_pids])
        common_labels     = np.array([lstm_labels[lstm_idx[p]] for p in common_pids])

        # Tek model sonuçları
        lstm_auc = evaluate(common_lstm_probs, common_labels,
                            f"LSTM tek başına ({split})")
        rf_auc = evaluate(common_rf_probs, common_labels,
                          f"Radiomics RF tek başına ({split})")

        # Grid search: en iyi ağırlık bul (sadece val'da)
        if split == "val":
            print(f"\n{'='*55}")
            print("AGIRLIK GRID SEARCH (val)")
            print(f"{'='*55}")
            best_auc = 0
            best_w = 0.5
            for w in np.arange(0.1, 1.0, 0.1):
                fused = w * common_lstm_probs + (1 - w) * common_rf_probs
                lb = label_binarize(common_labels, classes=list(range(4)))
                auc = roc_auc_score(lb, fused, average="macro", multi_class="ovr")
                print(f"  LSTM w={w:.1f}: AUC {auc:.4f}")
                if auc > best_auc:
                    best_auc = auc
                    best_w = w
            print(f"\nEn iyi agirlik: LSTM={best_w:.1f}, RF={1-best_w:.1f}")
            print(f"En iyi val AUC: {best_auc:.4f}")
            optimal_w = best_w
        else:
            optimal_w = args.lstm_weight

        # Final fusion
        fused_probs = optimal_w * common_lstm_probs + (1 - optimal_w) * common_rf_probs
        fusion_auc = evaluate(fused_probs, common_labels,
                              f"FUSION (LSTM*{optimal_w:.1f} + RF*{1-optimal_w:.1f}) ({split})")

        results[split] = {
            "lstm_auc": float(lstm_auc),
            "rf_auc": float(rf_auc),
            "fusion_auc": float(fusion_auc),
            "optimal_lstm_weight": float(optimal_w),
            "n_patients": len(common_pids),
        }

        # Probları kaydet
        np.save(out / f"{split}_fusion_probs.npy", fused_probs)
        np.save(out / f"{split}_labels.npy", common_labels)

    # Sonuçları kaydet
    json.dump(results, open(out / "fusion_results.json", "w"), indent=2)
    print(f"\nKaydedildi: {out}/")
    print(f"\n{'='*55}")
    print("ÖZET")
    print(f"{'='*55}")
    for split, r in results.items():
        print(f"{split}: LSTM={r['lstm_auc']:.4f} | RF={r['rf_auc']:.4f} | FUSION={r['fusion_auc']:.4f}")
    print(f"\nWang & Hu Ensemble ResNet: 0.7700")
    print(f"Hedef: Fusion ile bu değeri yakalamak/geçmek")


if __name__ == "__main__":
    main()
