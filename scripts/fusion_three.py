"""
fusion_three.py
---------------
LSTM + Radiomics RF + 2D ResNet late fusion.
Val setinde grid search ile en iyi agirlik kombinasyonunu bulur.

Kullanim:
  python scripts/fusion_three.py `
    --dce_csv "C:/Users/PC/Desktop/data/dce_curves.csv" `
    --radiomics_csv "C:/Users/PC/Desktop/data/radiomics_features.csv" `
    --resnet2d_dir "C:/Users/PC/Desktop/data" `
    --lstm_model models/lstm/best_model.pth `
    --output_dir models/fusion3/
"""

import argparse
import json
from pathlib import Path
from itertools import product

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


# ── LSTM ─────────────────────────────────────────────────────────────────────
class DCELSTMClassifier(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_layers=2,
                 num_classes=4, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True,
                            bidirectional=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        lstm_out = hidden_size * 2
        self.attention = nn.Sequential(
            nn.Linear(lstm_out, 64), nn.Tanh(), nn.Linear(64, 1))
        self.head = nn.Sequential(
            nn.LayerNorm(lstm_out), nn.Dropout(dropout),
            nn.Linear(lstm_out, 64), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(64, num_classes))

    def forward(self, x):
        out, _ = self.lstm(x)
        attn_w = torch.softmax(self.attention(out), dim=1)
        return self.head((out * attn_w).sum(dim=1))


class DCEDataset(Dataset):
    def __init__(self, df):
        self.phase_cols = sorted([c for c in df.columns if c.startswith("phase_")])
        self.sequences, self.labels, self.pids = [], [], []
        for _, row in df.iterrows():
            seq = [0.0 if pd.isna(row[c]) else float(row[c]) for c in self.phase_cols]
            self.sequences.append(seq)
            self.labels.append(LABEL2IDX[row["label"]])
            self.pids.append(row["patient_id"])

    def __len__(self): return len(self.sequences)

    def __getitem__(self, idx):
        x = torch.tensor(self.sequences[idx], dtype=torch.float32).unsqueeze(-1)
        return x, torch.tensor(self.labels[idx], dtype=torch.long)


@torch.no_grad()
def get_lstm_probs(model, loader, device):
    model.eval()
    probs, labels = [], []
    for x, y in loader:
        p = torch.softmax(model(x.to(device)), dim=1)
        probs.append(p.cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(probs), np.concatenate(labels)


def get_rf_probs(rad_df, split):
    exclude = ["patient_id", "label", "split", "collection"]
    feat_cols = [c for c in rad_df.select_dtypes(include=["number"]).columns
                 if c not in exclude]
    rad_df = rad_df[rad_df["label"] != "unknown"].copy()
    train_df = rad_df[rad_df["split"] == "train"]
    split_df  = rad_df[rad_df["split"] == split]

    imputer = SimpleImputer(strategy="median")
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(imputer.fit_transform(train_df[feat_cols].values))
    X_split = scaler.transform(imputer.transform(split_df[feat_cols].values))
    y_train = train_df["label"].map(LABEL2IDX).values

    counts = train_df["label"].value_counts()
    rf = RandomForestClassifier(n_estimators=500, max_depth=8,
                                 random_state=42, n_jobs=-1,
                                 class_weight="balanced")
    rf.fit(X_train, y_train)
    return rf.predict_proba(X_split), \
           split_df["label"].map(LABEL2IDX).values, \
           split_df["patient_id"].values


def evaluate(probs, labels, name=""):
    lb = label_binarize(labels, classes=list(range(4)))
    macro = roc_auc_score(lb, probs, average="macro", multi_class="ovr")
    preds = probs.argmax(1)
    print(f"\n{'='*55}\n{name}\n{'='*55}")
    print(f"Macro AUC: {macro:.4f}")
    print("Per-class AUC:")
    for i, lbl in enumerate(LABELS):
        try:
            print(f"  {lbl}: {roc_auc_score(lb[:, i], probs[:, i]):.4f}")
        except Exception:
            pass
    pred_labels = [IDX2LABEL[p] for p in preds]
    true_labels = [IDX2LABEL[t] for t in labels]
    print(classification_report(true_labels, pred_labels, digits=3))
    return macro


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dce_csv", required=True)
    p.add_argument("--radiomics_csv", required=True)
    p.add_argument("--resnet2d_dir", required=True)
    p.add_argument("--lstm_model", required=True)
    p.add_argument("--output_dir", default="models/fusion3/")
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    resnet_dir = Path(args.resnet2d_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Veri yükle ───────────────────────────────────────────────────────────
    dce_df = pd.read_csv(args.dce_csv)
    rad_df = pd.read_csv(args.radiomics_csv)

    # 2D ResNet probs (RunPod'dan indirilmiş)
    rn2d_test_probs  = np.load(resnet_dir / "resnet2d_test_probs.npy")
    rn2d_val_probs   = np.load(resnet_dir / "resnet2d_val_probs.npy")
    rn2d_test_labels = np.load(resnet_dir / "resnet2d_test_labels.npy")
    rn2d_val_labels  = np.load(resnet_dir / "resnet2d_val_labels.npy")

    # LSTM
    lstm_model = DCELSTMClassifier().to(device)
    lstm_model.load_state_dict(torch.load(
        args.lstm_model, map_location=device, weights_only=True))

    results = {}
    optimal_weights = None

    for split in ["val", "test"]:
        print(f"\n\n{'#'*60}\nSPLIT: {split.upper()}\n{'#'*60}")

        # LSTM probs
        split_dce = dce_df[(dce_df["split"] == split) &
                           (dce_df["label"].isin(LABELS))].copy()
        ds = DCEDataset(split_dce)
        loader = DataLoader(ds, batch_size=64, shuffle=False)
        lstm_probs, lstm_labels = get_lstm_probs(lstm_model, loader, device)
        lstm_pids = np.array(split_dce["patient_id"].values)

        # RF probs
        rf_probs, rf_labels, rf_pids = get_rf_probs(rad_df, split)

        # 2D ResNet probs
        rn_probs  = rn2d_val_probs  if split == "val" else rn2d_test_probs
        rn_labels = rn2d_val_labels if split == "val" else rn2d_test_labels

        # Val/test split'teki hasta sırasını al (split CSV'den)
        split_csv = pd.read_csv(
            Path(args.lstm_model).parent.parent.parent / f"splits/subtype/{split}.csv")
        ordered_pids = split_csv["patient_id"].tolist()

        # LSTM index map
        lstm_idx = {pid: i for i, pid in enumerate(lstm_pids)}
        rf_idx   = {pid: i for i, pid in enumerate(rf_pids)}

        # Ortak hasta - sıralı
        common_pids = [p for p in ordered_pids
                       if p in lstm_idx and p in rf_idx]
        print(f"Ortak hasta: {len(common_pids)}")

        L = np.array([lstm_probs[lstm_idx[p]] for p in common_pids])
        R = np.array([rf_probs[rf_idx[p]]     for p in common_pids])
        N = rn_probs[:len(common_pids)]   # ResNet split ile aligned
        y = np.array([lstm_labels[lstm_idx[p]] for p in common_pids])

        # Tek model sonuçları
        evaluate(L, y, f"LSTM ({split})")
        evaluate(R, y, f"Radiomics RF ({split})")
        evaluate(N, y, f"2D ResNet ({split})")

        if split == "val":
            print(f"\n{'='*55}\nGRID SEARCH (val)\n{'='*55}")
            best_auc, best_w = 0, (0.5, 0.3, 0.2)
            weights_range = np.arange(0.1, 1.0, 0.1)
            for wl, wr in product(weights_range, repeat=2):
                wn = round(1.0 - wl - wr, 2)
                if wn <= 0 or wn >= 1:
                    continue
                fused = wl * L + wr * R + wn * N
                lb = label_binarize(y, classes=list(range(4)))
                auc = roc_auc_score(lb, fused, average="macro", multi_class="ovr")
                if auc > best_auc:
                    best_auc = auc
                    best_w = (round(wl, 2), round(wr, 2), round(wn, 2))

            print(f"En iyi: LSTM={best_w[0]}, RF={best_w[1]}, ResNet2D={best_w[2]}")
            print(f"En iyi val AUC: {best_auc:.4f}")
            optimal_weights = best_w

        wl, wr, wn = optimal_weights
        fused = wl * L + wr * R + wn * N
        fusion_auc = evaluate(fused, y,
                              f"3-MODEL FUSION (L={wl} R={wr} N={wn}) ({split})")

        np.save(out / f"{split}_fusion_probs.npy", fused)
        np.save(out / f"{split}_labels.npy", y)

        results[split] = {
            "lstm_auc": float(evaluate.__wrapped__(L, y) if hasattr(evaluate, '__wrapped__') else
                              roc_auc_score(label_binarize(y, classes=list(range(4))), L,
                                            average="macro", multi_class="ovr")),
            "rf_auc": float(roc_auc_score(label_binarize(y, classes=list(range(4))), R,
                                          average="macro", multi_class="ovr")),
            "resnet2d_auc": float(roc_auc_score(label_binarize(y, classes=list(range(4))), N,
                                                average="macro", multi_class="ovr")),
            "fusion_auc": float(fusion_auc),
            "weights": {"lstm": wl, "rf": wr, "resnet2d": wn},
        }

    json.dump(results, open(out / "fusion3_results.json", "w"), indent=2)

    print(f"\n{'='*55}\nFINAL ÖZET\n{'='*55}")
    for split, r in results.items():
        print(f"\n{split.upper()}:")
        print(f"  LSTM:     {r['lstm_auc']:.4f}")
        print(f"  RF:       {r['rf_auc']:.4f}")
        print(f"  ResNet2D: {r['resnet2d_auc']:.4f}")
        print(f"  FUSION:   {r['fusion_auc']:.4f}")
    print(f"\nWang & Hu Ensemble: 0.7700")


if __name__ == "__main__":
    main()