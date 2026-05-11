"""
fusion_binary.py
----------------
TNBC vs non-TNBC ikili siniflandirma.
Mevcut LSTM + Radiomics olasiliklarini kullanir, binary'ye donusturur.

Kullanim:
  python scripts/fusion_binary.py `
    --dce_csv "C:/Users/PC/Desktop/data/dce_curves.csv" `
    --radiomics_csv "C:/Users/PC/Desktop/data/radiomics_features_v2.csv" `
    --lstm_model models/lstm/best_model.pth `
    --output_dir models/fusion_binary/
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (roc_auc_score, classification_report,
                              confusion_matrix, roc_curve)
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier


LABELS = ["Luminal_A", "Luminal_B", "HER2", "TNBC"]
LABEL2IDX = {l: i for i, l in enumerate(LABELS)}
TNBC_IDX = 3  # TNBC index


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
            self.labels.append(LABEL2IDX.get(row["label"], 0))
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
    rad_df = rad_df[rad_df["label"].isin(LABELS)].copy()
    train_df = rad_df[rad_df["split"] == "train"]
    split_df  = rad_df[rad_df["split"] == split]

    imputer = SimpleImputer(strategy="median")
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(imputer.fit_transform(train_df[feat_cols].values))
    X_split = scaler.transform(imputer.transform(split_df[feat_cols].values))

    # Binary label: TNBC=1, diger=0
    y_train = (train_df["label"] == "TNBC").astype(int).values

    rf = RandomForestClassifier(n_estimators=500, max_depth=8,
                                 random_state=42, n_jobs=-1,
                                 class_weight="balanced")
    rf.fit(X_train, y_train)

    probs = rf.predict_proba(X_split)[:, 1]  # TNBC olasılığı
    labels = (split_df["label"] == "TNBC").astype(int).values
    pids = split_df["patient_id"].values
    return probs, labels, pids


def evaluate_binary(tnbc_probs, labels, name="", threshold=0.5):
    """Binary AUC ve metrikler."""
    auc = roc_auc_score(labels, tnbc_probs)
    preds = (tnbc_probs >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0

    print(f"\n{'='*55}")
    print(f"{name}")
    print(f"{'='*55}")
    print(f"AUC:         {auc:.4f}")
    print(f"Sensitivity: {sensitivity:.4f}  (TNBC'yi TNBC deme oranı)")
    print(f"Specificity: {specificity:.4f}  (non-TNBC'yi non-TNBC deme oranı)")
    print(f"PPV:         {ppv:.4f}  (TNBC dediğimizin gerçekten TNBC olma oranı)")
    print(f"NPV:         {npv:.4f}  (non-TNBC dediğimizin gerçekten non-TNBC olma oranı)")
    print(f"\nKarışıklık Matrisi (threshold={threshold}):")
    print(f"  TP={tp} FP={fp}")
    print(f"  FN={fn} TN={tn}")
    return auc, sensitivity, specificity


def find_best_threshold(probs, labels):
    """Youden index ile en iyi threshold bul."""
    fpr, tpr, thresholds = roc_curve(labels, probs)
    youden = tpr - fpr
    best_idx = np.argmax(youden)
    return thresholds[best_idx], tpr[best_idx], 1 - fpr[best_idx]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dce_csv", required=True)
    p.add_argument("--radiomics_csv", required=True)
    p.add_argument("--lstm_model", required=True)
    p.add_argument("--output_dir", default="models/fusion_binary/")
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dce_df = pd.read_csv(args.dce_csv)
    rad_df = pd.read_csv(args.radiomics_csv)

    lstm_model = DCELSTMClassifier().to(device)
    lstm_model.load_state_dict(torch.load(
        args.lstm_model, map_location=device, weights_only=True))

    results = {}
    optimal_w = 0.5

    for split in ["val", "test"]:
        print(f"\n\n{'#'*60}")
        print(f"SPLIT: {split.upper()}")
        print(f"{'#'*60}")

        # LSTM probs
        split_dce = dce_df[(dce_df["split"] == split) &
                           (dce_df["label"].isin(LABELS))].copy()
        ds = DCEDataset(split_dce)
        loader = DataLoader(ds, batch_size=64, shuffle=False)
        lstm_4class, lstm_labels_4class = get_lstm_probs(lstm_model, loader, device)
        lstm_pids = np.array(split_dce["patient_id"].values)

        # TNBC olasılığı (4. sınıf)
        lstm_tnbc = lstm_4class[:, TNBC_IDX]
        lstm_binary_labels = (lstm_labels_4class == TNBC_IDX).astype(int)

        # RF probs (binary)
        rf_tnbc, rf_binary_labels, rf_pids = get_rf_probs(rad_df, split)

        # Ortak hastalar
        lstm_idx = {pid: i for i, pid in enumerate(lstm_pids)}
        rf_idx   = {pid: i for i, pid in enumerate(rf_pids)}
        common_pids = [p for p in lstm_pids if p in rf_idx]
        print(f"Ortak hasta: {len(common_pids)}")
        print(f"TNBC: {sum(lstm_binary_labels[lstm_idx[p]] for p in common_pids)}")
        print(f"non-TNBC: {len(common_pids) - sum(lstm_binary_labels[lstm_idx[p]] for p in common_pids)}")

        L = np.array([lstm_tnbc[lstm_idx[p]] for p in common_pids])
        R = np.array([rf_tnbc[rf_idx[p]] for p in common_pids])
        y = np.array([lstm_binary_labels[lstm_idx[p]] for p in common_pids])

        # Tek modeller
        lstm_auc, _, _ = evaluate_binary(L, y, f"LSTM — TNBC vs non-TNBC ({split})")
        rf_auc, _, _   = evaluate_binary(R, y, f"Radiomics RF — TNBC vs non-TNBC ({split})")

        if split == "val":
            # Grid search
            print(f"\n{'='*55}")
            print("GRID SEARCH (val)")
            print(f"{'='*55}")
            best_auc, best_w = 0, 0.5
            for w in np.arange(0.1, 1.0, 0.1):
                fused = w * L + (1 - w) * R
                auc = roc_auc_score(y, fused)
                print(f"  LSTM w={w:.1f}: AUC {auc:.4f}")
                if auc > best_auc:
                    best_auc = auc
                    best_w = w
            print(f"\nEn iyi: LSTM={best_w:.1f}, RF={1-best_w:.1f}")
            optimal_w = best_w

            # En iyi threshold
            fused_val = optimal_w * L + (1 - optimal_w) * R
            best_thr, best_sens, best_spec = find_best_threshold(fused_val, y)
            print(f"En iyi threshold (Youden): {best_thr:.3f}")
            print(f"  Sensitivity: {best_sens:.4f}")
            print(f"  Specificity: {best_spec:.4f}")
        else:
            best_thr = 0.5

        # Final fusion
        fused = optimal_w * L + (1 - optimal_w) * R
        fusion_auc, sens, spec = evaluate_binary(
            fused, y,
            f"FUSION (LSTM*{optimal_w:.1f} + RF*{1-optimal_w:.1f}) — TNBC vs non-TNBC ({split})",
            threshold=best_thr
        )

        np.save(out / f"{split}_binary_probs.npy", fused)
        np.save(out / f"{split}_binary_labels.npy", y)

        results[split] = {
            "lstm_auc": float(lstm_auc),
            "rf_auc": float(rf_auc),
            "fusion_auc": float(fusion_auc),
            "sensitivity": float(sens),
            "specificity": float(spec),
            "optimal_weight": float(optimal_w),
        }

    json.dump(results, open(out / "binary_results.json", "w"), indent=2)

    print(f"\n{'='*60}")
    print("FINAL ÖZET — TNBC vs non-TNBC")
    print(f"{'='*60}")
    for split, r in results.items():
        print(f"\n{split.upper()}:")
        print(f"  LSTM:        {r['lstm_auc']:.4f}")
        print(f"  Radiomics:   {r['rf_auc']:.4f}")
        print(f"  FUSION:      {r['fusion_auc']:.4f}")
        print(f"  Sensitivity: {r['sensitivity']:.4f}")
        print(f"  Specificity: {r['specificity']:.4f}")
    print(f"\nWang & Hu TNBC AUC (4-sınıf): 0.800")
    print(f"Hedef: Binary TNBC AUC > 0.800")


if __name__ == "__main__":
    main()