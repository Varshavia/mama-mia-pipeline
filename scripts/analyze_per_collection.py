"""
analyze_per_collection.py
--------------------------
Her koleksiyon icin ayri ayri AUC hesaplar.
Wang & Hu'nun internal/external validation yapısını simüle eder.

ISPY2  → internal validation (en büyük, 980 hasta)
DUKE   → external validation 1
ISPY1  → external validation 2  
NACT   → external validation 3

Kullanim:
  python scripts/analyze_per_collection.py `
    --dce_csv "C:/Users/PC/Desktop/data/dce_curves.csv" `
    --radiomics_csv "C:/Users/PC/Desktop/data/radiomics_features.csv" `
    --fusion_probs "models/fusion/test_fusion_probs.npy" `
    --fusion_labels "models/fusion/test_labels.npy" `
    --split_csv splits/subtype/test.csv `
    --seg_csv results/all_ensemble5fold_phase1.csv `
    --output_dir results/per_collection/
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
import json

LABELS = ["Luminal_A", "Luminal_B", "HER2", "TNBC"]
LABEL2IDX = {l: i for i, l in enumerate(LABELS)}


def compute_aucs(probs, labels):
    """Macro ve per-class AUC hesapla."""
    lb = label_binarize(labels, classes=list(range(4)))
    results = {}
    try:
        results["macro"] = round(roc_auc_score(
            lb, probs, average="macro", multi_class="ovr"), 4)
    except Exception:
        results["macro"] = None

    for i, lbl in enumerate(LABELS):
        try:
            results[lbl] = round(roc_auc_score(lb[:, i], probs[:, i]), 4)
        except Exception:
            results[lbl] = None
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dce_csv", required=True)
    p.add_argument("--radiomics_csv", required=True)
    p.add_argument("--fusion_probs", required=True)
    p.add_argument("--fusion_labels", required=True)
    p.add_argument("--split_csv", required=True)
    p.add_argument("--seg_csv", required=True)
    p.add_argument("--output_dir", default="results/per_collection/")
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Test split bilgisi
    test_df = pd.read_csv(args.split_csv)
    test_df["collection"] = test_df["patient_id"].str.extract(r'^([A-Z]+\d*)_')

    # Fusion sonuçları
    fusion_probs  = np.load(args.fusion_probs)
    fusion_labels = np.load(args.fusion_labels)

    print(f"Test hasta: {len(test_df)}")
    print(f"Koleksiyonlar:\n{test_df['collection'].value_counts().to_string()}")

    # ── Segmentasyon per-collection ──────────────────────────────────────────
    seg_df = pd.read_csv(args.seg_csv)
    seg_df["collection"] = seg_df["patient_id"].str.extract(r'^([A-Z]+\d*)_')

    print(f"\n{'='*60}")
    print("SEGMENTASYON — PER-COLLECTION")
    print(f"{'='*60}")
    seg_results = {}
    for col in ["ISPY2", "DUKE", "ISPY1", "NACT"]:
        sub = seg_df[seg_df["collection"] == col]["dice"]
        if len(sub) == 0:
            continue
        seg_results[col] = {
            "n": len(sub),
            "mean_dice": round(sub.mean(), 4),
            "std_dice":  round(sub.std(), 4),
            "median_dice": round(sub.median(), 4),
        }
        print(f"  {col:8s}: N={len(sub):4d}  "
              f"Dice {sub.mean():.4f} ± {sub.std():.4f}  "
              f"(median {sub.median():.4f})")

    # ── Subtype classification per-collection ────────────────────────────────
    print(f"\n{'='*60}")
    print("SUBTYPE CLASSIFICATION — PER-COLLECTION (FUSION)")
    print(f"{'='*60}")

    col_results = {}
    for col in ["ISPY2", "DUKE", "ISPY1", "NACT"]:
        idx = test_df[test_df["collection"] == col].index
        # test_df index ile fusion_probs index eşleştir
        test_positions = [test_df.index.get_loc(i) for i in idx
                          if i in test_df.index]

        if len(test_positions) < 5:
            print(f"  {col}: yetersiz hasta ({len(test_positions)}), atlanıyor")
            continue

        col_probs  = fusion_probs[test_positions]
        col_labels = fusion_labels[test_positions]

        # Bu koleksiyonda hangi sınıflar var?
        unique_labels = np.unique(col_labels)
        if len(unique_labels) < 2:
            print(f"  {col}: tek sınıf, AUC hesaplanamıyor")
            continue

        aucs = compute_aucs(col_probs, col_labels)
        col_results[col] = {"n": len(test_positions), **aucs}

        print(f"\n  {col} (N={len(test_positions)}):")
        print(f"    Macro AUC: {aucs['macro']}")
        for lbl in LABELS:
            if aucs.get(lbl):
                print(f"    {lbl}: {aucs[lbl]}")

    # ── Tüm test seti özeti ──────────────────────────────────────────────────
    overall = compute_aucs(fusion_probs, fusion_labels)
    print(f"\n{'='*60}")
    print("GENEL TEST SONUCU (tüm koleksiyonlar)")
    print(f"{'='*60}")
    print(f"  Macro AUC: {overall['macro']}")
    for lbl in LABELS:
        print(f"  {lbl}: {overall.get(lbl)}")

    # ── Wang & Hu karşılaştırma tablosu ─────────────────────────────────────
    print(f"\n{'='*60}")
    print("WANG & HU 2025 KARŞILAŞTIRMASI")
    print(f"{'='*60}")
    print(f"{'Metrik':<20} {'Wang & Hu':>12} {'Bizim':>12}")
    print("-" * 46)

    # Segmentasyon
    our_seg = seg_df["dice"].mean()
    print(f"{'Seg Dice (ortalama)':<20} {'0.82-0.86':>12} {our_seg:.4f}")

    # Subtype classification
    wh = {
        "Luminal_A": "0.74-0.84",
        "Luminal_B": "0.68-0.72",
        "HER2":      "0.73-0.82",
        "TNBC":      "0.80-0.81",
        "Macro":     "0.7700",
    }
    for lbl in LABELS:
        our_val = overall.get(lbl, "N/A")
        print(f"  {lbl:<18} {wh[lbl]:>12} {our_val!s:>12}")
    print(f"  {'Macro AUC':<18} {wh['Macro']:>12} {overall['macro']:>12}")

    # ── Kaydet ──────────────────────────────────────────────────────────────
    final = {
        "segmentation": seg_results,
        "subtype_overall": overall,
        "subtype_per_collection": col_results,
        "wang_hu_comparison": wh,
    }
    with open(out / "per_collection_results.json", "w") as f:
        json.dump(final, f, indent=2)

    # CSV tablosu
    rows = []
    for col, r in col_results.items():
        rows.append({"collection": col, "role": "Internal" if col == "ISPY2" else "External",
                     **r})
    pd.DataFrame(rows).to_csv(out / "per_collection_table.csv", index=False)

    print(f"\nKaydedildi: {out}/")
    print(f"  per_collection_results.json")
    print(f"  per_collection_table.csv")


if __name__ == "__main__":
    main()