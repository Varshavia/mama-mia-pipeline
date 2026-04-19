"""
MAMA-MIA Stratified Split Builder
----------------------------------
clinical_and_imaging_info.xlsx'ten 70/15/15 train/val/test spliti oluşturur.

Strateji:
  - 6 etiketi 4 sınıfa indirger (klinik standart)
  - collection × tumor_subtype üzerinden stratified split
  - NaN subtypelı hastalar "unknown" olarak ayrı işlenir, dengeli dağıtılır
  - Seed: 42 (reproducibility)

Çalıştırma (Windows PowerShell - venv aktif olmalı):
  python scripts/build_split.py `
    --clinical "C:/Users/PC/Desktop/clinical_and_imaging_info.xlsx" `
    --output_dir "splits"

Çıktılar:
  splits/train.csv
  splits/val.csv
  splits/test.csv
  splits/split_info.json
  splits/split_stats.txt
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ── Sabitler ──────────────────────────────────────────────────────────────────

SEED = 42

# 6 ham etiket → 4 klinik sınıf
# Referans: onboarding promptu + defterdeki "4 tane subtip"
LABEL_MAP = {
    "luminal_a":      "Luminal_A",
    "luminal_b":      "Luminal_B",
    "luminal":        "Luminal_B",   # A/B ayrımı yok → B'ye ata (tam Excel gelince güncellenir)
    "her2_enriched":  "HER2",
    "her2_pure":      "HER2",        # HR-, HER2+ → HER2 sınıfı
    "triple_negative":"TNBC",
}

SPLIT_RATIOS = (0.70, 0.15, 0.15)   # train / val / test

# ── Yardımcı Fonksiyonlar ─────────────────────────────────────────────────────

def map_subtype(raw: str) -> str:
    """Ham etiketi 4 sınıftan birine veya 'unknown'a çevirir."""
    if pd.isna(raw):
        return "unknown"
    return LABEL_MAP.get(str(raw).strip().lower(), "unknown")


def collection_from_id(patient_id: str) -> str:
    """DUKE_001 → DUKE, ISPY2_100899 → ISPY2 vb."""
    return str(patient_id).split("_")[0].upper()


def stratified_3way_split(df: pd.DataFrame, strat_col: str,
                           ratios: tuple, seed: int):
    """
    Tek sütun üzerinden 3'lü stratified split.
    Önce train vs (val+test), sonra (val+test)'i ikiye böler.
    """
    train_ratio, val_ratio, test_ratio = ratios
    val_of_rest = val_ratio / (val_ratio + test_ratio)   # ≈ 0.5

    train, rest = train_test_split(
        df, test_size=(1 - train_ratio),
        stratify=df[strat_col], random_state=seed
    )
    val, test = train_test_split(
        rest, test_size=(1 - val_of_rest),
        stratify=rest[strat_col], random_state=seed
    )
    return train, val, test


def print_distribution(df: pd.DataFrame, label: str):
    """Split içindeki collection × subtype dağılımını yazdır."""
    print(f"\n{'─'*60}")
    print(f"  {label}  ({len(df)} hasta)")
    print(f"{'─'*60}")

    # Subtype dağılımı
    sub = df["subtype_4class"].value_counts().sort_index()
    print("  Subtype dağılımı:")
    for k, v in sub.items():
        pct = v / len(df) * 100
        print(f"    {k:<20} {v:>4}  ({pct:.1f}%)")

    # Collection dağılımı
    coll = df["collection"].value_counts().sort_index()
    print("  Collection dağılımı:")
    for k, v in coll.items():
        pct = v / len(df) * 100
        print(f"    {k:<20} {v:>4}  ({pct:.1f}%)")


# ── Ana Fonksiyon ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="MAMA-MIA stratified 70/15/15 split builder"
    )
    parser.add_argument(
        "--clinical",
        required=True,
        help="clinical_and_imaging_info.xlsx yolu"
    )
    parser.add_argument(
        "--output_dir",
        default="splits",
        help="Çıktı klasörü (default: splits/)"
    )
    args = parser.parse_args()

    clinical_path = Path(args.clinical)
    output_dir = Path(args.output_dir)

    # ── 1. Excel'i yükle ──────────────────────────────────────────────────────
    print(f"\nYükleniyor: {clinical_path}")
    if not clinical_path.exists():
        print(f"[HATA] Dosya bulunamadı: {clinical_path}")
        sys.exit(1)

    df = pd.read_excel(clinical_path)
    print(f"  Toplam satır: {len(df)}, kolonlar: {list(df.columns)}")

    # ── 2. Türetilmiş sütunlar ────────────────────────────────────────────────
    df["collection"]     = df["patient_id"].apply(collection_from_id)
    df["subtype_4class"] = df["tumor_subtype"].apply(map_subtype)
    df["strat_key"]      = df["collection"] + "__" + df["subtype_4class"]

    print("\nHam etiket → 4 sınıf dönüşümü:")
    mapping_summary = (
        df.groupby(["tumor_subtype", "subtype_4class"])
          .size()
          .reset_index(name="n")
    )
    print(mapping_summary.to_string(index=False))

    print(f"\nNaN subtype sayısı: {df['tumor_subtype'].isna().sum()} → 'unknown' olarak işaretlendi")

    # ── 3. unknown'ları ayır, ayrı split, sonra birleştir ────────────────────
    known   = df[df["subtype_4class"] != "unknown"].copy()
    unknown = df[df["subtype_4class"] == "unknown"].copy()

    print(f"\nBilinen subtype: {len(known)} hasta")
    print(f"Bilinmeyen subtype (NaN): {len(unknown)} hasta")

    # Bilinen: stratified split
    train_k, val_k, test_k = stratified_3way_split(
        known, "strat_key", SPLIT_RATIOS, SEED
    )

    # Bilinmeyen: collection bazında stratify yapılamaz (DUKE'ta sadece 3 kişi var)
    # → shuffle random split yeterli, zaten 26 hasta
    if len(unknown) > 0:
        unknown_shuffled = unknown.sample(frac=1, random_state=SEED).reset_index(drop=True)
        n_train = round(len(unknown_shuffled) * SPLIT_RATIOS[0])
        n_val   = round(len(unknown_shuffled) * SPLIT_RATIOS[1])
        train_u = unknown_shuffled.iloc[:n_train]
        val_u   = unknown_shuffled.iloc[n_train:n_train + n_val]
        test_u  = unknown_shuffled.iloc[n_train + n_val:]
        train = pd.concat([train_k, train_u], ignore_index=True)
        val   = pd.concat([val_k,   val_u],   ignore_index=True)
        test  = pd.concat([test_k,  test_u],  ignore_index=True)
    else:
        train, val, test = train_k, val_k, test_k

    # ── 4. Split etiketini ekle ───────────────────────────────────────────────
    train = train.copy(); train["split"] = "train"
    val   = val.copy();   val["split"]   = "val"
    test  = test.copy();  test["split"]  = "test"

    # ── 5. Dağılımları yazdır ─────────────────────────────────────────────────
    total = len(train) + len(val) + len(test)
    print(f"\n{'='*60}")
    print(f"  SPLIT SONUCU — Toplam: {total} hasta")
    print(f"  Hedef oran: {int(SPLIT_RATIOS[0]*100)}/{int(SPLIT_RATIOS[1]*100)}/{int(SPLIT_RATIOS[2]*100)}")
    print(f"  Gerçek oran: "
          f"{len(train)/total*100:.1f}/"
          f"{len(val)/total*100:.1f}/"
          f"{len(test)/total*100:.1f}")

    print_distribution(train, "TRAIN")
    print_distribution(val,   "VAL  ")
    print_distribution(test,  "TEST ")

    # ── 6. Dosyalara kaydet ───────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)

    train.to_csv(output_dir / "train.csv", index=False)
    val.to_csv(  output_dir / "val.csv",   index=False)
    test.to_csv( output_dir / "test.csv",  index=False)

    # JSON özet
    split_info = {
        "seed": SEED,
        "ratios": {"train": SPLIT_RATIOS[0], "val": SPLIT_RATIOS[1], "test": SPLIT_RATIOS[2]},
        "counts": {"train": len(train), "val": len(val), "test": len(test), "total": total},
        "label_map": LABEL_MAP,
        "train_ids": sorted(train["patient_id"].tolist()),
        "val_ids":   sorted(val["patient_id"].tolist()),
        "test_ids":  sorted(test["patient_id"].tolist()),
    }
    with open(output_dir / "split_info.json", "w", encoding="utf-8") as f:
        json.dump(split_info, f, ensure_ascii=False, indent=2)

    # İnsan okunabilir stats
    stats_lines = []
    for name, part in [("TRAIN", train), ("VAL", val), ("TEST", test)]:
        stats_lines.append(f"\n{name} ({len(part)} hasta)")
        stats_lines.append(part["subtype_4class"].value_counts().sort_index().to_string())
        stats_lines.append("")
        stats_lines.append(part["collection"].value_counts().sort_index().to_string())
        stats_lines.append("")
    (output_dir / "split_stats.txt").write_text("\n".join(stats_lines), encoding="utf-8")

    print(f"\n{'='*60}")
    print(f"  ÇIKTILAR → {output_dir}/")
    print(f"    train.csv       ({len(train)} satır)")
    print(f"    val.csv         ({len(val)} satır)")
    print(f"    test.csv        ({len(test)} satır)")
    print(f"    split_info.json (tüm ID listesi + metadata)")
    print(f"    split_stats.txt (insan okunabilir özet)")
    print(f"{'='*60}")
    print("\nBir sonraki adım:")
    print("  → Arda'dan pretrained nnU-Net ağırlıklarını iste")
    print("  → Gelince: run_baseline_inference.py ile tüm 1505 hastada Dice hesapla")


if __name__ == "__main__":
    main()
