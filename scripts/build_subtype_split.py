"""
build_subtype_split.py
----------------------
Moleküler alt tip sınıflandırması için train/val/test split üretir.

- clinical_and_imaging_info.xlsx'ten tumor_subtype etiketini alır
- crop_manifest.csv ile kesişim yapar (crop üretilmiş hastalar)
- NaN subtype ve "luminal" belirsizlerini işler
- 70/15/15 stratified split üretir

Kullanim:
  python scripts/build_subtype_split.py `
    --clinical_excel clinical_and_imaging_info.xlsx `
    --crop_manifest "C:/Users/PC/Desktop/data/crop_manifest.csv" `
    --output_dir splits/subtype/
"""

import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


LABEL_MAP = {
    "luminal_a":        "Luminal_A",
    "luminal_b":        "Luminal_B",
    "luminal":          "Luminal_B",   # tartismali ama simdilik B
    "her2_enriched":    "HER2",
    "her2_pure":        "HER2",
    "triple_negative":  "TNBC",
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--clinical_excel", required=True)
    p.add_argument("--crop_manifest", required=True)
    p.add_argument("--output_dir", default="splits/subtype/")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Klinik Excel
    clin = pd.read_excel(args.clinical_excel, sheet_name="dataset_info")
    clin = clin[["patient_id", "tumor_subtype"]].copy()

    # Crop manifest
    crops = pd.read_csv(args.crop_manifest)

    # Birlestir
    df = crops[["patient_id"]].merge(clin, on="patient_id", how="left")

    print(f"Toplam crop: {len(df)}")
    print(f"NaN subtype: {df['tumor_subtype'].isna().sum()}")

    # NaN ve bilinmeyenleri dusur
    df = df[df["tumor_subtype"].notna()].copy()

    # 4-sinif label
    df["label"] = df["tumor_subtype"].map(LABEL_MAP)
    unmapped = df["label"].isna().sum()
    if unmapped:
        print(f"[!] Eslesemeyen {unmapped} etiket dusuruldu")
    df = df[df["label"].notna()].copy()

    # Collection (DUKE/ISPY1/ISPY2/NACT)
    df["collection"] = df["patient_id"].str.extract(r'^([A-Z]+\d*)_')

    # Stratify key
    df["strat"] = df["collection"] + "__" + df["label"]
    strat_counts = df["strat"].value_counts()
    rare = strat_counts[strat_counts < 3].index
    df.loc[df["strat"].isin(rare), "strat"] = \
        df.loc[df["strat"].isin(rare), "collection"] + "__rare"

    print(f"\nEgitime girecek hasta: {len(df)}")
    print("\nLabel dagilimi:")
    print(df["label"].value_counts().to_string())
    print("\nCollection dagilimi:")
    print(df["collection"].value_counts().to_string())

    # Split
    train_val, test = train_test_split(
        df, test_size=0.15, stratify=df["strat"], random_state=args.seed)
    train, val = train_test_split(
        train_val, test_size=0.15 / 0.85,
        stratify=train_val["strat"], random_state=args.seed)

    print(f"\nTrain: {len(train)} | Val: {len(val)} | Test: {len(test)}")

    # Label dagilimi per split
    for name, part in [("train", train), ("val", val), ("test", test)]:
        print(f"\n{name} label dagilimi:")
        print(part["label"].value_counts().to_string())

    # JSON kaydet
    split = {
        "seed": args.seed,
        "label_map": LABEL_MAP,
        "n_train": len(train), "n_val": len(val), "n_test": len(test),
        "train": sorted(train["patient_id"].tolist()),
        "val":   sorted(val["patient_id"].tolist()),
        "test":  sorted(test["patient_id"].tolist()),
    }
    with open(out / "subtype_split.json", "w") as f:
        json.dump(split, f, indent=2)

    # CSV kaydet (her split icin)
    cols = ["patient_id", "collection", "tumor_subtype", "label"]
    train[cols].to_csv(out / "train.csv", index=False)
    val[cols].to_csv(out / "val.csv", index=False)
    test[cols].to_csv(out / "test.csv", index=False)

    print(f"\nKaydedildi: {out}/")
    print("Sonraki adim: train_swinunetr.py")


if __name__ == "__main__":
    main()