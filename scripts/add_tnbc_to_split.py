"""
add_tnbc_to_split.py
--------------------
57 yeni TNBC hastasini mevcut train split'ine ekler.
Test ve val degismez — sadece train buyur.

Kullanim:
  python scripts/add_tnbc_to_split.py `
    --tnbc_manifest "C:/Users/PC/Desktop/data/tnbc_manifest.csv" `
    --split_dir splits/subtype/
"""

import argparse
import json
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tnbc_manifest", required=True)
    p.add_argument("--split_dir", default="splits/subtype/")
    args = p.parse_args()

    split_dir = Path(args.split_dir)

    # Mevcut split'leri yukle
    train_df = pd.read_csv(split_dir / "train.csv")
    val_df   = pd.read_csv(split_dir / "val.csv")
    test_df  = pd.read_csv(split_dir / "test.csv")

    print(f"Mevcut train: {len(train_df)} hasta")
    print(f"Mevcut TNBC in train: {(train_df['label'] == 'TNBC').sum()}")

    # Yeni TNBC manifest
    tnbc_df = pd.read_csv(args.tnbc_manifest)
    print(f"\nYeni TNBC hasta: {len(tnbc_df)}")

    # Split formatina uyarla
    tnbc_df["label"] = "TNBC"
    tnbc_df["tumor_subtype"] = "triple_negative"

    # Sadece train'e ekle (val/test degismez)
    cols = ["patient_id", "collection", "tumor_subtype", "label"]

    # Eksik kolonlari doldur
    if "tumor_subtype" not in tnbc_df.columns:
        tnbc_df["tumor_subtype"] = "triple_negative"

    new_train = pd.concat([
        train_df,
        tnbc_df[["patient_id", "collection", "label"]].assign(
            tumor_subtype="triple_negative"
        )[cols]
    ], ignore_index=True)

    print(f"\nYeni train: {len(new_train)} hasta")
    print(f"Yeni TNBC in train: {(new_train['label'] == 'TNBC').sum()}")
    print(f"\nLabel dagilimi (train):")
    print(new_train["label"].value_counts().to_string())

    # Kaydet (yedek al)
    train_df.to_csv(split_dir / "train_backup.csv", index=False)
    new_train.to_csv(split_dir / "train.csv", index=False)

    # JSON guncelle
    with open(split_dir / "subtype_split.json") as f:
        split_json = json.load(f)

    split_json["n_train"] = len(new_train)
    split_json["train"] = sorted(new_train["patient_id"].tolist())
    split_json["tnbc_added"] = tnbc_df["patient_id"].tolist()

    with open(split_dir / "subtype_split.json", "w") as f:
        json.dump(split_json, f, indent=2)

    print(f"\nKaydedildi:")
    print(f"  splits/subtype/train.csv (yedek: train_backup.csv)")
    print(f"  splits/subtype/subtype_split.json")
    print(f"\nSonraki adim: DCE kinetik eğrisi çıkar (extract_dce_curves.py)")
    print(f"NOT: Yeni hastalar icin DCE eğrisi olmayabilir — LSTM sadece eski hastalarda")


if __name__ == "__main__":
    main()