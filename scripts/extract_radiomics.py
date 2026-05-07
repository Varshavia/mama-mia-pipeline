"""
extract_radiomics.py
--------------------
Her hasta icin crop + mask'ten PyRadiomics ozellikleri cikar.

Cikti: data/radiomics_features.csv
  - 107 standart radiomics ozellik
  - patient_id, label, split, collection kolonlari

Kullanim:
  python scripts/extract_radiomics.py `
    --crops_dir "C:/Users/PC/Desktop/data/crops_expert" `
    --split_dir splits/subtype/ `
    --output_csv "C:/Users/PC/Desktop/data/radiomics_features.csv" `
    --n_jobs 4
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import SimpleITK as sitk
import radiomics
from radiomics import featureextractor
import logging

# Radiomics loglarini sustur
logging.getLogger("radiomics").setLevel(logging.ERROR)


def get_extractor():
    params = {
        "imageType": {"Original": {}},
        "featureClass": {
            "shape": [],
            "firstorder": [],
            "glcm": [],
            "glrlm": [],
            "glszm": [],
            "ngtdm": [],
            "gldm": [],
        },
        "setting": {
            "binWidth": 25,
            "resampledPixelSpacing": None,
            "interpolator": "sitkBSpline",
            "normalize": True,
            "normalizeScale": 100,
        },
    }
    extractor = featureextractor.RadiomicsFeatureExtractor(**params)
    return extractor


def extract_patient(pid: str, crops_dir: Path, extractor):
    img_file = crops_dir / f"{pid}_image.nii.gz"
    msk_file = crops_dir / f"{pid}_mask.nii.gz"

    if not img_file.exists() or not msk_file.exists():
        return None

    try:
        img_sitk = sitk.ReadImage(str(img_file))
        msk_sitk = sitk.ReadImage(str(msk_file))

        # Mask binary olmali
        msk_sitk = sitk.Cast(msk_sitk > 0, sitk.sitkInt32)

        # Lezyon var mi kontrol
        arr = sitk.GetArrayFromImage(msk_sitk)
        if arr.sum() == 0:
            return None

        result = extractor.execute(img_sitk, msk_sitk)

        # Sadece sayisal ozellikleri al (diagnostics degil)
        features = {}
        for key, val in result.items():
            if key.startswith("original_"):
                try:
                    features[key] = float(val)
                except Exception:
                    features[key] = np.nan

        return features

    except Exception as e:
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--crops_dir", required=True)
    p.add_argument("--split_dir", default="splits/subtype/")
    p.add_argument("--output_csv",
                   default="C:/Users/PC/Desktop/data/radiomics_features.csv")
    p.add_argument("--n_jobs", type=int, default=1)
    args = p.parse_args()

    crops_dir = Path(args.crops_dir)
    split_dir = Path(args.split_dir)

    # Tum split'leri birlestir
    dfs = []
    for split in ["train", "val", "test"]:
        csv = split_dir / f"{split}.csv"
        if csv.exists():
            df = pd.read_csv(csv)
            df["split"] = split
            dfs.append(df)
    all_df = pd.concat(dfs, ignore_index=True)
    print(f"Toplam hasta: {len(all_df)}")

    extractor = get_extractor()

    records = []
    skipped = []

    for _, row in tqdm(all_df.iterrows(), total=len(all_df), desc="Radiomics"):
        pid = row["patient_id"]
        features = extract_patient(pid, crops_dir, extractor)

        if features is None:
            skipped.append(pid)
            continue

        record = {
            "patient_id": pid,
            "label": row["label"],
            "split": row["split"],
            "collection": row.get("collection",
                                   pid.split("_")[0]),
        }
        record.update(features)
        records.append(record)

    df_out = pd.DataFrame(records)
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(args.output_csv, index=False)

    print(f"\nTamamlandi: {len(records)} hasta")
    print(f"Atlanan:    {len(skipped)} hasta")
    print(f"Ozellik sayisi: {len([c for c in df_out.columns if c.startswith('original_')])}")
    print(f"Kayit: {args.output_csv}")

    # NaN analizi
    feat_cols = [c for c in df_out.columns if c.startswith("original_")]
    nan_pct = df_out[feat_cols].isna().mean() * 100
    high_nan = nan_pct[nan_pct > 10]
    if len(high_nan):
        print(f"\n>10% NaN olan ozellikler: {len(high_nan)}")
        print(high_nan.to_string())


if __name__ == "__main__":
    main()