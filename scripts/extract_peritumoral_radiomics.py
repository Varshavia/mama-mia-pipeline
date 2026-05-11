"""
extract_peritumoral_radiomics.py
---------------------------------
Tumorun cevresindeki 10mm bolgeden (peritumoral halka) radiomics ozellik cikarir.

Yontem:
  1. Maskeyi 10mm dilate et
  2. Orijinal maske cikar -> halka bolgesi
  3. PyRadiomics ile halka bolgesinden ozellik cikar

Kullanim:
  python extract_peritumoral_radiomics.py \
    --crops_dir /workspace/crops_expert \
    --split_dir /workspace/splits/subtype/ \
    --output_csv /workspace/peritumoral_features.csv
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor
from scipy.ndimage import binary_dilation
import logging
logging.getLogger("radiomics").setLevel(logging.ERROR)

from tqdm import tqdm


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
            "normalize": True,
            "normalizeScale": 100,
        },
    }
    return featureextractor.RadiomicsFeatureExtractor(**params)


def get_peritumoral_mask(mask_arr, dilation_mm=10, voxel_spacing=1.0):
    """
    Maskeyi dilate edip orijinali cikar -> halka bolgesi.
    dilation_mm: mm cinsinden genislik
    voxel_spacing: voxel boyutu (mm)
    """
    dilation_voxels = int(dilation_mm / voxel_spacing)
    struct = np.ones((dilation_voxels*2+1,)*3, dtype=bool)

    dilated = binary_dilation(mask_arr > 0, structure=struct)
    peritumoral = dilated & ~(mask_arr > 0)
    return peritumoral.astype(np.uint8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--crops_dir", required=True)
    p.add_argument("--split_dir", default="/workspace/splits/subtype/")
    p.add_argument("--output_csv", default="/workspace/peritumoral_features.csv")
    p.add_argument("--dilation_mm", type=int, default=10)
    args = p.parse_args()

    crops_dir = Path(args.crops_dir)
    split_dir = Path(args.split_dir)

    # Tum splitlari birlestir
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

    for _, row in tqdm(all_df.iterrows(), total=len(all_df), desc="Peritumoral"):
        pid = row["patient_id"]
        img_file = crops_dir / f"{pid}_image.nii.gz"
        msk_file = crops_dir / f"{pid}_mask.nii.gz"

        if not img_file.exists() or not msk_file.exists():
            skipped.append(pid)
            continue

        try:
            img_sitk = sitk.ReadImage(str(img_file))
            msk_sitk = sitk.ReadImage(str(msk_file))

            msk_arr = sitk.GetArrayFromImage(msk_sitk)

            if msk_arr.sum() == 0:
                skipped.append(pid)
                continue

            # Peritumoral halka olustur
            peri_arr = get_peritumoral_mask(msk_arr, dilation_mm=args.dilation_mm)

            if peri_arr.sum() == 0:
                skipped.append(pid)
                continue

            # SimpleITK maskeye donustur
            peri_sitk = sitk.GetImageFromArray(peri_arr)
            peri_sitk.CopyInformation(msk_sitk)
            peri_sitk = sitk.Cast(peri_sitk, sitk.sitkInt32)

            # Radiomics cikar
            result = extractor.execute(img_sitk, peri_sitk)
            features = {}
            for key, val in result.items():
                if key.startswith("original_"):
                    try:
                        features[f"peri_{key}"] = float(val)
                    except Exception:
                        features[f"peri_{key}"] = np.nan

            features["patient_id"] = pid
            features["label"] = row["label"]
            features["split"] = row["split"]
            features["collection"] = row.get("collection",
                                              pid.split("_")[0])
            records.append(features)

        except Exception as e:
            skipped.append(pid)
            continue

    df_out = pd.DataFrame(records)
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(args.output_csv, index=False)

    print(f"\nTamamlandi: {len(records)} hasta")
    print(f"Atlanan:    {len(skipped)} hasta")
    feat_cols = [c for c in df_out.columns if c.startswith("peri_")]
    print(f"Ozellik sayisi: {len(feat_cols)}")
    print(f"Kayit: {args.output_csv}")

    # TNBC vs non-TNBC karsilastirma
    print(f"\n=== PERITUMORAL OZELLIK ORTALAMA (TNBC vs non-TNBC) ===")
    feat_cols_sample = feat_cols[:5]
    for fc in feat_cols_sample:
        tnbc_mean = df_out[df_out["label"] == "TNBC"][fc].mean()
        other_mean = df_out[df_out["label"] != "TNBC"][fc].mean()
        print(f"  {fc}: TNBC={tnbc_mean:.3f} | non-TNBC={other_mean:.3f}")


if __name__ == "__main__":
    main()