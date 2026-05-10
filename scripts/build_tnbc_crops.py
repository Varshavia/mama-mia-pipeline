"""
build_tnbc_crops.py
-------------------
FNH ve MEDIPOL TNBC hastalarından lezyon-merkezli 96x96x96 crop uretir.

Dosya yapisi:
  medai/1) FNH-MRI/images/Ph1/FNH_001_ph1_0000.nii.gz
  medai/1) FNH-MRI/masks_edited/editedFNH_001_ph2.nii

  medai/2) MEDIPOL-MRI/images/Ph2/MED_001_ph2.nii.gz
  medai/2) MEDIPOL-MRI/masks_edited/editedMED_001_ph2.nii

Cikti:
  data/crops_tnbc/FNH_001_image.nii.gz
  data/crops_tnbc/FNH_001_mask.nii.gz
  data/crops_tnbc/MED_001_image.nii.gz
  data/crops_tnbc/MED_001_mask.nii.gz
  data/tnbc_manifest.csv

Kullanim:
  python scripts/build_tnbc_crops.py `
    --medai_dir "C:/Users/PC/Desktop/medai" `
    --output_dir "C:/Users/PC/Desktop/data/crops_tnbc" `
    --crop_size 96
"""

import argparse
from pathlib import Path
import nibabel as nib
import numpy as np
from tqdm import tqdm
import pandas as pd


def crop_around_centroid(volume: np.ndarray, mask: np.ndarray, size: int):
    if mask.sum() == 0:
        return None, None, None
    coords = np.argwhere(mask > 0)
    centroid = coords.mean(axis=0).astype(int)
    half = size // 2
    starts = centroid - half
    ends = starts + size
    pad_before = np.maximum(-starts, 0)
    pad_after = np.maximum(ends - np.array(volume.shape), 0)
    starts_c = np.maximum(starts, 0)
    ends_c = np.minimum(ends, np.array(volume.shape))
    img_crop = volume[starts_c[0]:ends_c[0],
                      starts_c[1]:ends_c[1],
                      starts_c[2]:ends_c[2]]
    msk_crop = mask[starts_c[0]:ends_c[0],
                    starts_c[1]:ends_c[1],
                    starts_c[2]:ends_c[2]]
    pad_widths = list(zip(pad_before, pad_after))
    img_crop = np.pad(img_crop, pad_widths, mode="constant", constant_values=0)
    msk_crop = np.pad(msk_crop, pad_widths, mode="constant", constant_values=0)
    return img_crop.astype(np.float32), msk_crop.astype(np.uint8), centroid


def process_collection(collection_dir: Path, img_subdir: str,
                       img_suffix: str, mask_prefix: str,
                       pid_prefix: str, output_dir: Path,
                       crop_size: int):
    """Bir koleksiyonu isler, crop uretir."""
    img_dir  = collection_dir / "images" / img_subdir
    mask_dir = collection_dir / "masks_edited"

    records = []
    skipped = []

    img_files = sorted(img_dir.glob("*.nii.gz"))
    print(f"\n{collection_dir.name}: {len(img_files)} hasta")

    for img_file in tqdm(img_files, desc=collection_dir.name):
        # Patient ID cikart: FNH_001_ph1_0000.nii.gz -> FNH_001
        stem = img_file.name.replace(".nii.gz", "")
        parts = stem.split("_")
        pid = f"{parts[0]}_{parts[1]}"  # FNH_001 veya MED_001

        # Mask bul - esnek arama (farkli isimlendirme toleransi)
        num_padded = parts[1]             # "001"
        num_short  = str(int(parts[1]))   # "1"
        mask_file = None
        for num in [num_padded, num_short]:
            for ph in ["ph2", "ph1"]:
                for ext in [".nii", ".nii.gz"]:
                    candidate = mask_dir / f"{mask_prefix}{num}_{ph}{ext}"
                    if candidate.exists():
                        mask_file = candidate
                        break
                if mask_file: break
            if mask_file: break
        if mask_file is None:
            skipped.append((pid, "mask bulunamadi"))
            continue

        try:
            img_nib = nib.load(str(img_file))
            volume  = np.squeeze(img_nib.get_fdata(dtype=np.float32))
            seg_nib = nib.load(str(mask_file))
            mask    = seg_nib.get_fdata().astype(np.uint8)

            # Shape uyumu kontrol et
            if volume.shape != mask.shape:
                # Mask resampling gerekebilir
                skipped.append((pid, f"shape uyumsuz {volume.shape} vs {mask.shape}"))
                continue

            img_c, msk_c, centroid = crop_around_centroid(volume, mask, crop_size)
            if img_c is None:
                skipped.append((pid, "bos mask"))
                continue

            # Z-score normalize
            nz = img_c[img_c > 0]
            if len(nz) > 0:
                mu, sd = nz.mean(), nz.std() + 1e-8
                img_c = (img_c - mu) / sd

            # Affine
            new_affine = img_nib.affine.copy()
            half = crop_size // 2
            offset = centroid - half
            new_affine[:3, 3] += new_affine[:3, :3] @ offset

            nib.save(nib.Nifti1Image(img_c, new_affine),
                     str(output_dir / f"{pid}_image.nii.gz"))
            nib.save(nib.Nifti1Image(msk_c, new_affine),
                     str(output_dir / f"{pid}_mask.nii.gz"))

            coverage = msk_c.sum() / max(mask.sum(), 1)
            records.append({
                "patient_id": pid,
                "collection": pid_prefix,
                "label": "TNBC",
                "centroid_x": int(centroid[0]),
                "centroid_y": int(centroid[1]),
                "centroid_z": int(centroid[2]),
                "lesion_voxels": int(mask.sum()),
                "crop_lesion_voxels": int(msk_c.sum()),
                "coverage": round(float(coverage), 3),
            })

        except Exception as e:
            skipped.append((pid, str(e)))

    return records, skipped


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--medai_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--crop_size", type=int, default=96)
    args = p.parse_args()

    medai_dir  = Path(args.medai_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_records = []
    all_skipped = []

    # FNH koleksiyonu (Ph1 goruntu, edited mask)
    rec, skp = process_collection(
        collection_dir=medai_dir / "1) FNH-MRI",
        img_subdir="Ph1",
        img_suffix="_ph1_0000",
        mask_prefix="editedFNH_",
        pid_prefix="FNH",
        output_dir=output_dir,
        crop_size=args.crop_size,
    )
    all_records.extend(rec)
    all_skipped.extend(skp)

    # MEDIPOL koleksiyonu (Ph2 goruntu, edited mask)
    rec, skp = process_collection(
        collection_dir=medai_dir / "2) MEDIPOL-MRI",
        img_subdir="Ph2",
        img_suffix="_ph2",
        mask_prefix="editedMED_",
        pid_prefix="MED",
        output_dir=output_dir,
        crop_size=args.crop_size,
    )
    all_records.extend(rec)
    all_skipped.extend(skp)

    # Manifest kaydet
    df = pd.DataFrame(all_records)
    manifest_path = Path(args.output_dir).parent / "tnbc_manifest.csv"
    df.to_csv(manifest_path, index=False)

    print(f"\n{'='*55}")
    print(f"TAMAMLANDI")
    print(f"{'='*55}")
    print(f"Basarili: {len(all_records)} hasta")
    print(f"Atlanan:  {len(all_skipped)} hasta")
    print(f"Manifest: {manifest_path}")

    if all_records:
        cov = df["coverage"]
        print(f"\nLezyon coverage:")
        print(f"  Mean:   {cov.mean()*100:.1f}%")
        print(f"  Median: {cov.median()*100:.1f}%")
        print(f"  <80%:   {(cov < 0.8).sum()} hasta")

    if all_skipped:
        print(f"\nAtlanan hastalar:")
        for pid, reason in all_skipped:
            print(f"  {pid}: {reason}")


if __name__ == "__main__":
    main()
