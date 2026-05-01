"""
build_lesion_crops.py
---------------------
Her hasta icin lezyon merkezli 96x96x96 crop uret.

Input:
  - images/PATIENT_ID/patient_id_0001.nii.gz  (1. post-contrast faz)
  - segmentations/expert/PATIENT_ID.nii.gz    (ground truth mask)

Output (her hasta icin):
  - data/crops/PATIENT_ID_image.nii.gz   (96x96x96 lezyon merkezli)
  - data/crops/PATIENT_ID_mask.nii.gz    (ayni crop'tan mask)

Strateji:
  1. Mask'in centroid'ini bul (lezyon merkezi)
  2. Centroid etrafinda 96^3 crop al
  3. Goruntu sinirina yakinsa pad et (zero padding)
  4. Tek faz (Phase 1) kullan - SwinUNETR icin optimal

Kullanim:
  python scripts/build_lesion_crops.py `
    --images_root "C:/Users/PC/Desktop/images" `
    --expert_seg "C:/Users/PC/Desktop/segmentations/expert" `
    --output_dir "C:/Users/PC/Desktop/data/crops" `
    --phase 1 `
    --crop_size 96
"""

import argparse
from pathlib import Path
import nibabel as nib
import numpy as np
from tqdm import tqdm
import pandas as pd


def find_seg(pid: str, seg_root: Path):
    pid_l = pid.lower()
    for f in seg_root.glob("*.nii.gz"):
        if f.stem.replace(".nii", "").lower() == pid_l:
            return f
    return None


def find_phase(patient_dir: Path, phase: int):
    suffix = f"_{phase:04d}.nii.gz"
    for f in patient_dir.glob("*.nii.gz"):
        if f.name.endswith(suffix):
            return f
    return None


def crop_around_centroid(volume: np.ndarray, mask: np.ndarray, size: int):
    """
    Lezyon centroid'i etrafinda size^3 crop al.
    Sinir disindaysa zero-pad.
    Donus: cropped_image, cropped_mask, centroid (orijinal koordinat)
    """
    if mask.sum() == 0:
        return None, None, None
    
    coords = np.argwhere(mask > 0)
    centroid = coords.mean(axis=0).astype(int)
    
    half = size // 2
    
    # Crop sinirlari
    starts = centroid - half
    ends = starts + size
    
    # Pad miktarlari (negative starts veya ends > shape)
    pad_before = np.maximum(-starts, 0)
    pad_after = np.maximum(ends - np.array(volume.shape), 0)
    
    # Volume sinirlarinda crop
    starts_c = np.maximum(starts, 0)
    ends_c = np.minimum(ends, volume.shape)
    
    img_crop = volume[starts_c[0]:ends_c[0], starts_c[1]:ends_c[1], starts_c[2]:ends_c[2]]
    msk_crop = mask[starts_c[0]:ends_c[0], starts_c[1]:ends_c[1], starts_c[2]:ends_c[2]]
    
    # Pad
    pad_widths = list(zip(pad_before, pad_after))
    img_crop = np.pad(img_crop, pad_widths, mode="constant", constant_values=0)
    msk_crop = np.pad(msk_crop, pad_widths, mode="constant", constant_values=0)
    
    return img_crop.astype(np.float32), msk_crop.astype(np.uint8), centroid


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images_root", required=True)
    p.add_argument("--expert_seg", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--phase", type=int, default=1)
    p.add_argument("--crop_size", type=int, default=96)
    args = p.parse_args()
    
    images_root = Path(args.images_root)
    seg_root = Path(args.expert_seg)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    patient_dirs = sorted([d for d in images_root.iterdir() if d.is_dir()])
    print(f"Toplam hasta: {len(patient_dirs)}")
    
    records = []
    skipped = []
    
    for pdir in tqdm(patient_dirs, desc="Crops"):
        pid = pdir.name
        
        phase_file = find_phase(pdir, args.phase)
        if phase_file is None:
            skipped.append((pid, f"phase {args.phase} yok"))
            continue
        
        seg_file = find_seg(pid, seg_root)
        if seg_file is None:
            skipped.append((pid, "seg yok"))
            continue
        
        try:
            img_nib = nib.load(str(phase_file))
            volume = img_nib.get_fdata(dtype=np.float32)
            seg_nib = nib.load(str(seg_file))
            mask = seg_nib.get_fdata().astype(np.uint8)
            
            if volume.shape != mask.shape:
                skipped.append((pid, f"shape uyumsuz {volume.shape} vs {mask.shape}"))
                continue
            
            img_c, msk_c, centroid = crop_around_centroid(volume, mask, args.crop_size)
            if img_c is None:
                skipped.append((pid, "bos mask"))
                continue
            
            # Z-score normalize (sadece nonzero region uzerinde)
            nz = img_c[img_c > 0]
            if len(nz) > 0:
                mu, sd = nz.mean(), nz.std() + 1e-8
                img_c = (img_c - mu) / sd
            
            # Affine'i koru ama crop offset uygula
            new_affine = img_nib.affine.copy()
            half = args.crop_size // 2
            offset = centroid - half
            new_affine[:3, 3] += new_affine[:3, :3] @ offset
            
            nib.save(nib.Nifti1Image(img_c, new_affine), 
                     str(output_dir / f"{pid}_image.nii.gz"))
            nib.save(nib.Nifti1Image(msk_c, new_affine),
                     str(output_dir / f"{pid}_mask.nii.gz"))
            
            records.append({
                "patient_id": pid,
                "centroid_x": int(centroid[0]),
                "centroid_y": int(centroid[1]),
                "centroid_z": int(centroid[2]),
                "lesion_voxels": int((mask > 0).sum()),
                "crop_lesion_voxels": int((msk_c > 0).sum()),
            })
        except Exception as e:
            skipped.append((pid, f"hata: {e}"))
            continue
    
    # CSV kaydet
    df = pd.DataFrame(records)
    df.to_csv(output_dir.parent / "crop_manifest.csv", index=False)
    
    print(f"\n{'='*60}")
    print(f"Tamamlandi: {len(records)} hasta")
    print(f"Atlanan:    {len(skipped)} hasta")
    print(f"Manifest:   {output_dir.parent / 'crop_manifest.csv'}")
    
    if skipped:
        print(f"\nIlk 10 atlanan:")
        for pid, reason in skipped[:10]:
            print(f"  {pid}: {reason}")
    
    # Ozet stat
    if records:
        coverage = df["crop_lesion_voxels"] / df["lesion_voxels"]
        print(f"\nLezyon coverage:")
        print(f"  Mean: {coverage.mean()*100:.1f}%")
        print(f"  Median: {coverage.median()*100:.1f}%")
        print(f"  Min: {coverage.min()*100:.1f}%")
        print(f"  <80% coverage: {(coverage < 0.8).sum()} hasta (lezyon 96^3'e sigmadi)")


if __name__ == "__main__":
    main()