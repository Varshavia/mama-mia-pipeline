"""
prepare_test50.py
-----------------
Test seti hazirlama scripti.

Ne yapiyor:
  1. splits/test.csv'den ilk 50 hastayi al
  2. Her hastanin duke_xxx_0001.nii.gz (1. post-contrast) dosyasini
  3. Hedef klasore CASE_ID_0000.nii.gz ismiyle kopyala
     (nnU-Net tek kanal bekliyor, kanal indexi 0 olmali)

Kullanim:
  python scripts/prepare_test50.py `
    --split_csv splits/test.csv `
    --images_root "C:/Users/PC/Desktop/images" `
    --target_dir "C:/Users/PC/Desktop/nnunet_predictions/test50" `
    --phase 1 `
    --n 50
"""

import argparse
import shutil
from pathlib import Path
import pandas as pd
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split_csv", required=True,
                   help="splits/test.csv dosyasi")
    p.add_argument("--images_root", required=True,
                   help="Hasta klasorlerini iceren ana dizin")
    p.add_argument("--target_dir", required=True,
                   help="Hedef klasor (nnUNet'in okuyacagi)")
    p.add_argument("--phase", type=int, default=1,
                   help="Hangi fazi kullan (0=pre, 1=1.post, 2=2.post...)")
    p.add_argument("--n", type=int, default=50,
                   help="Kac hasta alinacak")
    args = p.parse_args()
    
    images_root = Path(args.images_root)
    target_dir = Path(args.target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Hedef klasoru temizle (eski test dosyalari kalmasin)
    print(f"Hedef klasor temizleniyor: {target_dir}")
    for f in target_dir.glob("*.nii.gz"):
        f.unlink()
    
    # Test CSV'yi oku
    df = pd.read_csv(args.split_csv)
    print(f"test.csv toplam: {len(df)} hasta")
    
    # Ilk N hasta
    selected = df.head(args.n)
    print(f"Secilen: {len(selected)} hasta")
    
    copied = 0
    skipped = 0
    missing = []
    
    for _, row in selected.iterrows():
        pid = row["patient_id"]  # DUKE_001
        patient_dir = images_root / pid
        
        if not patient_dir.exists():
            skipped += 1
            missing.append(f"{pid} (klasor yok)")
            continue
        
        # duke_001_0001.nii.gz ara (kucuk harfli)
        # phase=1 -> _0001
        phase_suffix = f"_{args.phase:04d}.nii.gz"
        phase_files = [
            f for f in patient_dir.glob("*.nii.gz")
            if f.name.endswith(phase_suffix)
        ]
        
        if not phase_files:
            skipped += 1
            missing.append(f"{pid} (faz {args.phase} yok)")
            continue
        
        src = phase_files[0]
        # nnU-Net formati: CASE_ID_0000.nii.gz (kanal 0)
        dst = target_dir / f"{pid}_0000.nii.gz"
        shutil.copy2(src, dst)
        copied += 1
        if copied <= 5 or copied % 10 == 0:
            print(f"  [{copied}/{len(selected)}] {src.name} -> {dst.name}")
    
    print(f"\n{'='*60}")
    print(f"TAMAMLANDI")
    print(f"{'='*60}")
    print(f"Kopyalanan: {copied}")
    print(f"Atlanan:    {skipped}")
    if missing:
        print(f"\nEksik hastalar:")
        for m in missing[:10]:
            print(f"  {m}")
    print(f"\nHedef klasor: {target_dir}")
    print(f"Sonraki adim: run_inference_test50.py calistir")


if __name__ == "__main__":
    main()