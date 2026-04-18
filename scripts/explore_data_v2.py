"""
MAMA-MIA Dataset - Kesif ve Sanity Check Scripti (v2)
------------------------------------------------------
GERCEK dosya yapisina gore guncellenmis hali:
  images/DUKE_XXX/duke_xxx_0000.nii.gz  (Faz 0: pre-contrast)
  images/DUKE_XXX/duke_xxx_0001.nii.gz  (Faz 1: post-contrast 1)
  images/DUKE_XXX/duke_xxx_0002.nii.gz  (Faz 2: post-contrast 2)
  images/DUKE_XXX/duke_xxx_0003.nii.gz  (Faz 3: post-contrast 3)
  images/DUKE_XXX/*.tsv                  (Klinik metadata - BU COK ONEMLI)
  segmentations/expert/DUKE_XXX.nii.gz  (Ground truth)
  segmentations/automatic/duke_xxx.nii.gz (Baseline model output)

Calistirma (Windows PowerShell):
  pip install nibabel numpy matplotlib tqdm pandas
  python explore_data_v2.py `
    --images_root "C:/Users/PC/Desktop/images" `
    --expert_seg "C:/Users/PC/Desktop/segmentations/expert" `
    --auto_seg "C:/Users/PC/Desktop/segmentations/automatic"
"""

import argparse
from pathlib import Path
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import sys


def find_segmentation(patient_id: str, seg_root: Path):
    """Case-insensitive segmentasyon bulucu."""
    pid_lower = patient_id.lower()
    for f in seg_root.glob("*.nii.gz"):
        if f.stem.replace(".nii", "").lower() == pid_lower:
            return f
    return None


def inspect_single_patient(patient_dir: Path, expert_seg_root: Path,
                            auto_seg_root: Path, output_png: Path):
    """Bir hastayi detayli incele: tum fazlar + expert seg + auto seg."""
    patient_id = patient_dir.name
    print(f"\n{'='*70}")
    print(f"HASTA: {patient_id}")
    print(f"{'='*70}")
    
    # Fazlari bul
    phase_files = sorted(patient_dir.glob("*.nii.gz"))
    print(f"Faz dosyasi sayisi: {len(phase_files)}")
    
    phases = []
    for pf in phase_files:
        img = nib.load(str(pf))
        data = img.get_fdata(dtype=np.float32)
        phases.append(data)
        print(f"  {pf.name}: shape={data.shape}, "
              f"voxel={tuple(round(float(v), 3) for v in img.header.get_zooms())}, "
              f"dtype={img.get_data_dtype()}, "
              f"range=[{data.min():.0f}, {data.max():.0f}]")
    
    # TSV dosyalarini bul ve ilk satirini yazdir
    tsv_files = list(patient_dir.glob("*.tsv"))
    if tsv_files:
        print(f"\n  TSV METADATA DOSYASI: {tsv_files[0].name}")
        try:
            df = pd.read_csv(tsv_files[0], sep="\t")
            print(f"    Satir sayisi: {len(df)}")
            print(f"    Kolonlar: {list(df.columns)}")
            print(f"    Ilk satir:\n{df.head(1).T.to_string()}")
        except Exception as e:
            print(f"    [TSV okunamadi]: {e}")
    else:
        print(f"\n  [UYARI] TSV dosyasi bulunamadi")
    
    # Expert segmentasyon
    expert_file = find_segmentation(patient_id, expert_seg_root)
    expert_seg = None
    if expert_file:
        expert_img = nib.load(str(expert_file))
        expert_seg = expert_img.get_fdata()
        print(f"\n  EXPERT SEG: {expert_file.name}")
        print(f"    shape={expert_seg.shape}, unique={np.unique(expert_seg)}")
        print(f"    Lezyon voxel: {int((expert_seg > 0).sum())}, "
              f"oran: {(expert_seg > 0).sum() / expert_seg.size * 100:.4f}%")
        # Shape eslesmiyor mu?
        if expert_seg.shape != phases[0].shape:
            print(f"    [!!!] SHAPE UYUMSUZLUGU: seg {expert_seg.shape} vs faz {phases[0].shape}")
            print(f"          Segmentasyonlar resample edilmis olabilir!")
    else:
        print(f"\n  [UYARI] Expert segmentasyon bulunamadi: {patient_id}")
    
    # Automatic segmentasyon
    auto_file = find_segmentation(patient_id, auto_seg_root)
    if auto_file:
        auto_img = nib.load(str(auto_file))
        auto_seg = auto_img.get_fdata()
        print(f"\n  AUTO SEG: {auto_file.name}")
        print(f"    shape={auto_seg.shape}, lezyon voxel: {int((auto_seg > 0).sum())}")
    
    # Gorselleme - sadece expert seg ile
    if expert_seg is not None and expert_seg.shape == phases[0].shape:
        # Lezyonun oldugu slice'lari bul
        lesion_z = np.where(expert_seg.sum(axis=(0, 1)) > 0)[0]
        if len(lesion_z) > 0:
            center_z = int(np.median(lesion_z))
        else:
            center_z = phases[0].shape[2] // 2
        
        n = len(phases)
        fig, axes = plt.subplots(2, n, figsize=(4 * n, 8))
        if n == 1:
            axes = axes.reshape(2, 1)
        
        for i, phase in enumerate(phases):
            # Ust sira: saf faz
            axes[0, i].imshow(phase[:, :, center_z].T, cmap="gray", origin="lower")
            axes[0, i].set_title(f"Faz {i}")
            axes[0, i].axis("off")
            
            # Alt sira: faz + seg overlay
            axes[1, i].imshow(phase[:, :, center_z].T, cmap="gray", origin="lower")
            mask = expert_seg[:, :, center_z].T
            masked = np.ma.masked_where(mask == 0, mask)
            axes[1, i].imshow(masked, cmap="autumn", alpha=0.6, origin="lower")
            axes[1, i].set_title(f"Faz {i} + Lezyon")
            axes[1, i].axis("off")
        
        plt.suptitle(f"{patient_id} (slice z={center_z})", fontsize=14)
        plt.tight_layout()
        plt.savefig(str(output_png), dpi=100, bbox_inches="tight")
        plt.close()
        print(f"\n  Gorsellestirme: {output_png}")
    elif expert_seg is not None:
        print(f"\n  [!] Shape uyumsuzlugu nedeniyle gorsellestirme atlandi")


def dataset_statistics(images_root: Path, expert_seg_root: Path,
                        auto_seg_root: Path, output_csv: Path):
    """Tum dataset uzerinde istatistik."""
    print(f"\n{'='*70}")
    print(f"DATASET GENEL ISTATISTIKLERI")
    print(f"{'='*70}")
    
    patient_dirs = sorted([d for d in images_root.iterdir() if d.is_dir()])
    print(f"Toplam hasta klasoru: {len(patient_dirs)}")
    
    expert_files = list(expert_seg_root.glob("*.nii.gz"))
    auto_files = list(auto_seg_root.glob("*.nii.gz"))
    print(f"Expert seg dosyasi: {len(expert_files)}")
    print(f"Auto seg dosyasi: {len(auto_files)}")
    
    records = []
    for pd_ in tqdm(patient_dirs, desc="Hastalar taraniyor"):
        pid = pd_.name
        phase_files = sorted(pd_.glob("*.nii.gz"))
        n_phases = len(phase_files)
        
        shape = voxel = None
        if phase_files:
            try:
                img = nib.load(str(phase_files[0]))
                shape = img.shape
                voxel = tuple(round(float(v), 3) for v in img.header.get_zooms())
            except Exception as e:
                print(f"  [HATA] {pid}: {e}")
        
        expert = find_segmentation(pid, expert_seg_root)
        auto = find_segmentation(pid, auto_seg_root)
        tsv_files = list(pd_.glob("*.tsv"))
        
        records.append({
            "patient_id": pid,
            "n_phases": n_phases,
            "shape_x": shape[0] if shape else None,
            "shape_y": shape[1] if shape else None,
            "shape_z": shape[2] if shape else None,
            "voxel": str(voxel) if voxel else None,
            "has_expert_seg": expert is not None,
            "has_auto_seg": auto is not None,
            "has_tsv": len(tsv_files) > 0,
        })
    
    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
    print(f"\nOzet CSV: {output_csv}")
    
    print(f"\n--- Faz sayisi dagilimi ---")
    print(df["n_phases"].value_counts().sort_index().to_string())
    
    print(f"\n--- Segmentasyon durumu ---")
    print(f"Expert seg olan: {df['has_expert_seg'].sum()}/{len(df)}")
    print(f"Auto seg olan:   {df['has_auto_seg'].sum()}/{len(df)}")
    print(f"TSV olan:        {df['has_tsv'].sum()}/{len(df)}")
    
    missing = df[~df["has_expert_seg"]]["patient_id"].tolist()
    if missing:
        print(f"\n--- Expert seg EKSIK olanlar ({len(missing)}) ---")
        for m in missing[:20]:
            print(f"  {m}")
    
    # Hangi seg dosyalari hasta klasorune karsilik GELMIYOR?
    patient_ids_lower = {p.name.lower() for p in patient_dirs}
    orphan_segs = []
    for sf in expert_files:
        seg_id = sf.stem.replace(".nii", "").lower()
        if seg_id not in patient_ids_lower:
            orphan_segs.append(sf.name)
    if orphan_segs:
        print(f"\n--- ORPHAN expert seg'ler (hasta klasoru yok) ({len(orphan_segs)}) ---")
        for o in orphan_segs[:20]:
            print(f"  {o}")
    
    print(f"\n--- Shape cesitliligi ---")
    shape_combo = df[["shape_x", "shape_y", "shape_z"]].dropna().astype(int)
    if len(shape_combo) > 0:
        unique_shapes = shape_combo.apply(tuple, axis=1).value_counts()
        print(f"Farkli shape sayisi: {len(unique_shapes)}")
        print(f"En yaygin 5 shape:")
        print(unique_shapes.head().to_string())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images_root", required=True)
    p.add_argument("--expert_seg", required=True)
    p.add_argument("--auto_seg", required=True)
    p.add_argument("--sample_patient", default="DUKE_001")
    args = p.parse_args()
    
    images_root = Path(args.images_root)
    expert_seg_root = Path(args.expert_seg)
    auto_seg_root = Path(args.auto_seg)
    
    # 1. Ornek hasta
    sample = images_root / args.sample_patient
    if sample.exists():
        inspect_single_patient(sample, expert_seg_root, auto_seg_root,
                                Path("sample_patient_viz.png"))
    
    # 2. Geneldataset
    dataset_statistics(images_root, expert_seg_root, auto_seg_root,
                        Path("dataset_summary.csv"))
    
    print(f"\n{'='*70}")
    print("TAMAMLANDI. Bir sonraki adim:")
    print("  1. sample_patient_viz.png'i kontrol et (lezyon dogru yerde mi?)")
    print("  2. dataset_summary.csv'yi Excel'de ac, filtreleme yap")
    print("  3. TSV kolonlarini bana yapistir (metadata kritik)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()