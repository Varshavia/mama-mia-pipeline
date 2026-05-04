"""
extract_dce_curves.py
---------------------
Her hasta icin DCE kinetik egrisini cikar.
LSTM'e input olacak zaman serisi: [faz0, faz1, faz2, faz3, ...]

Her faz icin:
  - Lezyon maskesi uygula
  - Lezyon icerisindeki voxellerin ortalama intensitesini al
  - Normalize et (faz0'a gore)

Cikti: data/dce_curves.csv
  patient_id, phase_0, phase_1, phase_2, phase_3, phase_4, phase_5, label

Kullanim:
  python scripts/extract_dce_curves.py `
    --images_root "C:/Users/PC/Desktop/images" `
    --expert_seg "C:/Users/PC/Desktop/segmentations/expert" `
    --split_dir splits/subtype/ `
    --output_csv "C:/Users/PC/Desktop/data/dce_curves.csv"
"""

import argparse
from pathlib import Path
import numpy as np
import nibabel as nib
import pandas as pd
from tqdm import tqdm


def find_seg(pid: str, seg_root: Path):
    pid_l = pid.lower()
    for f in seg_root.glob("*.nii.gz"):
        if f.stem.replace(".nii", "").lower() == pid_l:
            return f
    return None


def extract_curve(patient_dir: Path, seg_file: Path, max_phases: int = 6):
    """
    Lezyon maskesi icerisindeki her fazin ortalama intensitesini don.
    Eksik fazlar NaN olarak doldurulur.
    """
    phase_files = sorted(patient_dir.glob("*.nii.gz"))
    if not phase_files:
        return None

    seg = nib.load(str(seg_file)).get_fdata()
    mask = seg > 0

    if mask.sum() == 0:
        return None

    curve = []
    for i in range(max_phases):
        suffix = f"_{i:04d}.nii.gz"
        matched = [f for f in phase_files if f.name.endswith(suffix)]
        if matched:
            vol = nib.load(str(matched[0])).get_fdata(dtype=np.float32)
            if vol.shape == seg.shape:
                mean_val = float(vol[mask].mean())
            else:
                mean_val = np.nan
        else:
            mean_val = np.nan
        curve.append(mean_val)

    # Faz 0'a gore normalize et (relative enhancement)
    base = curve[0]
    if base and base != 0 and not np.isnan(base):
        curve_norm = [v / base if not np.isnan(v) else np.nan for v in curve]
    else:
        curve_norm = curve

    return curve_norm


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images_root", required=True)
    p.add_argument("--expert_seg", required=True)
    p.add_argument("--split_dir", default="splits/subtype/")
    p.add_argument("--output_csv", default="C:/Users/PC/Desktop/data/dce_curves.csv")
    p.add_argument("--max_phases", type=int, default=6)
    args = p.parse_args()

    images_root = Path(args.images_root)
    seg_root = Path(args.expert_seg)
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

    records = []
    skipped = []

    for _, row in tqdm(all_df.iterrows(), total=len(all_df), desc="DCE curves"):
        pid = row["patient_id"]
        label = row["label"]
        split = row["split"]

        patient_dir = images_root / pid
        if not patient_dir.exists():
            skipped.append(pid)
            continue

        seg_file = find_seg(pid, seg_root)
        if seg_file is None:
            skipped.append(pid)
            continue

        curve = extract_curve(patient_dir, seg_file, args.max_phases)
        if curve is None:
            skipped.append(pid)
            continue

        record = {"patient_id": pid, "label": label, "split": split}
        for i, val in enumerate(curve):
            record[f"phase_{i}"] = val
        records.append(record)

    df_out = pd.DataFrame(records)
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(args.output_csv, index=False)

    print(f"\nTamamlandi: {len(records)} hasta")
    print(f"Atlanan: {len(skipped)} hasta")
    print(f"Kayit: {args.output_csv}")

    # Ornek kinetik analiz
    print(f"\n=== KINETIK EGRI ANALIZI ===")
    phase_cols = [c for c in df_out.columns if c.startswith("phase_")]
    print(f"Faz sayisi: {len(phase_cols)}")
    for label in ["TNBC", "Luminal_A", "Luminal_B", "HER2"]:
        sub = df_out[df_out["label"] == label][phase_cols]
        means = sub.mean()
        print(f"\n{label} (N={len(sub)}):")
        for ph, val in means.items():
            print(f"  {ph}: {val:.3f}")


if __name__ == "__main__":
    main()