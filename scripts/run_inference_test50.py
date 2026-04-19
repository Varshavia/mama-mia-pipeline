"""
run_inference_test50.py
------------------------
MAMA-MIA pretrained nnU-Net ile inference + Dice hesaplama.

Ne yapiyor:
  1. nnU-Net environment variable'lari set eder
  2. nnUNetv2 Python API ile test klasorundeki 50 hastada inference yapar
     (sadece fold_0, hizli test icin)
  3. Her hasta icin expert ground truth ile Dice hesaplar
  4. Collection ve subtype bazinda breakdown yapar
  5. CSV'ye yazar

Kullanim:
  python scripts/run_inference_test50.py `
    --input_dir "C:/Users/PC/Desktop/nnunet_predictions/test50" `
    --output_dir "C:/Users/PC/Desktop/nnunet_predictions/output_phase1" `
    --expert_seg_dir "C:/Users/PC/Desktop/segmentations/expert" `
    --nnunet_results "C:/Users/PC/Desktop/nnunet_pretrained/nnUNet_results" `
    --split_csv splits/test.csv `
    --results_csv results/test50_fold0_results.csv
"""

import argparse
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import nibabel as nib
from tqdm import tqdm


def set_nnunet_env(results_root: str, raw_root: str = None, preprocessed_root: str = None):
    """nnU-Net environment variable'larini set et."""
    os.environ["nnUNet_results"] = str(results_root)
    # Inference icin diger ikisi kullanilmaz ama nnUNet import ederken gerekebilir
    os.environ["nnUNet_raw"] = str(raw_root or results_root)
    os.environ["nnUNet_preprocessed"] = str(preprocessed_root or results_root)
    print(f"[ENV] nnUNet_results = {results_root}")


def compute_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    """Binary Dice coefficient."""
    pred_b = (pred > 0).astype(np.uint8)
    gt_b = (gt > 0).astype(np.uint8)
    intersection = (pred_b & gt_b).sum()
    denom = pred_b.sum() + gt_b.sum()
    if denom == 0:
        return 1.0  # her ikisi de bos, perfect match
    return 2.0 * intersection / denom


def find_expert_seg(pid: str, expert_root: Path):
    """Case-insensitive expert segmentasyon bulucu."""
    pid_lower = pid.lower()
    for f in expert_root.glob("*.nii.gz"):
        if f.stem.replace(".nii", "").lower() == pid_lower:
            return f
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", required=True, help="CASE_ID_0000.nii.gz dosyalari")
    p.add_argument("--output_dir", required=True, help="Prediction outputlari")
    p.add_argument("--expert_seg_dir", required=True, help="Ground truth klasoru")
    p.add_argument("--nnunet_results", required=True,
                   help="nnUNet_results klasoru (Dataset105_full_image icerir)")
    p.add_argument("--split_csv", required=True, help="test.csv (collection/subtype icin)")
    p.add_argument("--results_csv", default="results/test50_fold0_results.csv")
    p.add_argument("--dataset_id", type=int, default=105)
    p.add_argument("--configuration", default="3d_fullres")
    p.add_argument("--folds", nargs="+", default=["0"],
                   help="Kullanilacak fold'lar (varsayilan: sadece 0)")
    args = p.parse_args()
    
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    expert_root = Path(args.expert_seg_dir).resolve()
    results_csv = Path(args.results_csv)
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    
    # 1. ENV setup (IMPORT'TAN ONCE!)
    set_nnunet_env(args.nnunet_results)
    
    # 2. Input klasorunu dogrula
    input_files = sorted(input_dir.glob("*_0000.nii.gz"))
    if not input_files:
        print(f"HATA: {input_dir} icinde *_0000.nii.gz bulunamadi")
        sys.exit(1)
    print(f"Input klasor: {input_dir} | {len(input_files)} dosya")
    
    # 3. Output klasoru temizle
    output_dir.mkdir(parents=True, exist_ok=True)
    for f in output_dir.glob("*.nii.gz"):
        f.unlink()
    
    # 4. nnUNetv2 import (env setlendikten sonra!)
    print("\nnnUNetv2 yukleniyor...")
    try:
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
        import torch
    except ImportError as e:
        print(f"HATA: nnunetv2 import edilemedi: {e}")
        print("cozum: pip install nnunetv2")
        sys.exit(1)
    
    print(f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU:  {torch.cuda.get_device_name(0)}")
    
    # 5. Predictor olustur
    print(f"\nPretrained model yukleniyor...")
    print(f"  Dataset: {args.dataset_id}")
    print(f"  Config:  {args.configuration}")
    print(f"  Folds:   {args.folds}")
    
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True,
    )
    
    # Model yolunu manuel kur (nnUNet_results env kullanilir)
    import nnunetv2
    from nnunetv2.utilities.file_path_utilities import get_output_folder
    model_folder = get_output_folder(
        args.dataset_id,
        "nnUNetTrainer",
        "nnUNetPlans",
        args.configuration,
    )
    print(f"  Model:   {model_folder}")
    
    folds_tuple = tuple(int(f) for f in args.folds)
    predictor.initialize_from_trained_model_folder(
        model_folder,
        use_folds=folds_tuple,
        checkpoint_name="checkpoint_final.pth",
    )
    
    # 6. Inference
    print(f"\n{'='*60}")
    print(f"INFERENCE BASLIYOR ({len(input_files)} hasta)")
    print(f"{'='*60}")
    
    predictor.predict_from_files(
        list_of_lists_or_source_folder=str(input_dir),
        output_folder_or_list_of_truncated_output_files=str(output_dir),
        save_probabilities=False,
        overwrite=True,
        num_processes_preprocessing=2,
        num_processes_segmentation_export=2,
        folder_with_segs_from_prev_stage=None,
        num_parts=1,
        part_id=0,
    )
    
    # 7. Dice hesapla
    print(f"\n{'='*60}")
    print(f"DICE HESAPLANIYOR")
    print(f"{'='*60}")
    
    split_df = pd.read_csv(args.split_csv)
    
    records = []
    pred_files = sorted(output_dir.glob("*.nii.gz"))
    
    for pred_file in tqdm(pred_files, desc="Dice"):
        pid = pred_file.stem.replace(".nii", "")  # DUKE_001
        
        # Expert seg bul
        expert_file = find_expert_seg(pid, expert_root)
        if expert_file is None:
            print(f"  [UYARI] {pid} icin expert seg bulunamadi")
            continue
        
        # Yukle
        pred = nib.load(str(pred_file)).get_fdata()
        gt = nib.load(str(expert_file)).get_fdata()
        
        # Shape kontrol
        if pred.shape != gt.shape:
            print(f"  [UYARI] {pid} shape uyumsuzlugu: pred {pred.shape}, gt {gt.shape}")
            # nnUNet cogu zaman orijinal shape'e geri getirir ama yine de ayrica resample olabilir
            continue
        
        dice = compute_dice(pred, gt)
        
        # Collection ve subtype bilgisi
        row = split_df[split_df["patient_id"] == pid]
        collection = row["collection"].iloc[0] if len(row) else "?"
        # label_4class var mi bak, yoksa tumor_subtype'a dus
        if "label_4class" in split_df.columns and len(row):
            subtype = row["label_4class"].iloc[0]
        elif "tumor_subtype" in split_df.columns and len(row):
            subtype = row["tumor_subtype"].iloc[0]
        else:
            subtype = "?"
        
        records.append({
            "patient_id": pid,
            "collection": collection,
            "subtype": subtype,
            "dice": dice,
            "pred_voxels": int((pred > 0).sum()),
            "gt_voxels": int((gt > 0).sum()),
        })
    
    # 8. Sonuclari kaydet ve ozet yaz
    results_df = pd.DataFrame(records)
    results_df.to_csv(results_csv, index=False)
    
    print(f"\n{'='*60}")
    print(f"SONUCLAR")
    print(f"{'='*60}")
    print(f"Toplam: {len(results_df)} hasta")
    print(f"\nDice istatistikleri:")
    print(f"  Mean:   {results_df['dice'].mean():.4f}")
    print(f"  Std:    {results_df['dice'].std():.4f}")
    print(f"  Median: {results_df['dice'].median():.4f}")
    print(f"  Min:    {results_df['dice'].min():.4f}")
    print(f"  Max:    {results_df['dice'].max():.4f}")
    
    print(f"\nPer-collection breakdown:")
    print(results_df.groupby("collection")["dice"].agg(["count", "mean", "std"]).to_string())
    
    if len(results_df) > 0 and "subtype" in results_df.columns:
        print(f"\nPer-subtype breakdown:")
        print(results_df.groupby("subtype")["dice"].agg(["count", "mean", "std"]).to_string())
    
    print(f"\nSonuc CSV: {results_csv}")
    print(f"\nKarsilastirma:")
    print(f"  Paper nnU-Net (full-image) Dice: 0.76 (mean)")
    print(f"  Bizim Dice:                      {results_df['dice'].mean():.4f}")


if __name__ == "__main__":
    main()