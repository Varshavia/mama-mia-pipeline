"""
compare_phases.py
-----------------
Iki farkli fazin (_0001 vs _0002) inference sonuclarini kiyaslar.
Hasta bazli Dice farkini, collection/subtype breakdown'u, paired t-test yapar.

Kullanim:
  python scripts/compare_phases.py `
    --phase1_csv results/test50_fold0_results.csv `
    --phase2_csv results/test50_fold0_phase2.csv `
    --output_dir results/phase_comparison/
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase1_csv", required=True)
    p.add_argument("--phase2_csv", required=True)
    p.add_argument("--output_dir", default="results/phase_comparison")
    args = p.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Yukle
    p1 = pd.read_csv(args.phase1_csv)
    p2 = pd.read_csv(args.phase2_csv)
    
    # Isimleri netle ve merge et
    p1 = p1.rename(columns={"dice": "dice_p1", "pred_voxels": "pred_p1", "gt_voxels": "gt_p1"})
    p2 = p2.rename(columns={"dice": "dice_p2", "pred_voxels": "pred_p2", "gt_voxels": "gt_p2"})
    
    # Hasta ID uzerinden birlestir
    merged = p1.merge(p2, on=["patient_id", "collection", "subtype"], how="inner")
    
    if len(merged) == 0:
        print("HATA: Iki CSV arasinda ortak hasta bulunamadi")
        return
    
    merged["dice_diff"] = merged["dice_p2"] - merged["dice_p1"]
    merged["abs_diff"] = merged["dice_diff"].abs()
    
    # Genel ozet
    print(f"\n{'='*70}")
    print(f"FAZ KARSILASTIRMASI - {len(merged)} hasta")
    print(f"{'='*70}")
    
    print(f"\n--- OZET ---")
    print(f"Phase 1 (_0001) Mean Dice: {merged['dice_p1'].mean():.4f} (std {merged['dice_p1'].std():.4f})")
    print(f"Phase 2 (_0002) Mean Dice: {merged['dice_p2'].mean():.4f} (std {merged['dice_p2'].std():.4f})")
    print(f"Ortalama Fark (p2 - p1):   {merged['dice_diff'].mean():+.4f}")
    
    # Paired t-test (ayni hastalarda iki faz)
    t_stat, p_val = stats.ttest_rel(merged["dice_p2"], merged["dice_p1"])
    print(f"\n--- PAIRED T-TEST ---")
    print(f"t-statistic: {t_stat:.4f}")
    print(f"p-value:     {p_val:.4f}")
    if p_val < 0.05:
        winner = "Phase 2" if merged["dice_diff"].mean() > 0 else "Phase 1"
        print(f"[KARAR] {winner} istatistiksel anlamli olarak daha iyi (p<0.05)")
    else:
        print(f"[KARAR] Iki faz arasinda anlamli fark YOK (p>=0.05)")
    
    # Wilcoxon signed-rank (non-parametric, daha saglam)
    try:
        w_stat, w_p = stats.wilcoxon(merged["dice_p2"], merged["dice_p1"])
        print(f"\n--- WILCOXON SIGNED-RANK ---")
        print(f"W-statistic: {w_stat:.4f}, p-value: {w_p:.4f}")
    except Exception as e:
        print(f"\nWilcoxon atlandi: {e}")
    
    # Hasta bazinda kimde hangisi daha iyi?
    p1_better = (merged["dice_diff"] < -0.01).sum()
    p2_better = (merged["dice_diff"] > 0.01).sum()
    similar = ((merged["dice_diff"].abs() <= 0.01)).sum()
    
    print(f"\n--- HASTA BAZLI KAZANIMI ---")
    print(f"Phase 1 daha iyi (>0.01 fark):  {p1_better} hasta")
    print(f"Phase 2 daha iyi (>0.01 fark):  {p2_better} hasta")
    print(f"Esit (0.01 icinde):             {similar} hasta")
    
    # Per-collection
    print(f"\n--- PER-COLLECTION ---")
    col_stats = merged.groupby("collection").agg(
        n=("patient_id", "count"),
        p1_mean=("dice_p1", "mean"),
        p2_mean=("dice_p2", "mean"),
        diff=("dice_diff", "mean"),
    ).round(4)
    print(col_stats.to_string())
    
    # Per-subtype
    print(f"\n--- PER-SUBTYPE ---")
    sub_stats = merged.groupby("subtype").agg(
        n=("patient_id", "count"),
        p1_mean=("dice_p1", "mean"),
        p2_mean=("dice_p2", "mean"),
        diff=("dice_diff", "mean"),
    ).round(4)
    print(sub_stats.to_string())
    
    # En buyuk farklar
    print(f"\n--- EN BUYUK 5 FARK (Phase 2 lehine) ---")
    top_p2 = merged.nlargest(5, "dice_diff")[
        ["patient_id", "collection", "subtype", "dice_p1", "dice_p2", "dice_diff"]
    ]
    print(top_p2.to_string(index=False))
    
    print(f"\n--- EN BUYUK 5 FARK (Phase 1 lehine) ---")
    top_p1 = merged.nsmallest(5, "dice_diff")[
        ["patient_id", "collection", "subtype", "dice_p1", "dice_p2", "dice_diff"]
    ]
    print(top_p1.to_string(index=False))
    
    # Kaydet
    out_csv = output_dir / "phase_comparison_full.csv"
    merged.to_csv(out_csv, index=False)
    print(f"\n[OK] Tam karsilastirma CSV: {out_csv}")
    
    # Ozet markdown raporu
    summary_md = output_dir / "phase_comparison_summary.md"
    with open(summary_md, "w", encoding="utf-8") as f:
        f.write(f"# Phase 1 vs Phase 2 Karsilastirma Raporu\n\n")
        f.write(f"**Test seti:** {len(merged)} hasta\n")
        f.write(f"**Input:** Phase 1 (_0001) vs Phase 2 (_0002)\n")
        f.write(f"**Model:** MAMA-MIA pretrained nnU-Net v2 fold 0\n\n")
        f.write(f"## Ozet\n\n")
        f.write(f"| Metrik | Phase 1 | Phase 2 | Fark |\n")
        f.write(f"|---|---|---|---|\n")
        f.write(f"| Mean Dice | {merged['dice_p1'].mean():.4f} | {merged['dice_p2'].mean():.4f} | {merged['dice_diff'].mean():+.4f} |\n")
        f.write(f"| Median Dice | {merged['dice_p1'].median():.4f} | {merged['dice_p2'].median():.4f} | {merged['dice_p2'].median()-merged['dice_p1'].median():+.4f} |\n")
        f.write(f"| Std | {merged['dice_p1'].std():.4f} | {merged['dice_p2'].std():.4f} | - |\n\n")
        f.write(f"**Paired t-test:** t={t_stat:.3f}, p={p_val:.4f}\n\n")
        f.write(f"**Karar:** ")
        if p_val < 0.05:
            winner = "Phase 2" if merged["dice_diff"].mean() > 0 else "Phase 1"
            f.write(f"{winner} istatistiksel anlamli olarak daha iyi (p<0.05)\n")
        else:
            f.write(f"Iki faz arasinda anlamli fark yok (p>=0.05)\n")
        f.write(f"\n## Per-Collection\n\n")
        f.write(col_stats.to_markdown() + "\n\n")
        f.write(f"## Per-Subtype\n\n")
        f.write(sub_stats.to_markdown() + "\n")
    print(f"[OK] Markdown rapor: {summary_md}")


if __name__ == "__main__":
    main()