# MAMA-MIA Breast Cancer MRI Analysis

Meme kanseri DCE-MRI görüntüleri üzerinde uçtan uca AI tanı pipeline'ı: lezyon segmentasyonu, moleküler alt tip sınıflandırması, pCR ve Onkotype tahmini.

---

## 1. Proje Özeti

Bu proje **MAMA-MIA datasetini** (Scientific Data 2025, 1506 hasta, 4 kaynaktan — DUKE, ISPY1, ISPY2, NACT) kullanarak 4 aşamalı bir klinik karar destek sistemi geliştirir:

1. **Lezyon Segmentasyonu** — MR görüntüsünden tümörü otomatik işaretle
2. **Moleküler Alt Tip Sınıflandırması** — Tümör hangi subtip? (Luminal A/B, HER2-enriched, TNBC)
3. **pCR Tahmini** — Neoadjuvan kemoterapiye patolojik tam yanıt olacak mı?
4. **Onkotype DX Skor Tahmini** — Recurrence Score öngörüsü (5000$'lık testin AI alternatifi)

**Referans:** MAMA-MIA ekibinin Nature Scientific Data 2025 paper'ı ve MICCAI 2025 Challenge.

---

## 2. Veri Durumu

### 2.1. Elimizde Olan

- **1505 hasta klasörü** — her biri 4-5 faz DCE-MRI (`DUKE_001/duke_001_0000.nii.gz` ... `_0004.nii.gz`)
- **1506 expert segmentasyonu** — radyolog tarafından çizilmiş ground truth
- **1506 automatic segmentasyonu** — MAMA-MIA baseline nnU-Net çıktısı (Dice ~0.83)
- **clinical_and_imaging_info.xlsx (filtrelenmiş)** — 1506 satır, sadece 2 kolon: `patient_id`, `tumor_subtype`

### 2.2. Eksikler (Arda'dan Alınacak)

- **Tam klinik Excel** — 49 değişken içeren orijinal: pCR, ER, PR, HER2, age, menopause, ethnicity, oncotype, mammaprint, survival, days_to_recurrence...
- **Onkotype etiketli ayrı Excel** — Arda'nın elinde 70 + 230 hasta için Onkotype skorları var
- **Mevcut nnU-Net kodu ve sonuçları** — Arda'nın %77-80 Dice aldığı pipeline
- **ECR Viyana sunumu** — 5 model × 3 mod deney sonuçları
- **MAMA-MIA train/test split tablosu** — reproducibility için

### 2.3. Alt Tip Dağılımı (Mevcut Etiketler)

| Alt Tip | N | Notlar |
|---|---|---|
| triple_negative | 499 | TNBC, en agresif |
| luminal_a | 381 | En iyi prognoz |
| luminal | 211 | A/B ayrımı yok |
| her2_enriched | 169 | HER2+ |
| luminal_b | 155 | Ki-67 yüksek Luminal |
| her2_pure | 65 | HR-, HER2+ |
| **NaN** | **26** | Etiket eksik |

**Kritik uyarı:** ISPY2 (980 hasta) yalnızca 3 alt tipi içeriyor, Luminal A yok. **Collection-stratified split şart.**

### 2.4. Görüntü Özellikleri

- **Shape:** (448, 448, 160) — değişkenlik var, resample gerekir
- **Voxel:** 0.8 × 0.8 × 1.1 mm (anisotropic)
- **Faz sayısı:** Çoğunlukla 4, bazen 5
- **Kanallar:** `_0000` pre-contrast, `_0001`+ post-contrast

---

## 3. Aşamalı Yol Haritası

### Aşama 1 — Segmentasyon (Hedef: Dice ≥ 0.88)

**Baseline:** MAMA-MIA paper'ı Dice 0.8287 raporluyor. Arda %77-80 almış — muhtemelen preprocessing farkı. İlk iş baseline'ı aynı seviyede reproduce etmek.

**Deney matrisi — faz seçimi (Arda'nın yaptığı ama sistematikleştirilecek):**

| Deney | Input | Beklenti |
|---|---|---|
| S-A | Sadece faz 2 | Arda'nın baseline'ı |
| S-B | Faz 1 + Faz 2 (2 kanal) | Post-contrast erken |
| S-C | (Faz 1 − Faz 0) + Faz 2 | **Subtraction — yeni** |
| S-D | Tüm 4 faz | Standart DCE |
| S-E | Faz 0, 1, 2, 3 + SER map | **Radiomics-informed — yeni** |

**Model matrisi:**

| Model | Notlar |
|---|---|
| nnU-Net v2 (3d_fullres) | Baseline reproduce |
| nnU-Net v2 (3d_lowres + cascade) | Büyük hacim için |
| SwinUNETR (MONAI) | Transformer baseline |
| SegResNet (MONAI) | Lightweight alternatif |
| **Custom hybrid** | Filiz Hoca'nın istediği — "kendi modelimiz" |

**Split stratejisi:** 70/15/15 stratified by `collection × tumor_subtype` (veya MAMA-MIA'nın resmi splitini kullan).

**Değerlendirme:** Dice, Hausdorff95, sensitivity, specificity. Per-subtype ve per-collection breakdown.

### Aşama 2 — Moleküler Alt Tip (Hedef: Macro-F1 > %75, Arda %66'da)

**Problem:** Arda ResNet18/50 ile %66'da takılıyor. Çünkü moleküler alt tip görüntüden çok zor ayırt edilir — bu bir kimyasal özellik.

**Stratejiler:**

1. **Crop + multi-phase CNN** — Arda'nın yaklaşımı, baseline
2. **Radiomics + XGBoost** — PyRadiomics ile 100+ feature çıkar, klasik ML
3. **Deep + Radiomics fusion** — CNN features + radiomics → birleşik model
4. **Multi-phase Vision Transformer** — fazları sequence olarak işle
5. **Temporal LSTM** — DCE kinetik eğrisini zaman serisi olarak modelle

Notlardaki "Vision Transformer + LSTM + patch + radiomics" tam bu yönü gösteriyor.

**4 sınıfa indirgeme önerisi:**
- `Luminal A` ← luminal_a + (luminal ∩ düşük grade)
- `Luminal B` ← luminal_b + (luminal ∩ yüksek grade)
- `HER2-enriched` ← her2_enriched + her2_pure
- `Triple Negative` ← triple_negative

(Bu birleştirme tam klinik Excel gelince ER/PR/HER2/Ki-67 ile kesinleşecek.)

### Aşama 3 — pCR Tahmini (Binary classification)

Tam klinik Excel gerekli. MICCAI MAMA-MIA Challenge'ın 2. görevi bu, challenge'a başvurma ihtimali var.

### Aşama 4 — Onkotype Skor Tahmini (Regression)

70 + 230 hasta (Arda'da) az, sadece Luminal hastaları için anlamlı. İleri aşama, önce 1-3 bitsin.

---

## 4. Teknik Stack

- **Framework:** PyTorch + MONAI + PyTorch Lightning
- **Preprocessing:** SimpleITK, nibabel, N4 bias correction, z-score (phase-aware)
- **Radiomics:** PyRadiomics
- **Segmentasyon:** nnU-Net v2 (baseline), MONAI (custom)
- **Deney takibi:** Weights & Biases
- **GPU:** Arkadaşın makinesi (nnU-Net 3d_fullres için min. 11GB VRAM, 24GB önerilen)

---

## 5. Repo Yapısı

```
mama-mia-breast-pipeline/
├── README.md                          (bu dosya)
├── .gitignore                         (data/, models/, *.nii.gz, wandb/)
├── requirements.txt
├── scripts/
│   ├── explore_data.py                (veri keşif)
│   ├── build_split.py                 (stratified split)
│   └── run_baseline_inference.py      (MAMA-MIA pretrained model)
├── src/
│   ├── data/
│   │   ├── dataset.py
│   │   ├── transforms.py
│   │   └── split.py
│   ├── models/
│   │   ├── segmentation/
│   │   └── classification/
│   ├── training/
│   └── utils/
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_subtype_analysis.ipynb
│   └── 03_results_viz.ipynb
└── configs/
    ├── seg_nnunet.yaml
    └── subtype_baseline.yaml
```

---

## 6. Pazartesi İçin Somut Hedef (İlk Etap)

- [x] Veri keşfi tamamlandı (1505 hasta, 4 faz, shape 448×448×160)
- [x] Alt tip dağılımı anlaşıldı (6 etiket, 4 sınıfa indirgenecek)
- [ ] GitHub private repo kurulumu
- [ ] Stratified split (collection × subtype) oluşturma
- [ ] MAMA-MIA pretrained nnU-Net ile inference çalıştırma → Dice baseline
- [ ] Eğer baseline düşükse preprocessing audit → N4, resample, z-score kontrol

**Pazartesi sonu teslim:** Private GitHub repo + baseline Dice metriği + alt tip dağılım notebook'u + eksik dosya listesi Arda'ya.

## 7. güncelleme model %83.45 ile eğitildi
