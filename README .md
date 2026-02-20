# Gökyüzü Bulutluluk Analizi (Sky Cloudiness Analysis) — TUG Private Dataset

Bu proje, fisheye / all-sky gökyüzü görüntülerinden **bulutluluk (cloudiness)** oranını hesaplayan ve gözlem kalitesini takip etmek için **CSV rapor** + **grafik** çıktıları üreten Python tabanlı bir analiz pipeline’ıdır.  

Pipeline iki katmandan oluşur:

- **Kural-tabanlı görüntü işleme (varsayılan):** ROI (fisheye dairesi) tespiti + bulut maskesi + bulutluluk yüzdesi + kategori (clear/partly_cloudy/overcast)
- **Opsiyonel derin öğrenme (TensorFlow varsa):** EfficientNet (B0/B3) ile sınıflandırma eğitimi + metrik raporları + en iyi model seçimi

> **Veri Gizliliği (Önemli):** Bu projede kullanılan veri seti **Türkiye Ulusal Gözlemevleri (TUG)** tarafından sağlanmış **private** bir veri setidir ve **paylaşılamaz**.  
> Bu depo yalnızca **kod**, **pipeline**, **konfigürasyon** ve (varsa) **anonimleştirilmiş örnek çıktı** içerir. **Ham veri repoya eklenmemelidir.**

---

## İçerik
- [Özellikler](#özellikler)
- [Proje Akışı](#proje-akışı)
- [Kurulum](#kurulum)
- [Çalıştırma](#çalıştırma)
- [Komut Satırı Argümanları](#komut-satırı-argümanları)
- [Bulutluluk Hesabı ve Kategoriler](#bulutluluk-hesabı-ve-kategoriler)
- [ML (Opsiyonel) — Model, Eğitim ve Metrikler](#ml-opsiyonel--model-eğitim-ve-metrikler)
- [Çıktılar](#çıktılar)
- [Konfigürasyon](#konfigürasyon)
- [Proje Yapısı](#proje-yapısı)
- [Refactor Notları](#refactor-notları)
- [Lisans](#lisans)

---

## Özellikler
- **Sky ROI tespiti (0/1 binary):** `HoughCircles → kontur fallback → tüm frame` yaklaşımı
- **Bulut maskesi üretimi:** R/B oranı + düşük saturasyon + Otsu; maske 0/1; morfoloji için 0/255 dönüşümü
- **Gündüz/Gece ayrımı:** ROI içindeki ortalama parlaklık (HSV-V) ile `NIGHT_V_MEAN_THRESH`
- **Gece politikası:** `NIGHT_POLICY` ile gece çıktılarının yönetimi
- **Ay filtresi:** `MONTH_FILTER` + `AUTO_MONTH_LATEST` ile tek aya indirgeme
- **Raporlama & grafikler:** günlük zaman serisi, saatlik ortalama, kategori dağılımı, histogram
- **Segmentasyon örnekleri:** 4 panel (orijinal / roi_raw / final_roi / cloud_mask)
- **Debug üretimi:** `roi_debug.txt` (ROI unique değerleri, `sky_pixels==0` dosya listesi)
- **Opsiyonel ML:** Varsayılan kapalı; yalnızca `--train-ml` veya `TRAIN_ML=True` ile çalışır
- **Sistem & donanım teşhisi:** `system_info.json`, `hardware_info.json` ve `nvidia-smi` kontrolü (varsa)

---

## Proje Akışı
1. **Veri keşfi:** klasör taraması, EXIF veya dosya adından datetime çıkarımı, bozuk dosyaların ayrıştırılması
2. **ROI tespiti:** fisheye dairesi belirlenip ROI maskesi üretilir (yalnızca 0/1)
3. **Bulut maskesi:** çoklu ipucu (R/B, S, Otsu) + morfoloji ile bulut pikselleri çıkarılır
4. **Bulutluluk & kategori:** `cloud_pixels / sky_pixels` ile bulutluluk; gündüz için `clear/partly_cloudy/overcast`, gece için `NIGHT_POLICY`
5. **Raporlar:** CSV ve grafikler üretilir; örnek segmentasyon görselleri kaydedilir
6. **(Opsiyonel) ML:** EfficientNetB0/B3 eğitilir, test metrikleri hesaplanır, en iyi model seçilir

---

## Kurulum

### 1) Sanal ortam (önerilir)
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate
```

### 2) Bağımlılıkların kurulumu
`requirements.txt` varsa:
```bash
pip install -r requirements.txt
```

`requirements.txt` yoksa minimum (analiz + raporlama) için:
```bash
pip install opencv-python numpy pandas matplotlib pillow scikit-learn
```

Opsiyonel:
```bash
pip install seaborn psutil streamlit
```

ML (opsiyonel):
```bash
pip install tensorflow
```


---

## Çalıştırma

### 1) Veri klasörünü ayarla
`sky_analysis.py` içindeki `DATA_DIR` değişkenini kendi dataset yoluna göre güncelle:


### 2) Analiz + rapor üret (varsayılan)
```bash
python sky_analysis.py
```

### 3) ML eğitimini aç (opsiyonel)
```bash
python sky_analysis.py --train-ml
```

### 4) Sadece sistem/donanım bilgisini yazdır ve çık
```bash
python sky_analysis.py --print-system-info-only
```

---

## Komut Satırı Argümanları
Script aşağıdaki CLI argümanlarını destekler:

| Argüman | Açıklama |
|---|---|
| `--train-ml` | ML eğitimini çalıştırır (`TRAIN_ML=True`) |
| `--log-file outputs/run.log` | Logları dosyaya da yazar |
| `--log-level DEBUG|INFO|WARNING|ERROR` | Log seviyesi |
| `--force-cpu` | GPU olsa bile CPU kullanır |
| `--print-system-info-only` | Sadece sistem/donanım bilgisini üretir ve çıkar |
| `--no-progress` | İlerleme çıktısını azaltmak için bayrak (gelecek genişletmeler için) |
| `--log-every-batches N` | `N>0` ise her N batch’te loss/acc + throughput loglar |

Örnek:
```bash
python sky_analysis.py --log-level DEBUG --log-file outputs/run.log --log-every-batches 50
```

---

## Bulutluluk Hesabı ve Kategoriler

### Bulutluluk yüzdesi
- ROI (final) içindeki piksel sayısı: `sky_pixels = final_roi.sum()`
- Bulut piksel sayısı: `cloud_pixels = cloud_mask.sum()`
- Bulutluluk: `cloudiness = cloud_pixels / sky_pixels`

> `sky_pixels == 0` olursa `cloudiness = NaN` üretilir ve bu dosyalar `roi_debug.txt` içinde listelenir.

### Gündüz/Gece ayrımı
- Ortalama parlaklık: HSV-V kanalı ortalaması **ROI içinde** hesaplanır.
- `mean_brightness < NIGHT_V_MEAN_THRESH` ise görüntü **gece** kabul edilir.

### Kategori kuralları (gündüz)
- `clear` : bulutluluk ≤ **%10**
- `partly_cloudy` : bulutluluk ≤ **%30**
- `overcast` : bulutluluk > **%30**

### Gece politikası
`NIGHT_POLICY` iki mod sunar:

- `unknown_night` *(varsayılan)*  
  - `category = unknown_night`  
  - `cloudiness = NaN`

- `clear_overcast_night`  
  - `clear_night` : bulutluluk ≤ **%30**  
  - `overcast_night` : bulutluluk > **%30**

---

## ML (Opsiyonel) — Model, Eğitim ve Metrikler

### Ne zaman çalışır?
ML bölümü yalnızca:
- `--train-ml` argümanı verilirse **ve**
- TensorFlow kuruluysa (`HAS_TENSORFLOW=True`)
çalışır.

### Model
- Backbone adayları: **EfficientNetB0**, **EfficientNetB3**
- Transfer learning: backbone başlangıçta donuk (`trainable=False`)
- Head: GAP → Dropout(0.3) → Dense(softmax)
- Preprocess: `keras.applications.efficientnet.preprocess_input` (rescale kullanılmaz)

### Etiket kaynağı
- `ML_LABEL_COLUMN="category"` *(varsayılan)*: `clear / partly_cloudy / overcast` pseudo-label  
- `ML_LABEL_COLUMN="folder"`: klasör adı etiket olur

> `category` ile eğitimde yalnızca `clear/partly_cloudy/overcast` kullanılır; gece/unknown örnekler dışarıda bırakılır.

### Split
- Stratified train/val/test:
  - test: %20
  - val: train_val’ın %25’i (yaklaşık toplamın %20’si)

### Dengesiz sınıflar (class weight)
- `compute_class_weight("balanced", ...)` ile otomatik ağırlık hesaplanır.
- Loglarda sınıf ağırlıkları yazdırılır.

### Değerlendirme metrikleri
`evaluate_model()` fonksiyonu aşağıdaki metrikleri üretir ve `outputs/` altına kaydeder:

- **Accuracy**
- **Precision/Recall/F1 (macro)**
- **Precision/Recall/F1 (weighted)**
- **Balanced Accuracy**
- **Matthews Correlation Coefficient (MCC)**
- **ROC-AUC (OVR, multi-class)** *(uygunsa)*
- **Confusion Matrix** (görsel + sayısal tablo)
- **Classification Report** (sınıf bazlı precision/recall/f1)

### Metrik dosyaları ve sonuçların okunması
Her backbone için:
- `outputs/model_metrics_EfficientNetB0.json`
- `outputs/model_metrics_EfficientNetB3.json`

Confusion matrix:
- `outputs/confusion_matrix_EfficientNetB0.png`
- `outputs/confusion_matrix_EfficientNetB3.png`

ROC eğrileri:
- `outputs/roc_curves_EfficientNetB0.png`
- `outputs/roc_curves_EfficientNetB3.png`



### En iyi modelin seçimi
- Öncelik: `f1_macro`
- Eşitlik durumunda: `accuracy`
- En iyi model: `outputs/best_cloud_classifier.keras`
- Inference konfigurasyonu: `outputs/best_model_config.json`

---

## Çıktılar
Çalıştırma sonunda `outputs/` altında tipik olarak:

### Analiz / Rapor
- `dataset_profile.csv` : keşif çıktısı (dosya, boyut, datetime, parlaklık, gece/gündüz)
- `broken_files.txt` : okunamayan/bozuk dosyalar
- `per_image_cloudiness.csv` : görüntü bazında bulutluluk + kategori
- `roi_debug.txt` : ROI unique değerleri + `sky_pixels==0` dosyaları
- `system_info.json` : OS, Python, paket sürümleri (best-effort)
- `hardware_info.json` : TF cihazları + nvidia-smi çıktısı (varsa)

### Grafikler
- `daily_cloudiness.png`
- `hourly_cloudiness.png`
- `cloud_category_pie.png`
- `cloudiness_histogram.png`
- `cloudiness_by_label_boxplot.png`
- `cloudiness_by_label_violin.png` *(seaborn varsa)*
- `segmentation_example_*.png` *(4 panel)*

### ML (TensorFlow + `--train-ml`)
- `training_log.csv` ve `training_log_EfficientNetB0.csv` / `training_log_EfficientNetB3.csv`
- `training_curves_EfficientNetB0.png` / `training_curves_EfficientNetB3.png`
- `model_summary_EfficientNetB0.txt` / `model_summary_EfficientNetB3.txt`
- `cloud_classifier_EfficientNetB0.keras` / `cloud_classifier_EfficientNetB3.keras`
- `best_cloud_classifier.keras`
- `best_model_config.json`

---

## Konfigürasyon
`sky_analysis.py` içindeki başlıca ayarlar:

- `DATA_DIR` : veri klasörü
- `IMAGE_SIZE` : (H, W)
- `BATCH_SIZE`, `EPOCHS`, `SEED`
- `MONTH_FILTER` : `"YYYY-MM"` veya `None`
- `AUTO_MONTH_LATEST`
- `NIGHT_V_MEAN_THRESH`
- `NIGHT_POLICY` : `"unknown_night"` veya `"clear_overcast_night"`
- `INCLUDE_NIGHT_IN_SUMMARY`
- `TRAIN_ML`, `ML_LABEL_COLUMN`
- `OUTPUT_DIR`

---

## Proje Yapısı
```text
.
├─ sky_analysis.py          # Ana analiz betiği (CV + rapor + opsiyonel ML)
├─ app.py                   # Opsiyonel Streamlit UI (minimal)
├─ REFACTOR_OZET.md         # Refactor notları / davranış değişiklikleri
├─ requirements.txt         # Bağımlılıklar (varsa)
├─ outputs/                 # Üretilen raporlar ve görseller
└─ data/                    # (Repo’da yok) Private veri seti bu klasöre konumlandırılır
```


---

## Refactor Notları
Refactor süreci ve davranış değişikliklerinin tamamı: **`REFACTOR_OZET.md`**  
Öne çıkanlar:
- ROI maskesi yalnızca `{0,1}`
- Hough → kontur fallback → tüm frame ROI stratejisi
- Morfoloji için 0/255 dönüşümü
- Gece tespiti ROI parlaklığına taşındı; `NIGHT_POLICY` eklendi
- ML varsayılan kapalı; `--train-ml` ile aktif

---

## Lisans
Bu depo **kod** paylaşımı içindir. Veri seti TUG koşullarına tabidir ve bu repo kapsamında değildir.  



**EfficientNetB0**
- accuracy: ...
- f1_macro: ...
- balanced_accuracy: ...
- mcc: ...
- roc_auc_ovr: ...

**EfficientNetB3**
- accuracy: ...
- f1_macro: ...
- balanced_accuracy: ...
- mcc: ...
- roc_auc_ovr: ...

> Görseller: `outputs/confusion_matrix_*.png`, `outputs/roc_curves_*.png`, `outputs/training_curves_*.png`
