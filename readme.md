# 🧠 NöroLens: Açıklanabilir Beyin MR Analizi

Beyin MR görüntülerinde **tümör şüphesini** tespit eden ve kararını **Grad-CAM ısı haritasıyla** görsel olarak açıklayan, eğitim amaçlı bir yapay zekâ projesidir. Kullanıcı bir MR görüntüsü yükler; sistem sonucu, güven skorunu ve modelin karar verirken baktığı bölgeyi tek ekranda gösterir.

> ⚠️ **Uyarı:** Bu proje yalnızca eğitim amaçlıdır. Tıbbi tanı koymaz, uzman değerlendirmesinin yerine geçmez.

---

## 🎯 Projenin amacı

Beyin tümörlerinde erken tespit tedavi başarısını doğrudan etkiler. Ancak MR değerlendirmesi uzman gerektirir ve yoğun iş yükü gözden kaçma riski doğurabilir. Yapay zekâ bu yükü azaltabilir, fakat iki sorun vardır:

1. Modeller çoğu zaman **"kara kutu"** gibi çalışır, kararın nedenini göstermez.
2. Raporlanan yüksek doğruluklar, **veri sızıntısı ve tekrar eden görseller** yüzünden gerçeği yansıtmayabilir.

NöroLens bu iki soruna yanıt arar: **dürüstçe ölçülmüş** bir model ve **açıklanabilir** bir karar.

---

## ✨ Özellikler

- EfficientNetB0 tabanlı tümör var/yok sınıflandırması (transfer öğrenme)
- Grad-CAM++ ile karar gerekçesi ısı haritası
- Streamlit tabanlı web arayüzü (görsel yükle, sonucu gör)
- Güven skoru ve ham skor gösterimi
- Kırpmalı / kırpmasız tahmin karşılaştırması (teşhis amaçlı)
- Tümörlü vakaların kaçırılmasını azaltmak için ayarlanmış karar eşiği (0,4)
- Yerel çalışır: yüklenen görüntüler dışarıya gönderilmez

---

## 🔬 Yöntem

### Veri hazırlama
- Kaggle kaynaklı, `no` (tümörsüz) ve `yes` (tümörlü) etiketli herkese açık beyin MR görüntüleri kullanıldı.
- **Veri sızıntısı giderildi:** İlk denemede diske yazılan augmentation kopyalarının test setine karıştığı fark edildi. Bu dosyalar silindi, augmentation eğitim sırasında canlı uygulanmaya başlandı.
- **Tekrar eden görseller temizlendi:** md5 karması ile birebir aynı yaklaşık **2.900** görsel silindi (`duplicate_check.py`, `remove_duplicates.py`).
- **Beyin kırpma:** Otsu eşikleme, erode/dilate ve kontur analiziyle MR'lardaki siyah kenarlar kesildi (`crop_dataset.py`).
- **Bölme:** Sınıf oranı korunarak (stratified) %70 eğitim / %15 doğrulama / %15 test.

| Küme | Görsel sayısı |
|---|---|
| Eğitim | 8.397 |
| Doğrulama | 1.799 |
| Test | 1.800 |

### Model ve eğitim
- **Backbone:** EfficientNetB0 (ImageNet ağırlıkları), girdi 224×224
- **Başlık:** GlobalAveragePooling2D → Dropout(0,3) → Dense(128, ReLU) → Dense(1, sigmoid)
- **İki aşamalı eğitim:**
  1. Backbone dondurulur, yalnızca başlık eğitilir (lr = 1e-3)
  2. Son 40 katman açılır, düşük öğrenme hızıyla ince ayar yapılır (lr = 1e-4); BatchNormalization katmanları dondurulmuş kalır
- **Canlı augmentation:** yatay çevirme, döndürme, zoom, kaydırma, kontrast
- **Sınıf ağırlıkları** (sınıf dengesizliği için)
- **Callback'ler:** ModelCheckpoint (en iyi `val_accuracy`), EarlyStopping, ReduceLROnPlateau
- **Karar eşiği:** 0,4 (yanlış negatifleri azaltmak için)
- Veri `tf.data` ile diskten akış halinde okunur (RAM şişmez)

### Açıklanabilirlik (XAI)
- **Grad-CAM++**, sigmoid doygunluğunu aşmak için **sigmoid öncesi ham değer (logit)** üzerinden hesaplanır.
- Varsayılan katman: `block6a_expand_activation` (14×14). Alternatifler: `top_activation` (7×7), `block4a_expand_activation` (28×28).
- Isı haritası eşiklenerek yalnızca güçlü bölgeler orijinal görüntüye bindirilir.

---

## 📊 Sonuçlar

### 1) İç test (1.800 görsel, eğitimde görülmemiş)

**Test doğruluğu: ≈ %99,1**

| Eşik | Karışıklık matrisi (gerçek no / yes) | Yanlış pozitif | Yanlış negatif |
|---|---|---|---|
| 0,5 | [[254, 10], [6, 1530]] | 10 | 6 |
| **0,4** | [[254, 10], [4, 1532]] | 10 | **4** |

Eşik 0,4 seçildi: tümörlü bir vakanın kaçırılması, tümörsüz bir vakanın yanlışlıkla işaretlenmesinden daha ağır bir hata kabul edildi.

### 2) Dış test (225 görsel, ayrı klasör)

Eğitime hiç katılmayan ayrı bir test klasörüyle ölçüldü (105 tümörsüz, 120 tümörlü).

**Dış test doğruluğu: ≈ %95,6**

| | Tahmin: no | Tahmin: yes |
|---|---|---|
| **Gerçek: no** | 95 | 10 |
| **Gerçek: yes** | 0 | 120 |

| Sınıf | Precision | Recall | F1 |
|---|---|---|---|
| no | 1,00 | 0,90 | 0,95 |
| yes | 0,92 | 1,00 | 0,96 |

➡️ Dış testte **tümörlü 120 görselin hiçbiri kaçırılmadı** (yanlış negatif = 0). Tüm hatalar yanlış pozitiftir.

---

## ⚠️ Sınırlılıklar (dürüst değerlendirme)

- **Hata deseni:** İç ve dış testteki yanlışların çoğu `no` sınıfında ve `image(N).jpg` adlı belirli bir kaynak grubunda toplanıyor. Bu, olası **etiket hatası** veya **veri kaynağı kayması** işaretidir; ayrıca incelenmelidir.
- **Dış test kapsamı:** Dış test seti küçüktür ve eğitim verisiyle aynı kaynak ailesinden olabilir. Gerçek genelleme için tamamen farklı kaynaklı verilerle yeniden doğrulama gerekir.
- **Grad-CAM yorumu:** Isı haritası tümörün kesin konumunu değil, **kararın dayandığı bölgeleri** gösterir. Bazı görsellerde geniş alanlara yayılabilir. Doğruluğunu ölçmek için tümör maskeli veri gerekir.
- **Hasta bazlı bölme yok:** Aynı hastanın birden çok kesiti farklı kümelere düşmüş olabilir.
- Model tıbbi tanı için doğrulanmamıştır.

---

| Dosya | Görev |
|---|---|
| `app.py` | Streamlit web arayüzü |
| `gradcam.py` | Grad-CAM++ ısı haritası ve bindirme |
| `train.py` | Veri hazırlama, iki aşamalı eğitim, test değerlendirmesi |
| `crop_dataset.py` | Beyin kırpma ön işlemi |
| `duplicate_check.py` | Tekrar eden görselleri bulma |
| `remove_duplicates.py` | Tekrarları silme |
| `test_dis.py` | Ayrı klasörle dış test |
| `best_model.keras` | Eğitilmiş model |
| `requirements.txt` | Bağımlılıklar |


## 🛠️ Kullanılan teknolojiler

Python 3.9 · TensorFlow/Keras · EfficientNetB0 · OpenCV · NumPy · scikit-learn · Pillow · Streamlit

## 🔭 Gelecek planları

- İndirilebilir **PDF ön MR raporu** (sonuç, güven skoru, belirsizlik uyarısı, ısı haritası)
- Çok sınıflı tür ayrımı (glioma, meningioma, hipofiz)
- Tümör segmentasyonu ve Grad-CAM'in maskeyle sayısal doğrulanması
- Kalibrasyon ve belirsizlik tahmini (Monte Carlo Dropout)
- Etiket hatası tespiti ve farklı kaynaklı verilerle yeniden doğrulama

## 📚 Veri kaynakları

Bu projede Kaggle'da herkese açık beyin MR veri setleri kullanılmıştır. Veri setleri bu repoda dağıtılmaz; kullanım koşulları için ilgili Kaggle sayfalarına bakınız.

https://www.kaggle.com/datasets/ishans24/brain-tumor-dataset
https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation
https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection



