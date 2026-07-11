# Applied Data Science — Sınıflandırma Modeli Karşılaştırması

Bu proje, uçtan uca (end-to-end) bir Makine Öğrenmesi hattını (pipeline) gerçek bir
Veri Bilimci bakış açısıyla uygulamayı amaçlar: bir sınıflandırma veri setini yükleme,
temizleme, ölçekleme, eğitim/test olarak ayırma, **birden fazla** sınıflandırma
algoritmasını **aynı** veri üzerinde eğitip test etme ve performanslarını
karşılaştırarak veriye en uygun modeli belirleme.

## Problem

**Meme kanseri teşhisi (disease diagnosis).** scikit-learn içindeki Breast Cancer
Wisconsin veri seti kullanılır: 30 sayısal öznitelikten hareketle tümörün
`malignant` (kötü huylu) mu yoksa `benign` (iyi huylu) mu olduğunu tahmin eden ikili
bir sınıflandırma problemidir.

## Pipeline

1. **Veri yükleme** — `src/data_loader.py` veriyi yükleyip `data/raw/` altına yazar.
2. **Ön işleme** — `src/preprocessing.py` özellik ölçekleme (StandardScaler) uygular ve
   veriyi eğitim/test olarak (stratify ile) ayırır.
3. **Modelleme** — `src/models.py` en az iki sınıflandırıcıyı (Logistic Regression,
   Random Forest, k-NN) aynı eğitim verisiyle eğitir; her biri için **Accuracy Score**
   ve **Confusion Matrix** üretir.
4. **Raporlama** — sonuçlar karşılaştırılır, confusion matrix görselleri
   `reports/figures/` altına, kısa değerlendirme `reports/conclusion.md` içine yazılır.

## Çalıştırma

```bash
pip install -r requirements.txt
python main.py
```

`python main.py` tüm hattı sırayla çalıştırır ve her model için doğruluk skorunu ve
confusion matrix'i ekrana basar.

## Proje Yapısı

```
.
├── main.py                 # Uçtan uca pipeline
├── requirements.txt
├── src/
│   ├── data_loader.py      # Veri setini yükler
│   ├── preprocessing.py    # Ölçekleme + eğitim/test ayrımı
│   └── models.py           # Sınıflandırıcıların eğitimi ve değerlendirmesi
├── data/
│   ├── raw/                # Ham veri
│   └── processed/          # İşlenmiş veri
└── reports/
    ├── figures/            # Confusion matrix görselleri
    └── conclusion.md       # Model karşılaştırması ve sonuç
```

## Teorik Sorular (3.1.1)

**1. Neden hiçbir algoritmanın (ör. Random Forest) tüm veri setleri için bir
diğerinden (ör. k-NN) kesin olarak üstün olduğu söylenemez?**
"No Free Lunch" teoremi gereği hiçbir algoritma her problem için en iyi değildir;
çünkü her algoritmanın verinin yapısına dair farklı varsayımları (bias) vardır ve bir
veri setinin geometrisine uyan varsayım başka bir veri setinde başarısız olabilir.

**2. Accuracy, Precision ve Recall arasındaki temel fark nedir?**
Accuracy tüm tahminler içindeki doğru oranıdır; Precision pozitif dediklerimizin ne
kadarının gerçekten pozitif olduğunu (yanlış alarmı), Recall ise gerçek pozitiflerin
ne kadarını yakaladığımızı (kaçırılanları) ölçer.

**3. Neden veriyi değerlendirmeden önce "Eğitim" ve "Test" olarak ayırmak zorunludur?**
Modelin daha önce görmediği veri üzerindeki genelleme başarısını ölçmek ve ezberlemeyi
(overfitting) tespit etmek için; aynı veriyle hem eğitip hem test etmek yanıltıcı
şekilde yüksek skorlar verir.
