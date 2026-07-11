# Sonuç ve Değerlendirme

## Model Karşılaştırması

| Model | Accuracy |
| --- | --- |
| Logistic Regression | 0.9825 |
| Random Forest | 0.9561 |
| k-NN (k=5) | 0.9561 |

## Yorum

Bu veri setinde en yüksek doğruluğu **Logistic Regression** (0.9825) elde etti; en düşük skor **k-NN (k=5)** (0.9561) modelindeydi.

Breast Cancer Wisconsin veri seti doğrusal olarak büyük ölçüde ayrılabilir olduğu için, standardize edilmiş özniteliklerle Logistic Regression güçlü bir temel (baseline) sunar. Random Forest doğrusal olmayan etkileşimleri yakalayabilir ancak bu veride ek kazanç sınırlıdır. k-NN ise ölçeklemeden sonra rekabetçi olsa da yüksek boyutlu uzayda uzaklık temelli olduğundan diğer modellerin biraz gerisinde kalabilir. Kazanan modelin öne çıkmasının temel nedeni, problemin yapısının o algoritmanın varsayımlarıyla (No Free Lunch teoremi) örtüşmesidir.
