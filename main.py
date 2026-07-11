"""Uçtan uca sınıflandırma pipeline'ı: veri yükleme -> ön işleme -> modelleme -> rapor.

Kullanım:
    python main.py
"""

from __future__ import annotations

import os
import sys

# src klasörünü modül yoluna ekle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import data_loader  # noqa: E402
import models  # noqa: E402
import preprocessing  # noqa: E402

CONCLUSION_PATH = os.path.join("reports", "conclusion.md")


def write_conclusion(table) -> None:
    """Model karşılaştırmasını ve kısa değerlendirmeyi rapora yazar."""
    best = table.iloc[0]
    worst = table.iloc[-1]
    os.makedirs(os.path.dirname(CONCLUSION_PATH), exist_ok=True)

    lines = [
        "# Sonuç ve Değerlendirme",
        "",
        "## Model Karşılaştırması",
        "",
        "| Model | Accuracy |",
        "| --- | --- |",
    ]
    for _, row in table.iterrows():
        lines.append(f"| {row['Model']} | {row['Accuracy']:.4f} |")

    lines += [
        "",
        "## Yorum",
        "",
        f"Bu veri setinde en yüksek doğruluğu **{best['Model']}** "
        f"({best['Accuracy']:.4f}) elde etti; en düşük skor **{worst['Model']}** "
        f"({worst['Accuracy']:.4f}) modelindeydi.",
        "",
        "Breast Cancer Wisconsin veri seti doğrusal olarak büyük ölçüde ayrılabilir "
        "olduğu için, standardize edilmiş özniteliklerle Logistic Regression güçlü bir "
        "temel (baseline) sunar. Random Forest doğrusal olmayan etkileşimleri "
        "yakalayabilir ancak bu veride ek kazanç sınırlıdır. k-NN ise ölçeklemeden "
        "sonra rekabetçi olsa da yüksek boyutlu uzayda uzaklık temelli olduğundan "
        "diğer modellerin biraz gerisinde kalabilir. Kazanan modelin öne çıkmasının "
        "temel nedeni, problemin yapısının o algoritmanın varsayımlarıyla (No Free "
        "Lunch teoremi) örtüşmesidir.",
        "",
    ]
    with open(CONCLUSION_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nDeğerlendirme raporu yazıldı -> {CONCLUSION_PATH}")


def main() -> None:
    print("=" * 60)
    print("1) Veri yükleniyor")
    print("=" * 60)
    df = data_loader.load_dataframe()
    data_loader.save_raw(df)

    print("\n" + "=" * 60)
    print("2) Ön işleme (ölçekleme + eğitim/test ayrımı)")
    print("=" * 60)
    dataset = preprocessing.prepare(df)
    print(f"Eğitim: {dataset.X_train.shape}, Test: {dataset.X_test.shape}")

    print("\n" + "=" * 60)
    print("3) Modelleme ve değerlendirme")
    print("=" * 60)
    table = models.run(dataset)

    print("\n" + "=" * 60)
    print("4) Rapor")
    print("=" * 60)
    write_conclusion(table)

    print("\nPipeline tamamlandı.")


if __name__ == "__main__":
    main()
