"""Sınıflandırma veri setini yükler ve diske kaydeder.

Breast Cancer Wisconsin veri seti (scikit-learn) kullanılır. 30 sayısal öznitelik ile
tümörün kötü huylu (malignant) veya iyi huylu (benign) olduğu tahmin edilir.
"""

from __future__ import annotations

import os

import pandas as pd
from sklearn.datasets import load_breast_cancer

RAW_PATH = os.path.join("data", "raw", "breast_cancer.csv")

# Orijinal veri setinde 0 = malignant, 1 = benign. Okunabilirlik için etiketleri
# anlamlı isimlere çeviriyoruz.
TARGET_NAMES = {0: "malignant", 1: "benign"}


def load_dataframe() -> pd.DataFrame:
    """Veri setini bir pandas DataFrame olarak döndürür."""
    dataset = load_breast_cancer(as_frame=True)
    df = dataset.frame.copy()
    df["diagnosis"] = df["target"].map(TARGET_NAMES)
    return df


def save_raw(df: pd.DataFrame, path: str = RAW_PATH) -> str:
    """DataFrame'i ham veri klasörüne CSV olarak yazar."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    return path


def main() -> pd.DataFrame:
    df = load_dataframe()
    path = save_raw(df)
    print(f"Veri yüklendi: {df.shape[0]} satır, {df.shape[1]} sütun")
    print(f"Hedef dağılımı:\n{df['diagnosis'].value_counts().to_string()}")
    print(f"Ham veri kaydedildi -> {path}")
    return df


if __name__ == "__main__":
    main()
