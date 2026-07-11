"""Veri temizleme, özellik ölçekleme ve eğitim/test ayrımı.

Sınıflandırma modelleri (özellikle k-NN ve Logistic Regression) özniteliklerin
ölçeğine duyarlı olduğundan StandardScaler ile standardizasyon uygulanır. Ölçekleyici
yalnızca eğitim verisiyle fit edilir; böylece test verisi bilgisi eğitime sızmaz.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

PROCESSED_PATH = os.path.join("data", "processed", "breast_cancer_scaled.csv")

FEATURE_DROP = ["target", "diagnosis"]
TARGET_COL = "target"

RANDOM_STATE = 42
TEST_SIZE = 0.20


@dataclass
class Dataset:
    """Ölçeklenmiş eğitim/test bölümlerini taşıyan basit kap."""

    X_train: np.ndarray
    X_test: np.ndarray
    y_train: pd.Series
    y_test: pd.Series
    feature_names: list[str]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Temel temizlik: yinelenen satırları at, eksik değerleri kontrol et."""
    df = df.drop_duplicates().reset_index(drop=True)
    missing = df.isna().sum().sum()
    if missing:
        # Bu veri setinde eksik değer beklenmez; yine de savunmacı davranıyoruz.
        df = df.dropna().reset_index(drop=True)
    return df


def prepare(df: pd.DataFrame, save: bool = True) -> Dataset:
    """Öznitelikleri ölçekler ve stratify ile eğitim/test olarak ayırır."""
    df = clean(df)

    feature_names = [c for c in df.columns if c not in FEATURE_DROP]
    X = df[feature_names]
    y = df[TARGET_COL]

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    if save:
        _save_processed(X_train, y_train, feature_names)

    return Dataset(X_train, X_test, y_train, y_test, feature_names)


def _save_processed(X_train: np.ndarray, y_train: pd.Series, feature_names: list[str]) -> str:
    os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)
    out = pd.DataFrame(X_train, columns=feature_names)
    out[TARGET_COL] = y_train.to_numpy()
    out.to_csv(PROCESSED_PATH, index=False)
    return PROCESSED_PATH


def main() -> Dataset:
    from data_loader import load_dataframe

    data = prepare(load_dataframe())
    print(f"Eğitim seti: {data.X_train.shape}, Test seti: {data.X_test.shape}")
    print(f"Öznitelik sayısı: {len(data.feature_names)}")
    print(f"İşlenmiş eğitim verisi kaydedildi -> {PROCESSED_PATH}")
    return data


if __name__ == "__main__":
    import sys

    sys.path.insert(0, os.path.dirname(__file__))
    main()
