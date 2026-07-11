"""Birden fazla sınıflandırıcının eğitimi, değerlendirmesi ve karşılaştırması.

Yönerge gereği en az iki algoritma AYNI eğitim verisiyle eğitilir ve AYNI (görülmemiş)
test verisiyle test edilir. Her model için Accuracy Score ve Confusion Matrix üretilir.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")  # Ekran/başlıksız ortamlarda güvenli çizim
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

FIG_DIR = os.path.join("reports", "figures")
CLASS_LABELS = ["malignant", "benign"]  # 0, 1 sırasına karşılık gelir


@dataclass
class ModelResult:
    name: str
    accuracy: float
    conf_matrix: np.ndarray


def build_models() -> dict[str, object]:
    """Karşılaştırılacak sınıflandırıcılar (en az iki farklı algoritma)."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=5000),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "k-NN (k=5)": KNeighborsClassifier(n_neighbors=5),
    }


def _plot_confusion(name: str, cm: np.ndarray) -> str:
    os.makedirs(FIG_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {name}")
    ax.set_xlabel("Tahmin")
    ax.set_ylabel("Gerçek")
    ax.set_xticks(range(len(CLASS_LABELS)))
    ax.set_yticks(range(len(CLASS_LABELS)))
    ax.set_xticklabels(CLASS_LABELS)
    ax.set_yticklabels(CLASS_LABELS)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    slug = name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("=", "")
    path = os.path.join(FIG_DIR, f"cm_{slug}.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def evaluate(data, save_figures: bool = True) -> list[ModelResult]:
    """Tüm modelleri eğitir, test eder ve sonuçlarını döndürür."""
    results: list[ModelResult] = []
    for name, model in build_models().items():
        model.fit(data.X_train, data.y_train)
        y_pred = model.predict(data.X_test)

        acc = accuracy_score(data.y_test, y_pred)
        cm = confusion_matrix(data.y_test, y_pred)

        print(f"\n=== {name} ===")
        print(f"Accuracy Score: {acc:.4f}")
        print("Confusion Matrix:")
        print(cm)

        if save_figures:
            _plot_confusion(name, cm)

        results.append(ModelResult(name=name, accuracy=acc, conf_matrix=cm))

    return results


def summarize(results: list[ModelResult]) -> pd.DataFrame:
    """Modelleri doğruluk skoruna göre sıralı bir tablo olarak özetler."""
    table = pd.DataFrame(
        [{"Model": r.name, "Accuracy": round(r.accuracy, 4)} for r in results]
    ).sort_values("Accuracy", ascending=False).reset_index(drop=True)
    return table


def run(data=None) -> pd.DataFrame:
    if data is None:
        import sys

        sys.path.insert(0, os.path.dirname(__file__))
        from data_loader import load_dataframe
        from preprocessing import prepare

        data = prepare(load_dataframe())

    results = evaluate(data)
    table = summarize(results)
    print("\n--- Karşılaştırma ---")
    print(table.to_string(index=False))
    return table


if __name__ == "__main__":
    run()
