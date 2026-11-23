"""
Akademik Değerlendirme ve Görselleştirme Modülü (src/evaluate.py)

Tüm modelleri (End-to-End DL ve Hibrit ML) Test seti üzerinde değerlendirir ve
makalenin "4. Deneysel Sonuçlar" bölümü için 300 DPI çözünürlükte zorunlu
çıktıları `outputs/` klasörüne otomatik olarak kaydeder:
1. preprocessing_comparison.png (mevcut değilse örnek test görüntüsünden üretilir)
2. training_curves.png (Train/Val Accuracy & Loss eğrileri)
3. confusion_matrix_dl.png ve confusion_matrix_hybrid.png (sayı + yüzdelik ısı haritaları)
4. roc_curves.png (4 sınıf için One-vs-Rest ROC eğrileri, sınıf bazlı + makro AUC)
5. model_comparison_table.csv ve konsol tablosu
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

import config
from src.dataset import create_dataloaders
from src.preprocessing import generate_preprocessing_comparison
from src.train_dl import predict_dl_model


# Akademik makale görselleri için temiz Matplotlib/Seaborn stili
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.8,
    }
)


def plot_training_curves(
    checkpoints_dir: Path = config.CHECKPOINTS_DIR,
    output_path: Path = config.OUTPUTS_DIR / "training_curves.png",
) -> Optional[Path]:
    """
    Eğitilen DL modellerinin (ResNet50, DenseNet121) epoch bazlı Train/Validation
    Accuracy ve Loss grafiklerini 300 DPI PNG olarak çizer.
    """
    history_files = sorted(checkpoints_dir.glob("*_history.csv"))
    if not history_files:
        print("[UYARI] Eğitim geçmişi CSV dosyası bulunamadı; training_curves.png atlanıyor.")
        return None

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), dpi=config.FIGURE_DPI)
    colors = {
        "resnet50": ("#1f77b4", "#6baed6"),
        "densenet121": ("#d62728", "#fc9272"),
    }
    display_map = {"resnet50": "ResNet50", "densenet121": "DenseNet121"}

    for hist_path in history_files:
        model_key = hist_path.stem.replace("_history", "").lower()
        df = pd.read_csv(hist_path)
        label_name = display_map.get(model_key, model_key)
        c_train, c_val = colors.get(model_key, ("#2ca02c", "#98df8a"))

        # Accuracy Grafiği
        axes[0].plot(
            df["epoch"],
            df["train_acc"] * 100.0,
            color=c_train,
            linestyle="-",
            linewidth=2.0,
            marker="o",
            markersize=3.5,
            label=f"{label_name} Train Acc",
        )
        axes[0].plot(
            df["epoch"],
            df["val_acc"] * 100.0,
            color=c_val,
            linestyle="--",
            linewidth=2.0,
            marker="s",
            markersize=3.5,
            label=f"{label_name} Val Acc",
        )

        # Loss Grafiği
        axes[1].plot(
            df["epoch"],
            df["train_loss"],
            color=c_train,
            linestyle="-",
            linewidth=2.0,
            marker="o",
            markersize=3.5,
            label=f"{label_name} Train Loss",
        )
        axes[1].plot(
            df["epoch"],
            df["val_loss"],
            color=c_val,
            linestyle="--",
            linewidth=2.0,
            marker="s",
            markersize=3.5,
            label=f"{label_name} Val Loss",
        )

    axes[0].set_title("Epoch Bazlı Doğruluk (Accuracy) Eğrileri", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epoch", fontsize=10)
    axes[0].set_ylabel("Doğruluk (%)", fontsize=10)
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend(frameon=True, fontsize=8.5, loc="lower right")

    axes[1].set_title("Epoch Bazlı Kayıp (Loss) Eğrileri", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Epoch", fontsize=10)
    axes[1].set_ylabel("Cross-Entropy Loss", fontsize=10)
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend(frameon=True, fontsize=8.5, loc="upper right")

    fig.suptitle(
        "Derin Öğrenme Modelleri Eğitim ve Doğrulama Performans Eğrileri",
        fontsize=13.5,
        fontweight="bold",
        y=0.99,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[GÖRSEL KAYDEDİLDİ] Eğitim eğrileri: {output_path}")
    return output_path


def _plot_confusion_matrix_grid(
    model_results: list[dict],
    title: str,
    output_path: Path,
    class_names: list[str] = config.CLASS_NAMES,
) -> Optional[Path]:
    """
    Verilen model sonuçları için hem sayısal hem de satır bazlı yüzdelik değerleri
    içeren seaborn karmaşıklık matrisi (Confusion Matrix) ısı haritalarını çizer.
    """
    if not model_results:
        return None

    n_models = len(model_results)
    n_cols = 2 if n_models > 1 else 1
    n_rows = math.ceil(n_models / n_cols)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(6.8 * n_cols, 5.6 * n_rows),
        dpi=config.FIGURE_DPI,
        squeeze=False,
    )
    flat_axes = axes.flatten()

    labels_idx = list(range(len(class_names)))
    short_names = [
        name.replace("VeryMildDemented", "VeryMild")
        .replace("MildDemented", "Mild")
        .replace("ModerateDemented", "Moderate")
        .replace("NonDemented", "NonDem")
        for name in class_names
    ]

    for idx, res in enumerate(model_results):
        ax = flat_axes[idx]
        cm = confusion_matrix(res["y_true"], res["y_pred"], labels=labels_idx)
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_pct = np.divide(cm, np.where(row_sums == 0, 1, row_sums)) * 100.0

        # Hücre içi etiketler: Sayı + Yüzdelik (örn: 142\n(94.7%))
        annot = np.empty_like(cm, dtype=object)
        for r in range(cm.shape[0]):
            for c in range(cm.shape[1]):
                annot[r, c] = f"{cm[r, c]}\n({cm_pct[r, c]:.1f}%)"

        sns.heatmap(
            cm_pct,
            annot=annot,
            fmt="",
            cmap="Blues",
            vmin=0,
            vmax=100,
            cbar=True,
            cbar_kws={"label": "Sınıf İçi Doğruluk (%)"},
            xticklabels=short_names,
            yticklabels=short_names,
            linewidths=0.6,
            linecolor="#e0e0e0",
            annot_kws={"size": 9.5, "weight": "bold"},
            ax=ax,
        )
        acc = accuracy_score(res["y_true"], res["y_pred"]) * 100.0
        ax.set_title(
            f"{res['display_name']} (Test Acc: {acc:.2f}%)",
            fontsize=11.5,
            fontweight="bold",
            pad=10,
        )
        ax.set_xlabel("Tahmin Edilen Evre", fontsize=10)
        ax.set_ylabel("Gerçek Evre", fontsize=10)
        ax.tick_params(axis="x", rotation=20)
        ax.tick_params(axis="y", rotation=0)

    # Boş kalan alt eksenleri gizle
    for j in range(n_models, len(flat_axes)):
        flat_axes[j].axis("off")

    fig.suptitle(title, fontsize=13.5, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[GÖRSEL KAYDEDİLDİ] Confusion Matrix: {output_path}")
    return output_path


def plot_roc_curves(
    dl_results: list[dict],
    hybrid_results: list[dict],
    output_path: Path = config.OUTPUTS_DIR / "roc_curves.png",
    class_names: list[str] = config.CLASS_NAMES,
) -> Optional[Path]:
    """
    En iyi DL modeli ve en iyi Hibrit ML modeli için 4 sınıflı One-vs-Rest (OvR)
    ROC eğrilerini, sınıf bazlı AUC değerlerini ve Makro-Ortalama AUC skorlarını çizer.
    """
    panels: list[tuple[str, dict]] = []
    if dl_results:
        best_dl = max(dl_results, key=lambda r: accuracy_score(r["y_true"], r["y_pred"]))
        panels.append(("En İyi Derin Öğrenme Modeli", best_dl))
    if hybrid_results:
        best_hyb = max(hybrid_results, key=lambda r: accuracy_score(r["y_true"], r["y_pred"]))
        panels.append(("En İyi Hibrit ML Modeli", best_hyb))

    if not panels:
        return None

    fig, axes = plt.subplots(
        1,
        len(panels),
        figsize=(7.0 * len(panels), 5.6),
        dpi=config.FIGURE_DPI,
        squeeze=False,
    )
    class_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for idx, (panel_title, res) in enumerate(panels):
        ax = axes[0, idx]
        y_true = res["y_true"]
        y_prob = res["y_prob"]
        n_classes = len(class_names)
        y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))

        fpr_dict: dict[int, np.ndarray] = {}
        tpr_dict: dict[int, np.ndarray] = {}
        roc_auc_dict: dict[int, float] = {}

        for c in range(n_classes):
            if np.sum(y_true_bin[:, c]) == 0 or np.sum(y_true_bin[:, c]) == len(y_true):
                roc_auc_dict[c] = float("nan")
                continue
            fpr_dict[c], tpr_dict[c], _ = roc_curve(y_true_bin[:, c], y_prob[:, c])
            roc_auc_dict[c] = float(auc(fpr_dict[c], tpr_dict[c]))
            ax.plot(
                fpr_dict[c],
                tpr_dict[c],
                color=class_colors[c % len(class_colors)],
                linewidth=1.9,
                label=f"{class_names[c]} (AUC = {roc_auc_dict[c]:.4f})",
            )

        # Makro-Ortalama ROC eğrisi hesaplama
        valid_classes = [c for c in range(n_classes) if c in fpr_dict]
        if valid_classes:
            all_fpr = np.unique(np.concatenate([fpr_dict[c] for c in valid_classes]))
            mean_tpr = np.zeros_like(all_fpr)
            for c in valid_classes:
                mean_tpr += np.interp(all_fpr, fpr_dict[c], tpr_dict[c])
            mean_tpr /= len(valid_classes)
            macro_auc = float(auc(all_fpr, mean_tpr))

            ax.plot(
                all_fpr,
                mean_tpr,
                color="#111111",
                linestyle="--",
                linewidth=2.3,
                label=f"Makro-Ortalama (AUC = {macro_auc:.4f})",
            )

        ax.plot([0, 1], [0, 1], color="#888888", linestyle=":", linewidth=1.2, label="Rastgele (AUC = 0.50)")
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.04])
        ax.set_xlabel("Yanlış Pozitif Oranı (1 - Özgüllük)", fontsize=10)
        ax.set_ylabel("Doğru Pozitif Oranı (Duyarlılık)", fontsize=10)
        ax.set_title(f"{panel_title}: {res['display_name']}", fontsize=11.5, fontweight="bold", pad=10)
        ax.grid(True, linestyle=":", alpha=0.55)
        ax.legend(loc="lower right", fontsize=8.3, frameon=True)

    fig.suptitle(
        "4-Sınıflı Alzheimer Evreleme One-vs-Rest (OvR) ROC Eğrileri",
        fontsize=13.5,
        fontweight="bold",
        y=0.99,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[GÖRSEL KAYDEDİLDİ] ROC eğrileri: {output_path}")
    return output_path


def _compute_metrics_row(res: dict, class_names: list[str] = config.CLASS_NAMES) -> dict:
    """Tek bir model için genel, sınıf bazlı ve süre metriklerini sözlük olarak hesaplar."""
    y_true = res["y_true"]
    y_pred = res["y_pred"]
    y_prob = res["y_prob"]
    labels_idx = list(range(len(class_names)))

    overall_acc = accuracy_score(y_true, y_pred) * 100.0
    macro_prec = precision_score(y_true, y_pred, average="macro", zero_division=0) * 100.0
    macro_rec = recall_score(y_true, y_pred, average="macro", zero_division=0) * 100.0
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0) * 100.0

    # Sınıf bazlı doğruluk (confusion matrix satır doğruluğu = sınıf duyarlılığı)
    cm = confusion_matrix(y_true, y_pred, labels=labels_idx)
    per_class_acc = {}
    for i, cname in enumerate(class_names):
        denom = cm[i].sum()
        per_class_acc[f"Acc_{cname} (%)"] = round((cm[i, i] / denom) * 100.0, 2) if denom > 0 else 0.0

    # Çok sınıflı Macro OvR AUC
    try:
        y_true_bin = label_binarize(y_true, classes=labels_idx)
        macro_auc = float(roc_auc_score(y_true_bin, y_prob, average="macro", multi_class="ovr"))
    except ValueError:
        macro_auc = float("nan")

    row = {
        "Model": res["display_name"],
        "Accuracy (%)": round(overall_acc, 2),
        **per_class_acc,
        "Macro_Precision (%)": round(macro_prec, 2),
        "Macro_Recall (%)": round(macro_rec, 2),
        "Macro_F1 (%)": round(macro_f1, 2),
        "Macro_AUC": round(macro_auc, 4) if not np.isnan(macro_auc) else np.nan,
        "Train_Time_s": round(float(res.get("train_time_sec", 0.0)), 2),
        "Inference_Time_s": round(float(res.get("inference_time_sec", 0.0)), 4),
    }
    return row


def evaluate_all_models(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    batch_size: int = config.BATCH_SIZE,
) -> pd.DataFrame:
    """
    Tüm eğitilmiş DL ve Hibrit ML modellerini Test seti üzerinde değerlendirir,
    zorunlu akademik grafik ve CSV tablolarını outputs/ klasörüne üretir.
    """
    config.set_seed(config.SEED)
    device = config.get_device()

    # 1. Ön işleme karşılaştırma görseli henüz üretilmediyse Test setinden örnek ile üret
    prep_fig_path = config.OUTPUTS_DIR / "preprocessing_comparison.png"
    if not prep_fig_path.exists() and len(test_df) > 0:
        sample_row = test_df.iloc[0]
        generate_preprocessing_comparison(
            image_path=sample_row["filepath"],
            output_path=prep_fig_path,
            sample_label=sample_row["label_name"],
        )

    # 2. Eğitim eğrilerini çiz
    plot_training_curves()

    # 3. Derin Öğrenme modellerinin Test seti değerlendirmesi
    loaders = create_dataloaders(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        batch_size=batch_size,
    )
    test_loader = loaders["test"]

    dl_results: list[dict] = []
    for model_name in config.DL_MODELS_TO_TRAIN:
        ckpt_path = config.CHECKPOINTS_DIR / f"{model_name}_best.pt"
        if ckpt_path.exists():
            print(f"[DEĞERLENDİRME] DL modeli test ediliyor: {model_name}")
            dl_results.append(predict_dl_model(ckpt_path, test_loader, device=device))
        else:
            print(f"[UYARI] Checkpoint bulunamadı, atlanıyor: {ckpt_path}")

    # 4. Hibrit ML modellerinin Test seti sonuçlarını yükle
    hybrid_results: list[dict] = []
    hybrid_preds_path = config.HYBRID_MODELS_DIR / "hybrid_test_predictions.joblib"
    if hybrid_preds_path.exists():
        preds_bundle = joblib.load(hybrid_preds_path)
        for model_key, data in preds_bundle.items():
            hybrid_results.append(
                {
                    "display_name": model_key,
                    "y_true": data["y_true"],
                    "y_pred": data["y_pred"],
                    "y_prob": data["y_prob"],
                    "train_time_sec": data["train_time_sec"],
                    "inference_time_sec": data["inference_time_sec"],
                }
            )
    else:
        print(f"[UYARI] Hibrit test tahminleri bulunamadı: {hybrid_preds_path}")

    # 5. Confusion Matrix görsellerini oluştur
    if dl_results:
        _plot_confusion_matrix_grid(
            model_results=dl_results,
            title="Model 1 (End-to-End Derin Öğrenme) Karmaşıklık Matrisleri",
            output_path=config.OUTPUTS_DIR / "confusion_matrix_dl.png",
        )
    if hybrid_results:
        _plot_confusion_matrix_grid(
            model_results=hybrid_results,
            title="Model 2 (Hibrit CNN + Öznitelik Seçimi + ML) Karmaşıklık Matrisleri",
            output_path=config.OUTPUTS_DIR / "confusion_matrix_hybrid.png",
        )

    # 6. ROC eğrilerini oluştur
    plot_roc_curves(dl_results=dl_results, hybrid_results=hybrid_results)

    # 7. Karşılaştırma tablosu oluştur, CSV olarak kaydet ve konsola yazdır
    all_results = dl_results + hybrid_results
    if not all_results:
        raise RuntimeError("Değerlendirilecek hiçbir eğitilmiş model sonucu bulunamadı.")

    rows = [_compute_metrics_row(res) for res in all_results]
    comparison_df = pd.DataFrame(rows)

    csv_path = config.OUTPUTS_DIR / "model_comparison_table.csv"
    comparison_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 95)
    print("DENEYSEL SONUÇLAR: TÜM MODELLERİN TEST SETİ KARŞILAŞTIRMA TABLOSU")
    print("=" * 95)
    print(comparison_df.to_string(index=False))
    print("=" * 95)
    print(f"[TABLO KAYDEDİLDİ] {csv_path}")

    return comparison_df
