"""
Hibrit Derin Öğrenme + Makine Öğrenmesi Modülü (src/hybrid_ml.py)

Model 2 Akışı:
1. En iyi eğitilmiş CNN modelini (varsayılan olarak DenseNet121 veya en yüksek
   doğrulama başarımına sahip model) yükler.
2. Son global havuzlama (Global Average Pooling) katmanından derin öznitelikleri
   (Train, Validation, Test) çıkarır.
3. Öznitelikleri standartlaştırır (StandardScaler - yalnızca Train setine fit edilir).
4. İki farklı boyut indirgeme / öznitelik seçimi stratejisi uygular:
   - PCA (Principal Component Analysis): Varyansın %95'ini koruyan ortogonal bileşenler
   - SelectKBest + Fisher Score: Sınıflar arası ayrıştırıcılığı (Fisher oranı) en yüksek K özellik
5. Seçilen özellikler üzerinde GridSearchCV ile hiperparametre optimizasyonu yaparak
   SVM (RBF çekirdek) ve XGBoost sınıflandırıcılarını eğitir ve Test setinde değerlendirir.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Callable, Optional

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader
from tqdm import tqdm
from xgboost import XGBClassifier

import config
from src.dataset import AlzheimerMRIDataset
from src.models import build_model, get_display_name


def fisher_score_func(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Çok sınıflı Fisher Skoru (Fisher Score) hesaplar.
    Her özellik j için sınıflar arası varyansın sınıf içi varyansa oranını döndürür:
        F(j) = sum_c n_c * (mu_c,j - mu_j)^2 / (sum_c n_c * sigma_c,j^2 + eps)

    SelectKBest(score_func=fisher_score_func, k=...) ile doğrudan uyumludur.
    """
    classes = np.unique(y)
    global_mean = np.mean(X, axis=0)

    numerator = np.zeros(X.shape[1], dtype=np.float64)
    denominator = np.zeros(X.shape[1], dtype=np.float64)

    for cls in classes:
        X_c = X[y == cls]
        n_c = X_c.shape[0]
        if n_c == 0:
            continue
        mean_c = np.mean(X_c, axis=0)
        var_c = np.var(X_c, axis=0)
        numerator += n_c * (mean_c - global_mean) ** 2
        denominator += n_c * var_c

    scores = numerator / (denominator + 1e-8)
    # SelectKBest uyumluluğu için (scores, pvalues_placeholder) döndürülür
    pvalues = np.zeros_like(scores)
    return scores, pvalues


def _load_best_cnn_info() -> tuple[str, Path]:
    """
    Eğitilmiş DL modellerinin özetinden en iyi CNN modelinin adını ve checkpoint yolunu döndürür.
    """
    summary_path = config.OUTPUTS_DIR / "dl_training_summary.json"
    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        best_name = summary["best_cnn_model"]
        ckpt_path = Path(summary["models"][best_name]["checkpoint_path"])
        return best_name, ckpt_path

    # Özet dosyası yoksa mevcut checkpoint'lerden val_acc'a göre en iyisini seç
    candidates = list(config.CHECKPOINTS_DIR.glob("*_best.pt"))
    if not candidates:
        raise FileNotFoundError(
            "Eğitilmiş CNN checkpoint dosyası bulunamadı. Önce DL eğitimini (train_dl) çalıştırın."
        )

    best_ckpt_path = None
    best_name = ""
    best_score = -1.0
    for path in candidates:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        val_acc = float(ckpt.get("best_val_acc", 0.0))
        if val_acc >= best_score:
            best_score = val_acc
            best_name = ckpt["model_name"]
            best_ckpt_path = path

    return best_name, best_ckpt_path  # type: ignore[return-value]


@torch.no_grad()
def extract_cnn_features(
    checkpoint_path: Path | str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    batch_size: int = config.BATCH_SIZE,
    device: Optional[torch.device] = None,
) -> dict[str, np.ndarray]:
    """
    Eğitilmiş CNN modelinin son havuzlama katmanından Train, Val ve Test kümeleri için
    deterministik öznitelik matrislerini çıkarır ve .npz olarak kaydeder.
    """
    device = device or config.get_device()
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_name = ckpt["model_name"]
    display_name = ckpt.get("display_name", get_display_name(model_name))

    model = build_model(
        model_name=model_name,
        num_classes=config.NUM_CLASSES,
        dropout_rate=config.DROPOUT_RATE,
        pretrained=False,
        device=device,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    pin_memory = device.type == "cuda"
    split_dfs = {"train": train_df, "val": val_df, "test": test_df}
    extracted: dict[str, np.ndarray] = {}

    print(f"\n[ÖZNİTELİK ÇIKARIMI] {display_name} son havuzlama katmanından özellikler çıkarılıyor...")
    start_time = time.perf_counter()

    for split_name, df in split_dfs.items():
        # Öznitelik çıkarımında veri artırma (augmentation) kapalı tutulur
        ds = AlzheimerMRIDataset(df, is_train=False)
        loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=pin_memory,
        )

        feats_list: list[np.ndarray] = []
        labels_list: list[np.ndarray] = []
        for images, labels in tqdm(loader, desc=f"Feature Extract [{split_name}]", leave=False):
            images = images.to(device, non_blocking=True)
            feats = model.extract_features(images)
            feats_list.append(feats.cpu().numpy())
            labels_list.append(labels.numpy())

        extracted[f"X_{split_name}"] = np.concatenate(feats_list, axis=0)
        extracted[f"y_{split_name}"] = np.concatenate(labels_list, axis=0)

    extract_time = time.perf_counter() - start_time
    extracted["extract_time_sec"] = np.array([extract_time], dtype=np.float64)

    feat_path = config.FEATURES_DIR / f"{model_name}_features.npz"
    np.savez_compressed(feat_path, **extracted)
    print(
        f"[TAMAMLANDI] Özellik boyutları -> Train: {extracted['X_train'].shape}, "
        f"Val: {extracted['X_val'].shape}, Test: {extracted['X_test'].shape} "
        f"({extract_time:.2f} sn)"
    )

    return extracted


def _apply_pca_selection(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    variance_ratio: float = config.PCA_VARIANCE_RATIO,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, PCA]:
    """
    Standartlaştırılmış özellikler üzerinde PCA uygular.
    Küçük/sentetik veri setlerinde bileşen sayısı sınırını otomatik korur.
    """
    max_components = min(X_train.shape[0], X_train.shape[1])
    if max_components <= 2:
        n_components: int | float = max_components
    else:
        n_components = variance_ratio

    pca = PCA(n_components=n_components, random_state=config.SEED)
    X_train_pca = pca.fit_transform(X_train)
    X_val_pca = pca.transform(X_val)
    X_test_pca = pca.transform(X_test)
    return X_train_pca, X_val_pca, X_test_pca, pca


def _apply_fisher_selection(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    candidates: list[int] = config.SELECT_KBEST_CANDIDATES,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, SelectKBest]:
    """
    SelectKBest + Fisher Score ile en yüksek ayrıştırıcılığa sahip K özelliği seçer.
    En uygun K değerini doğrulama (Validation) setinde hızlı doğrusal/RBF SVC kontrolü ile belirler.
    """
    n_features = X_train.shape[1]
    valid_ks = sorted({min(k, n_features) for k in candidates if min(k, n_features) > 0})
    if not valid_ks:
        valid_ks = [n_features]

    best_k = valid_ks[0]
    best_val_acc = -1.0
    best_selector: Optional[SelectKBest] = None

    for k in valid_ks:
        selector = SelectKBest(score_func=fisher_score_func, k=k)
        X_tr_k = selector.fit_transform(X_train, y_train)
        X_va_k = selector.transform(X_val)

        # Hızlı doğrulama skoru ile en iyi K seçimi
        probe = SVC(kernel="rbf", C=1.0, class_weight="balanced", random_state=config.SEED)
        probe.fit(X_tr_k, y_train)
        acc = accuracy_score(y_val, probe.predict(X_va_k))
        if acc > best_val_acc:
            best_val_acc = acc
            best_k = k
            best_selector = selector

    assert best_selector is not None
    X_train_sel = best_selector.transform(X_train)
    X_val_sel = best_selector.transform(X_val)
    X_test_sel = best_selector.transform(X_test)
    print(f"[FİSHER SEÇİMİ] Seçilen optimal özellik sayısı K={best_k} (Val Probe Acc: {best_val_acc*100:.2f}%)")
    return X_train_sel, X_val_sel, X_test_sel, best_selector


def _optimize_and_evaluate_classifier(
    clf_name: str,
    base_estimator,
    param_grid: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    cv_folds: int = config.CV_FOLDS,
) -> dict:
    """
    GridSearchCV ile hiperparametre optimizasyonu yapar, en iyi sınıflandırıcıyı eğitir
    ve Test seti üzerinde tahmin + olasılık + süre metriklerini döndürür.
    """
    # Küçük veri setlerinde sınıf başına düşen örnek sayısına göre CV kat sayısını güvenli tut
    min_class_count = int(np.bincount(y_train).min())
    effective_folds = max(2, min(cv_folds, min_class_count))
    cv = StratifiedKFold(n_splits=effective_folds, shuffle=True, random_state=config.SEED)

    start_train = time.perf_counter()
    # Windows üzerinde loky alt süreçlerinin CUDA DLL'lerini tekrar yükleyip WinError 1455
    # (disk belleği sınırı) hatasına yol açmasını önlemek için tek süreçte çalıştırılır.
    grid_n_jobs = 1 if os.name == "nt" else -1
    grid = GridSearchCV(
        estimator=base_estimator,
        param_grid=param_grid,
        cv=cv,
        scoring="f1_macro",
        n_jobs=grid_n_jobs,
        refit=True,
    )
    grid.fit(X_train, y_train)
    train_time_sec = time.perf_counter() - start_train

    best_model = grid.best_estimator_
    val_acc = float(accuracy_score(y_val, best_model.predict(X_val)))

    start_inf = time.perf_counter()
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)
    inference_time_sec = time.perf_counter() - start_inf

    test_acc = float(accuracy_score(y_test, y_pred))
    print(
        f"  -> {clf_name}: CV Best F1={grid.best_score_*100:.2f}% | "
        f"Val Acc={val_acc*100:.2f}% | Test Acc={test_acc*100:.2f}% | "
        f"Params={grid.best_params_}"
    )

    return {
        "model": best_model,
        "best_params": grid.best_params_,
        "cv_best_f1_macro": float(grid.best_score_),
        "val_acc": val_acc,
        "test_acc": test_acc,
        "y_true": y_test,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "train_time_sec": float(train_time_sec),
        "inference_time_sec": float(inference_time_sec),
    }


def run_hybrid_ml_pipeline(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cnn_model_name: Optional[str] = None,
    batch_size: int = config.BATCH_SIZE,
    fast_grid: bool = False,
) -> dict:
    """
    Model 2 uçtan uca hibrit makine öğrenmesi pipeline'ını çalıştırır:
    En iyi CNN -> Feature Extraction -> StandardScaler -> (PCA ve SelectKBest/Fisher Score)
    -> SVM (RBF) ve XGBoost GridSearchCV eğitimi.
    """
    config.set_seed(config.SEED)
    device = config.get_device()

    if cnn_model_name is None:
        cnn_model_name, ckpt_path = _load_best_cnn_info()
    else:
        ckpt_path = config.CHECKPOINTS_DIR / f"{cnn_model_name}_best.pt"

    cnn_display = get_display_name(cnn_model_name)

    # 1. Son havuzlama katmanından öznitelik çıkarımı
    feats = extract_cnn_features(
        checkpoint_path=ckpt_path,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        batch_size=batch_size,
        device=device,
    )

    X_train_raw, y_train = feats["X_train"], feats["y_train"]
    X_val_raw, y_val = feats["X_val"], feats["y_val"]
    X_test_raw, y_test = feats["X_test"], feats["y_test"]
    extract_time_sec = float(feats["extract_time_sec"][0])

    # 2. Ölçeklendirme (yalnızca Train setine fit edilir)
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train_raw)
    X_val_sc = scaler.transform(X_val_raw)
    X_test_sc = scaler.transform(X_test_raw)

    # 3A. PCA ile Öznitelik Seçimi / Boyut İndirgeme
    t_pca = time.perf_counter()
    X_tr_pca, X_va_pca, X_te_pca, pca_obj = _apply_pca_selection(X_train_sc, X_val_sc, X_test_sc)
    pca_time = time.perf_counter() - t_pca
    print(
        f"[PCA] {X_train_sc.shape[1]} boyut -> {X_tr_pca.shape[1]} bileşen "
        f"(Açıklanan Varyans: {pca_obj.explained_variance_ratio_.sum()*100:.2f}%)"
    )

    # 3B. SelectKBest (Fisher Score) ile Öznitelik Seçimi
    t_fish = time.perf_counter()
    X_tr_fish, X_va_fish, X_te_fish, fisher_obj = _apply_fisher_selection(
        X_train_sc, y_train, X_val_sc, y_val, X_test_sc
    )
    fisher_time = time.perf_counter() - t_fish

    # Hiperparametre arama uzayları (smoke test / hızlı mod seçeneğiyle)
    if fast_grid:
        svm_grid = {"C": [1.0, 10.0], "gamma": ["scale"], "kernel": ["rbf"]}
        xgb_grid = {"n_estimators": [50], "max_depth": [3], "learning_rate": [0.1]}
    else:
        svm_grid = config.SVM_PARAM_GRID
        xgb_grid = config.XGB_PARAM_GRID

    feature_branches = {
        "PCA": (X_tr_pca, X_va_pca, X_te_pca, pca_obj, pca_time),
        "Fisher": (X_tr_fish, X_va_fish, X_te_fish, fisher_obj, fisher_time),
    }

    hybrid_outputs: dict[str, dict] = {}
    predictions_bundle: dict[str, dict] = {}

    for selector_name, (X_tr, X_va, X_te, selector_obj, sel_time) in feature_branches.items():
        print(f"\n{'-' * 65}")
        print(f"[HİBRİT EĞİTİM] Özellik Seçimi: {cnn_display} + {selector_name} ({X_tr.shape[1]} özellik)")
        print(f"{'-' * 65}")

        # SVM (RBF)
        svm_name = f"{cnn_display}+{selector_name}+SVM"
        svm_est = SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=config.SEED)
        svm_res = _optimize_and_evaluate_classifier(
            clf_name=svm_name,
            base_estimator=svm_est,
            param_grid=svm_grid,
            X_train=X_tr,
            y_train=y_train,
            X_val=X_va,
            y_val=y_val,
            X_test=X_te,
            y_test=y_test,
        )

        # XGBoost (Windows'ta OpenMP iş parçacıkları alt süreç açmadan güvenli paralel çalışır)
        xgb_name = f"{cnn_display}+{selector_name}+XGBoost"
        xgb_est = XGBClassifier(
            objective="multi:softprob",
            num_class=config.NUM_CLASSES,
            eval_metric="mlogloss",
            random_state=config.SEED,
            n_jobs=-1 if os.name == "nt" else 1,
        )
        xgb_res = _optimize_and_evaluate_classifier(
            clf_name=xgb_name,
            base_estimator=xgb_est,
            param_grid=xgb_grid,
            X_train=X_tr,
            y_train=y_train,
            X_val=X_va,
            y_val=y_val,
            X_test=X_te,
            y_test=y_test,
        )

        for model_key, res in [(svm_name, svm_res), (xgb_name, xgb_res)]:
            safe_filename = model_key.replace("+", "_")
            bundle_path = config.HYBRID_MODELS_DIR / f"{safe_filename}.joblib"
            joblib.dump(
                {
                    "scaler": scaler,
                    "selector": selector_obj,
                    "classifier": res["model"],
                    "cnn_model_name": cnn_model_name,
                    "selector_name": selector_name,
                    "class_names": config.CLASS_NAMES,
                },
                bundle_path,
            )

            total_train_time = extract_time_sec + sel_time + res["train_time_sec"]
            hybrid_outputs[model_key] = {
                "display_name": model_key,
                "selector": selector_name,
                "n_selected_features": int(X_tr.shape[1]),
                "best_params": res["best_params"],
                "cv_best_f1_macro": res["cv_best_f1_macro"],
                "val_acc": res["val_acc"],
                "test_acc": res["test_acc"],
                "train_time_sec": float(total_train_time),
                "inference_time_sec": float(res["inference_time_sec"]),
                "artifact_path": str(bundle_path),
            }
            predictions_bundle[model_key] = {
                "y_true": res["y_true"],
                "y_pred": res["y_pred"],
                "y_prob": res["y_prob"],
                "train_time_sec": float(total_train_time),
                "inference_time_sec": float(res["inference_time_sec"]),
            }

    # Özet JSON ve tahmin dizilerini değerlendirme modülü (evaluate.py) için kaydet
    summary_json_path = config.HYBRID_MODELS_DIR / "hybrid_results.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "base_cnn_model": cnn_model_name,
                "base_cnn_display": cnn_display,
                "models": hybrid_outputs,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    preds_npz_path = config.HYBRID_MODELS_DIR / "hybrid_test_predictions.joblib"
    joblib.dump(predictions_bundle, preds_npz_path)

    print(f"\n[BAŞARILI] Hibrit modeller ve test tahminleri kaydedildi: {config.HYBRID_MODELS_DIR}")
    return hybrid_outputs
