"""
Alzheimer MRI Evreleme (4 Sınıf) - Ana Çalıştırma Dosyası (main.py)

Tüm uçtan uca hattı tek komutla veya belirli adımlar (--steps) seçilerek çalıştırır:
1. split      : Veri seti alt klasörlerini tarar ve Stratified %70/%15/%15 böler
2. preprocess : Örnek MRI görüntüsü üzerinde Orijinal vs CLAHE+Gaussian karşılaştırması üretir
3. train_dl   : ResNet50 ve DenseNet121 modellerini AMP + EarlyStopping ile eğitir
4. hybrid     : En iyi CNN'den özellik çıkarır, PCA ve Fisher Score + SVM/XGBoost eğitir
5. evaluate   : 300 DPI grafikler ve model_comparison_table.csv dosyasını outputs/ altına üretir

Örnek Kullanım:
    python main.py
    python main.py --data-dir "C:/veriler/alzheimers" --epochs 25 --batch-size 32
    python main.py --steps train_dl hybrid evaluate
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

import config
from data.split_dataset import create_stratified_splits
from src.evaluate import evaluate_all_models
from src.hybrid_ml import run_hybrid_ml_pipeline
from src.preprocessing import generate_preprocessing_comparison
from src.train_dl import train_all_dl_models


ALL_STEPS = ("split", "preprocess", "train_dl", "hybrid", "evaluate")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Alzheimer MRI 4-Sınıflı Evreleme (DL + Hibrit ML) Pipeline"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(config.RAW_DATA_DIR),
        help="Kaggle ham veri seti klasör yolu.",
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        default=["all"],
        choices=["all", *ALL_STEPS],
        help="Çalıştırılacak pipeline adımları (varsayılan: all).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=config.EPOCHS,
        help="DL modelleri için maksimum epoch sayısı.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=config.BATCH_SIZE,
        help="DataLoader batch boyutu.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=config.LEARNING_RATE,
        help="AdamW başlangıç öğrenme oranı.",
    )
    parser.add_argument(
        "--download-kaggle",
        action="store_true",
        help="Veri setini kagglehub kullanarak otomatik indirir.",
    )
    parser.add_argument(
        "--force-resplit",
        action="store_true",
        help="Mevcut train/val/test CSV dosyalarını yeniden oluşturur.",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="ImageNet ön eğitimli ağırlıklar yerine rastgele başlatma kullanır (çevrimdışı testler için).",
    )
    parser.add_argument(
        "--fast-grid",
        action="store_true",
        help="Hibrit ML GridSearchCV arama uzayını küçülterek hızlı çalıştırır.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config.set_seed(config.SEED)
    device = config.get_device()

    selected_steps = list(ALL_STEPS) if "all" in args.steps else args.steps

    device_desc = f"cuda ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else "cpu"
    print("=" * 75)
    print("ALZHEIMER MRI EVRELEME PROJESİ (PyTorch DL + Hibrit ML)")
    print(f"Cihaz: {device_desc}")
    print(f"Sabit Seed: {config.SEED} | Adımlar: {selected_steps}")
    print("=" * 75)

    # 1. Veri seti bölünmesi (diğer adımlar da train/val/test DataFrame'lerine ihtiyaç duyar)
    train_df, val_df, test_df = create_stratified_splits(
        raw_data_dir=Path(args.data_dir),
        force_resplit=args.force_resplit,
        download_kaggle=args.download_kaggle,
    )

    # 2. Ön işleme görsel karşılaştırması (Orijinal vs CLAHE + Gaussian)
    if "preprocess" in selected_steps:
        # Mümkünse orta/hafif demans örneği seçerek doku kontrast farkını net göster
        sample_candidates = train_df[train_df["label_name"] == "ModerateDemented"]
        sample_row = sample_candidates.iloc[0] if len(sample_candidates) > 0 else train_df.iloc[0]
        generate_preprocessing_comparison(
            image_path=sample_row["filepath"],
            output_path=config.OUTPUTS_DIR / "preprocessing_comparison.png",
            sample_label=sample_row["label_name"],
        )

    # 3. Model 1: End-to-End Derin Öğrenme (ResNet50 & DenseNet121)
    if "train_dl" in selected_steps:
        train_all_dl_models(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            pretrained=not args.no_pretrained,
        )

    # 4. Model 2: Hibrit ML (En İyi CNN Feature Extraction -> PCA / Fisher Score -> SVM & XGBoost)
    if "hybrid" in selected_steps:
        run_hybrid_ml_pipeline(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            batch_size=args.batch_size,
            fast_grid=args.fast_grid,
        )

    # 5. Değerlendirme ve Akademik Rapor Çıktıları
    if "evaluate" in selected_steps:
        evaluate_all_models(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            batch_size=args.batch_size,
        )

    print("\n[PİPELİNE TAMAMLANDI] Tüm çıktılar 'outputs/' klasörüne kaydedildi.")


if __name__ == "__main__":
    main()
