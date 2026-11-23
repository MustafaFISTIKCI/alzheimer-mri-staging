"""
Alzheimer MRI Veri Seti Tarama ve Stratified Bölünme Modülü (data/split_dataset.py)

Kaggle'dan indirilen 'alzheimers-multiclass-dataset-equal-and-augmented' veri setinin
alt klasörlerini otomatik olarak tarar, sınıf etiketlerini standartlaştırır ve
veriyi Stratified %70 Eğitim (Train) / %15 Doğrulama (Validation) / %15 Test
olacak şekilde böler. Oluşturulan bölünme tablolarını CSV olarak kaydeder.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
from sklearn.model_selection import train_test_split

# Proje kök dizinini Python arama yoluna ekle (doğrudan çalıştırma desteği için)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config


def _normalize_label(folder_name: str) -> Optional[str]:
    """
    Klasör adını normalize ederek standart Alzheimer sınıf adına eşler.
    Örn: 'Very Mild Demented' veya 'very_mild_demented' -> 'VeryMildDemented'
    """
    cleaned = folder_name.strip().lower().replace("-", " ").replace("_", " ")
    # Birden fazla boşluğu teke indir ve boşluksuz halini de kontrol et
    collapsed = " ".join(cleaned.split())
    compact = collapsed.replace(" ", "")

    if collapsed in config.CLASS_ALIASES:
        return config.CLASS_ALIASES[collapsed]
    if compact in config.CLASS_ALIASES:
        return config.CLASS_ALIASES[compact]
    return None


def scan_dataset_directory(raw_data_dir: Path | str) -> pd.DataFrame:
    """
    Verilen kök dizin altındaki tüm klasörleri özyinelemeli (recursive) tarayarak
    desteklenen MRI görüntülerini ve ait oldukları sınıf etiketlerini toplar.

    Args:
        raw_data_dir: Ham veri seti klasör yolu.

    Returns:
        'filepath', 'label_name', 'label' sütunlarını içeren DataFrame.
    """
    root = Path(raw_data_dir)
    if not root.exists():
        raise FileNotFoundError(
            f"Veri seti klasörü bulunamadı: {root}\n"
            f"Lütfen Kaggle veri setini '{config.RAW_DATA_DIR}' konumuna yerleştirin "
            f"veya config.py / --data-dir üzerinden doğru yolu belirtin."
        )

    class_to_idx = {name: idx for idx, name in enumerate(config.CLASS_NAMES)}
    records: list[dict] = []
    seen_paths: set[str] = set()

    for dirpath, _, filenames in os.walk(root):
        current_dir = Path(dirpath)
        # Klasör yolundaki parçalardan (en içten dışa) sınıf adını tespit et
        matched_class: Optional[str] = None
        for part in reversed(current_dir.relative_to(root).parts):
            matched_class = _normalize_label(part)
            if matched_class is not None:
                break

        if matched_class is None:
            continue

        for fname in sorted(filenames):
            ext = Path(fname).suffix.lower()
            if ext not in config.SUPPORTED_EXTENSIONS:
                continue

            full_path = (current_dir / fname).resolve()
            path_str = str(full_path)
            if path_str in seen_paths:
                continue
            seen_paths.add(path_str)

            records.append(
                {
                    "filepath": path_str,
                    "label_name": matched_class,
                    "label": class_to_idx[matched_class],
                }
            )

    if not records:
        raise RuntimeError(
            f"'{root}' altında desteklenen uzantılarda ({config.SUPPORTED_EXTENSIONS}) "
            f"ve beklenen sınıf klasörlerinde ({config.CLASS_NAMES}) hiç görüntü bulunamadı."
        )

    df = pd.DataFrame(records)
    return df


def download_with_kagglehub(
    dataset_slug: str = config.KAGGLE_DATASET_SLUG,
) -> Path:
    """
    kagglehub kullanarak veri setini indirir (veya önbellekteki yolu döndürür).
    """
    try:
        import kagglehub
    except ImportError as exc:
        raise ImportError(
            "kagglehub paketi kurulu değil. Kurmak için: pip install kagglehub"
        ) from exc

    print(f"[BİLGİ] kagglehub ile veri seti indiriliyor/doğrulanıyor: {dataset_slug}")
    downloaded_path = Path(kagglehub.dataset_download(dataset_slug))
    print(f"[BAŞARILI] Veri seti konumu: {downloaded_path}")
    return downloaded_path


def create_stratified_splits(
    raw_data_dir: Path | str = config.RAW_DATA_DIR,
    splits_dir: Path | str = config.SPLITS_DIR,
    train_ratio: float = config.TRAIN_RATIO,
    val_ratio: float = config.VAL_RATIO,
    test_ratio: float = config.TEST_RATIO,
    seed: int = config.SEED,
    force_resplit: bool = False,
    download_kaggle: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Veri setini Stratified %70 Train / %15 Validation / %15 Test olarak böler
    ve CSV dosyalarını (train.csv, val.csv, test.csv) kaydeder.

    Returns:
        (train_df, val_df, test_df) demeti.
    """
    config.set_seed(seed)

    if not abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-6:
        raise ValueError("Train, Validation ve Test oranlarının toplamı 1.0 olmalıdır.")

    splits_path = Path(splits_dir)
    splits_path.mkdir(parents=True, exist_ok=True)

    train_csv = splits_path / "train.csv"
    val_csv = splits_path / "val.csv"
    test_csv = splits_path / "test.csv"

    # Eğer bölünme dosyaları mevcutsa ve yeniden bölme istenmiyorsa doğrudan yükle
    if not force_resplit and train_csv.exists() and val_csv.exists() and test_csv.exists():
        train_df = pd.read_csv(train_csv)
        val_df = pd.read_csv(val_csv)
        test_df = pd.read_csv(test_csv)
        print(f"[BİLGİ] Mevcut veri bölünmeleri yüklendi: {splits_path}")
        _print_split_summary(train_df, val_df, test_df)
        return train_df, val_df, test_df

    raw_path = Path(raw_data_dir)
    if download_kaggle or (not raw_path.exists()):
        if download_kaggle:
            raw_path = download_with_kagglehub()
        else:
            raise FileNotFoundError(
                f"Veri seti klasörü bulunamadı: {raw_path}\n"
                f"Seçenek 1: ZIP dosyasını '{config.RAW_DATA_DIR}' konumuna çıkarın.\n"
                f"Seçenek 2: Otomatik indirme için '--download-kaggle' parametresiyle çalıştırın."
            )

    print(f"[BİLGİ] Veri seti taranıyor: {raw_path}")
    full_df = scan_dataset_directory(raw_path)

    # Sınıf dağılımı kontrolü
    found_classes = set(full_df["label_name"].unique())
    missing_classes = set(config.CLASS_NAMES) - found_classes
    if missing_classes:
        print(f"[UYARI] Veri setinde eksik sınıflar tespit edildi: {sorted(missing_classes)}")

    # 1. Adım: Train (%70) ve Geçici Set (%30 -> Val + Test)
    temp_ratio = val_ratio + test_ratio
    train_df, temp_df = train_test_split(
        full_df,
        test_size=temp_ratio,
        stratify=full_df["label"],
        random_state=seed,
    )

    # 2. Adım: Geçici Seti eşit şekilde Validation (%15) ve Test (%15) olarak böl
    relative_test_ratio = test_ratio / temp_ratio
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_ratio,
        stratify=temp_df["label"],
        random_state=seed,
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    print(f"[BAŞARILI] Stratified bölünme tamamlandı ve '{splits_path}' dizinine kaydedildi.")
    _print_split_summary(train_df, val_df, test_df)

    return train_df, val_df, test_df


def _print_split_summary(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
) -> None:
    """Eğitim, doğrulama ve test setlerinin sınıf bazlı dağılımını konsola yazdırır."""
    total = len(train_df) + len(val_df) + len(test_df)
    print(f"Toplam Görüntü: {total} | Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    summary = pd.DataFrame(
        {
            "Train": train_df["label_name"].value_counts(),
            "Validation": val_df["label_name"].value_counts(),
            "Test": test_df["label_name"].value_counts(),
        }
    ).reindex(config.CLASS_NAMES).fillna(0).astype(int)
    summary["Toplam"] = summary.sum(axis=1)
    print(summary.to_string())
    print("-" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Alzheimer MRI Veri Seti Bölünme Aracı")
    parser.add_argument("--download-kaggle", action="store_true", help="Veri setini kagglehub ile indirir.")
    parser.add_argument("--force-resplit", action="store_true", help="Mevcut CSV dosyalarını yeniden oluşturur.")
    args = parser.parse_args()
    create_stratified_splits(force_resplit=args.force_resplit, download_kaggle=args.download_kaggle)
