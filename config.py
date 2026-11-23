"""
Alzheimer MRI Evreleme Projesi - Merkezi Yapılandırma Dosyası (config.py)
Tüm dosya yolları, rastgelelik tohumu (seed), donanım ayarları, ön işleme ve
model/eğitim hiperparametreleri bu dosyadan merkezi olarak yönetilir.
"""

import os
import random
from pathlib import Path
import numpy as np
import torch


# ==============================================================================
# 1. PROJE DİZİN VE VERİ SETİ YOLLARI
# ==============================================================================
BASE_DIR: Path = Path(__file__).resolve().parent

# Kaggle'dan indirilen 'alzheimers-multiclass-dataset-equal-and-augmented' veri seti yolu
# Kullanıcı farklı bir dizin kullanacaksa yalnızca RAW_DATA_DIR değişkenini güncellemesi yeterlidir.
DATA_DIR: Path = BASE_DIR / "data"
KAGGLE_DATASET_SLUG: str = "aryansinghal10/alzheimers-multiclass-dataset-equal-and-augmented"
RAW_DATA_DIR: Path = DATA_DIR / "alzheimers-multiclass-dataset-equal-and-augmented"
SPLITS_DIR: Path = DATA_DIR / "splits"

# Çıktı, model ağırlıkları ve ara özellik (feature) kayıt dizinleri
OUTPUTS_DIR: Path = BASE_DIR / "outputs"
CHECKPOINTS_DIR: Path = OUTPUTS_DIR / "checkpoints"
FEATURES_DIR: Path = OUTPUTS_DIR / "features"
HYBRID_MODELS_DIR: Path = OUTPUTS_DIR / "hybrid_models"

# Gerekli klasörlerin otomatik oluşturulması
for _dir in [DATA_DIR, SPLITS_DIR, OUTPUTS_DIR, CHECKPOINTS_DIR, FEATURES_DIR, HYBRID_MODELS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# 2. SINIF ETİKETLERİ VE VERİ BÖLÜNME ORANLARI
# ==============================================================================
# Alzheimer evreleri (4 Sınıf - klinik şiddet sırasına göre)
CLASS_NAMES: list[str] = [
    "NonDemented",
    "VeryMildDemented",
    "MildDemented",
    "ModerateDemented",
]
NUM_CLASSES: int = len(CLASS_NAMES)

# Klasör adı eşleştirme toleransı (küçük/büyük harf veya boşluk/alt çizgi farkları için)
CLASS_ALIASES: dict[str, str] = {
    "nondemented": "NonDemented",
    "non_demented": "NonDemented",
    "non demented": "NonDemented",
    "verymilddemented": "VeryMildDemented",
    "very_mild_demented": "VeryMildDemented",
    "very mild demented": "VeryMildDemented",
    "milddemented": "MildDemented",
    "mild_demented": "MildDemented",
    "mild demented": "MildDemented",
    "moderatedemented": "ModerateDemented",
    "moderate_demented": "ModerateDemented",
    "moderate demented": "ModerateDemented",
}

# Stratified Train / Validation / Test bölünme oranları (%70 / %15 / %15)
TRAIN_RATIO: float = 0.70
VAL_RATIO: float = 0.15
TEST_RATIO: float = 0.15

SUPPORTED_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


# ==============================================================================
# 3. TEKRARLANABİLİRLİK (REPRODUCIBILITY) VE DONANIM AYARLARI
# ==============================================================================
SEED: int = 42


def set_seed(seed: int = SEED) -> None:
    """
    Tüm kütüphanelerde (Python, NumPy, PyTorch CPU/GPU) rastgelelik tohumunu sabitler.
    Akademik deneylerin birebir tekrarlanabilir olmasını sağlar.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """
    CUDA destekli GPU varsa GPU'yu, yoksa otomatik olarak CPU'yu seçer.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ==============================================================================
# 4. GÖRÜNTÜ ÖN İŞLEME (PREPROCESSING) HİPERPARAMETRELERİ
# ==============================================================================
IMAGE_SIZE: tuple[int, int] = (224, 224)  # (Genişlik, Yükseklik)

# CLAHE (Contrast Limited Adaptive Histogram Equalization) parametreleri
CLAHE_CLIP_LIMIT: float = 2.0
CLAHE_TILE_GRID_SIZE: tuple[int, int] = (8, 8)

# Gaussian gürültü azaltma filtresi parametreleri
GAUSSIAN_KERNEL_SIZE: tuple[int, int] = (3, 3)
GAUSSIAN_SIGMA: float = 0.5

# ImageNet ön eğitimli modeller için kanal bazlı ortalama ve standart sapma
IMAGENET_MEAN: list[float] = [0.485, 0.456, 0.406]
IMAGENET_STD: list[float] = [0.229, 0.224, 0.225]


# ==============================================================================
# 5. DERİN ÖĞRENME (MODEL 1) EĞİTİM HİPERPARAMETRELERİ
# ==============================================================================
DL_MODELS_TO_TRAIN: list[str] = ["resnet50", "densenet121"]

BATCH_SIZE: int = 32
NUM_WORKERS: int = 2 if os.name != "nt" else 0  # Windows uyumluluğu için güvenli varsayılan
EPOCHS: int = 30
LEARNING_RATE: float = 1e-4
WEIGHT_DECAY: float = 1e-4
DROPOUT_RATE: float = 0.4
FREEZE_BACKBONE_EPOCHS: int = 0  # 0 ise tüm ağ baştan sona (end-to-end) eğitilir

# EarlyStopping ve ReduceLROnPlateau ayarları
EARLY_STOPPING_PATIENCE: int = 7
LR_SCHEDULER_FACTOR: float = 0.5
LR_SCHEDULER_PATIENCE: int = 3
LR_MIN: float = 1e-7
USE_AMP: bool = True  # GPU mevcutsa Automatic Mixed Precision (FP16) aktif edilir


# ==============================================================================
# 6. HİBRİT ML (MODEL 2: CNN FEATURE + PCA / SELECTKBEST + SVM / XGBOOST)
# ==============================================================================
# PCA açıklanan varyans oranı (örn. %95 varyansı koruyan bileşen sayısı)
PCA_VARIANCE_RATIO: float = 0.95

# SelectKBest (ANOVA F-Score / Fisher benzeri ayrıştırıcılık skoru) için aday özellik sayıları
SELECT_KBEST_CANDIDATES: list[int] = [64, 128, 256]

# GridSearchCV çapraz doğrulama kat sayısı
CV_FOLDS: int = 3

# SVM (RBF çekirdek) hiperparametre arama uzayı
SVM_PARAM_GRID: dict = {
    "C": [0.1, 1.0, 10.0, 50.0],
    "gamma": ["scale", "auto", 0.01, 0.001],
    "kernel": ["rbf"],
}

# XGBoost hiperparametre arama uzayı
XGB_PARAM_GRID: dict = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.03, 0.1],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
}


# ==============================================================================
# 7. AKADEMİK RAPORLAMA GRAFİK AYARLARI
# ==============================================================================
FIGURE_DPI: int = 300
