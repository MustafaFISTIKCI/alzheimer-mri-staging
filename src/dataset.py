"""
PyTorch Veri Seti ve DataLoader Modülü (src/dataset.py)

Stratified olarak bölünmüş CSV dosyalarını (train.csv, val.csv, test.csv) okur,
MRICLAHEPreprocessor ile görüntüleri ön işler ve deterministic PyTorch
DataLoader nesnelerini oluşturur. Ayrıca sınıf dengesizliğine karşı sınıf ağırlıklarını hesaplar.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset

import config
from src.preprocessing import MRICLAHEPreprocessor, get_torch_transforms


class AlzheimerMRIDataset(Dataset):
    """
    Alzheimer MRI evreleme veri seti için özel PyTorch Dataset sınıfı.
    Her örnekte OpenCV CLAHE + Gaussian ön işleme uygular.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        is_train: bool = False,
        preprocessor: Optional[MRICLAHEPreprocessor] = None,
    ) -> None:
        self.df = dataframe.reset_index(drop=True)
        self.filepaths = self.df["filepath"].tolist()
        self.labels = self.df["label"].astype(int).tolist()
        self.is_train = is_train
        self.preprocessor = preprocessor or MRICLAHEPreprocessor()
        self.transform = get_torch_transforms(is_train=is_train)

    def __len__(self) -> int:
        return len(self.filepaths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img_path = self.filepaths[idx]
        label = self.labels[idx]

        # OpenCV ile Gaussian + CLAHE + 224x224 + RGB dönüşümü
        rgb_image = self.preprocessor.preprocess_to_rgb(img_path)
        tensor_image = self.transform(rgb_image)
        return tensor_image, label


def compute_balanced_class_weights(
    train_df: pd.DataFrame,
    num_classes: int = config.NUM_CLASSES,
) -> torch.Tensor:
    """
    Eğitim setindeki sınıf frekanslarına göre ters orantılı (balanced) sınıf
    ağırlıklarını hesaplar. Veri setinde sınıf dengesizliği olsa dahi azınlık
    sınıflarının (örn. ModerateDemented) kayıp fonksiyonundaki etkisini dengeler.
    """
    y_train = train_df["label"].astype(int).to_numpy()
    present_classes = np.unique(y_train)
    weights = np.ones(num_classes, dtype=np.float32)

    computed = compute_class_weight(
        class_weight="balanced",
        classes=present_classes,
        y=y_train,
    )
    for cls_idx, w in zip(present_classes, computed):
        weights[int(cls_idx)] = float(w)

    return torch.tensor(weights, dtype=torch.float32)


def _seed_worker(worker_id: int) -> None:
    """DataLoader alt iş parçacıklarında tekrarlanabilirliği garanti altına alır."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)


def create_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = config.NUM_WORKERS,
    seed: int = config.SEED,
) -> dict[str, DataLoader]:
    """
    Train, Validation ve Test kümeleri için PyTorch DataLoader sözlüğü döndürür.
    """
    preprocessor = MRICLAHEPreprocessor()
    train_ds = AlzheimerMRIDataset(train_df, is_train=True, preprocessor=preprocessor)
    val_ds = AlzheimerMRIDataset(val_df, is_train=False, preprocessor=preprocessor)
    test_ds = AlzheimerMRIDataset(test_df, is_train=False, preprocessor=preprocessor)

    generator = torch.Generator()
    generator.manual_seed(seed)

    pin_memory = torch.cuda.is_available()

    loaders = {
        "train": DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            worker_init_fn=_seed_worker,
            generator=generator,
        ),
        "val": DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "test": DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
    }
    return loaders
