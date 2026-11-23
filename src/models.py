"""
Derin Öğrenme Transfer Learning Modelleri (src/models.py)

PyTorch tabanlı DenseNet121 ve ResNet50 mimarilerini Alzheimer 4-sınıflı
evreleme görevine uyarlar. Her iki model de:
- `extract_features(x)`: Son global havuzlama (Global Average Pooling) katmanından
  yüksek seviyeli öznitelik vektörünü döndürür (Hibrit ML pipeline'ı için).
- `forward(x)`: Öznitelik vektörünü sınıflandırıcı başlıktan (Dropout + Linear)
  geçirerek sınıf logitlerini döndürür.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

import config


SUPPORTED_BACKBONES = ("resnet50", "densenet121")


class AlzheimerTransferCNN(nn.Module):
    """
    ResNet50 ve DenseNet121 omurgalarını (backbone) birleştiren modüler Transfer Learning sınıfı.
    Son havuzlama katmanı çıktısını (feature vector) temiz bir arayüzle dışarı verir.
    """

    def __init__(
        self,
        model_name: str,
        num_classes: int = config.NUM_CLASSES,
        dropout_rate: float = config.DROPOUT_RATE,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.model_name = model_name.lower().strip()
        if self.model_name not in SUPPORTED_BACKBONES:
            raise ValueError(
                f"Desteklenmeyen model: '{model_name}'. Desteklenenler: {SUPPORTED_BACKBONES}"
            )

        if self.model_name == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            backbone = models.resnet50(weights=weights)
            self.feature_dim = backbone.fc.in_features  # 2048
            # Son FC katmanı hariç tüm katmanları koru (conv1 -> ... -> avgpool)
            self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])

        elif self.model_name == "densenet121":
            weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
            backbone = models.densenet121(weights=weights)
            self.feature_dim = backbone.classifier.in_features  # 1024
            self.feature_extractor = backbone.features

        # Yeni 4 sınıflı Alzheimer sınıflandırma başlığı
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate * 0.5),
            nn.Linear(256, num_classes),
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Son havuzlama katmanından (Global Average Pooling) düzleştirilmiş
        öznitelik vektörünü (B, feature_dim) çıkarır.
        """
        if self.model_name == "resnet50":
            feats = self.feature_extractor(x)  # (B, 2048, 1, 1)
            return torch.flatten(feats, 1)

        # DenseNet121: özellik haritalarından sonra ReLU + AdaptiveAvgPool2d(1,1) uygulanır
        feats = self.feature_extractor(x)
        out = F.relu(feats, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        return torch.flatten(out, 1)  # (B, 1024)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """İleri besleme: öznitelik çıkarımı ve sınıf logitlerinin üretilmesi."""
        pooled_features = self.extract_features(x)
        logits = self.classifier(pooled_features)
        return logits

    def set_backbone_trainable(self, trainable: bool) -> None:
        """Omurga (backbone) katmanlarının ağırlık güncellemesini açar veya dondurur."""
        for param in self.feature_extractor.parameters():
            param.requires_grad = trainable


def build_model(
    model_name: str,
    num_classes: int = config.NUM_CLASSES,
    dropout_rate: float = config.DROPOUT_RATE,
    pretrained: bool = True,
    device: torch.device | None = None,
) -> AlzheimerTransferCNN:
    """
    İstenen Transfer Learning modelini oluşturur ve hedef cihaza (CUDA/CPU) taşır.
    """
    model = AlzheimerTransferCNN(
        model_name=model_name,
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        pretrained=pretrained,
    )
    if device is not None:
        model = model.to(device)
    return model


def get_display_name(model_name: str) -> str:
    """Raporlama ve tablolar için standart büyük/küçük harfli model adını döndürür."""
    mapping = {
        "resnet50": "ResNet50",
        "densenet121": "DenseNet121",
    }
    return mapping.get(model_name.lower(), model_name)
