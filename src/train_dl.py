"""
Derin Öğrenme (Model 1) Eğitim Modülü (src/train_dl.py)

End-to-End DenseNet121 ve ResNet50 modellerini eğitir:
- Automatic Mixed Precision (AMP) desteği (CUDA üzerinde FP16 hızlandırma)
- EarlyStopping (val_loss iyileşmediğinde eğitimi durdurma ve en iyi ağırlığı saklama)
- ReduceLROnPlateau öğrenme oranı zamanlayıcısı
- Sınıf dengesizliğine karşı sınıf bazlı ağırlıklandırılmış CrossEntropyLoss
- Her epoch sonunda Train/Val Loss, Accuracy ve Learning Rate kaydı
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from src.dataset import compute_balanced_class_weights, create_dataloaders
from src.models import AlzheimerTransferCNN, build_model, get_display_name


class EarlyStopping:
    """
    Doğrulama kaybını (val_loss) izleyerek aşırı öğrenmeyi (overfitting) önleyen
    ve en iyi model ağırlıklarını bellekte/diskte tutan yardımcı sınıf.
    """

    def __init__(self, patience: int = config.EARLY_STOPPING_PATIENCE, min_delta: float = 1e-4) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.early_stop = False
        self.best_state_dict: Optional[dict] = None
        self.best_epoch = 0

    def step(self, val_loss: float, model: nn.Module, epoch: int) -> bool:
        """
        Her epoch sonunda çağrılır.
        İyileşme varsa True döndürür ve ağırlıkları günceller.
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state_dict = copy.deepcopy(model.state_dict())
            self.best_epoch = epoch
            return True

        self.counter += 1
        if self.counter >= self.patience:
            self.early_stop = True
        return False


def _create_grad_scaler(device: torch.device, amp_enabled: bool):
    """PyTorch 2.x sürümleri arasında uyumlu GradScaler oluşturur."""
    if not (amp_enabled and device.type == "cuda"):
        return None
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=True)
    return torch.cuda.amp.GradScaler(enabled=True)


def _autocast_context(device: torch.device, amp_enabled: bool):
    """PyTorch 2.x sürümleri arasında uyumlu AMP autocast bağlamı döndürür."""
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type=device.type, enabled=amp_enabled)
    return torch.cuda.amp.autocast(enabled=amp_enabled)


def _run_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler=None,
    amp_enabled: bool = False,
    desc: str = "Train",
) -> tuple[float, float]:
    """Tek bir eğitim veya doğrulama epoch'unu çalıştırır ve (ortalama_loss, accuracy) döndürür."""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for images, labels in tqdm(loader, desc=desc, leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            with _autocast_context(device=device, amp_enabled=amp_enabled):
                logits = model(images)
                loss = criterion(logits, labels)

            if is_train:
                if scaler is not None and amp_enabled:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += batch_size

    epoch_loss = running_loss / max(total, 1)
    epoch_acc = correct / max(total, 1)
    return epoch_loss, epoch_acc


def train_single_dl_model(
    model_name: str,
    loaders: dict[str, DataLoader],
    class_weights: torch.Tensor,
    epochs: int = config.EPOCHS,
    learning_rate: float = config.LEARNING_RATE,
    weight_decay: float = config.WEIGHT_DECAY,
    patience: int = config.EARLY_STOPPING_PATIENCE,
    pretrained: bool = True,
    device: Optional[torch.device] = None,
) -> dict:
    """
    Belirtilen CNN modelini (resnet50 veya densenet121) eğitir, en iyi checkpoint'i
    ve epoch geçmişini diske kaydeder.
    """
    config.set_seed(config.SEED)
    device = device or config.get_device()
    amp_enabled = bool(config.USE_AMP and device.type == "cuda")
    display_name = get_display_name(model_name)

    print(f"\n{'=' * 70}")
    print(f"[MODEL 1 EĞİTİMİ] {display_name} | Cihaz: {device} | AMP: {amp_enabled}")
    print(f"{'=' * 70}")

    model = build_model(
        model_name=model_name,
        num_classes=config.NUM_CLASSES,
        dropout_rate=config.DROPOUT_RATE,
        pretrained=pretrained,
        device=device,
    )

    if config.FREEZE_BACKBONE_EPOCHS > 0:
        model.set_backbone_trainable(False)

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.LR_SCHEDULER_FACTOR,
        patience=config.LR_SCHEDULER_PATIENCE,
        min_lr=config.LR_MIN,
    )
    scaler = _create_grad_scaler(device=device, amp_enabled=amp_enabled)
    early_stopper = EarlyStopping(patience=patience)

    history: list[dict] = []
    best_val_acc = 0.0
    start_time = time.perf_counter()

    for epoch in range(1, epochs + 1):
        # İstenirse belirli epoch sonrası omurga ağırlıklarını çöz (fine-tuning)
        if config.FREEZE_BACKBONE_EPOCHS > 0 and epoch == config.FREEZE_BACKBONE_EPOCHS + 1:
            model.set_backbone_trainable(True)
            print(f"[BİLGİ] Epoch {epoch}: Omurga katmanları fine-tuning için açıldı.")

        train_loss, train_acc = _run_one_epoch(
            model=model,
            loader=loaders["train"],
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            amp_enabled=amp_enabled,
            desc=f"{display_name} Ep {epoch}/{epochs} [Train]",
        )
        val_loss, val_acc = _run_one_epoch(
            model=model,
            loader=loaders["val"],
            criterion=criterion,
            device=device,
            optimizer=None,
            amp_enabled=amp_enabled,
            desc=f"{display_name} Ep {epoch}/{epochs} [Val]",
        )

        scheduler.step(val_loss)
        current_lr = float(optimizer.param_groups[0]["lr"])

        history.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "train_acc": round(train_acc, 6),
                "val_loss": round(val_loss, 6),
                "val_acc": round(val_acc, 6),
                "lr": current_lr,
            }
        )

        improved = early_stopper.step(val_loss, model, epoch)
        if improved:
            best_val_acc = val_acc
        status_tag = " [EN İYİ]" if improved else ""

        print(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc*100:.2f}% | "
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc*100:.2f}% | "
            f"LR: {current_lr:.2e}{status_tag}"
        )

        if early_stopper.early_stop:
            print(f"[EARLY STOPPING] {display_name} için Epoch {epoch}'da eğitim durduruldu.")
            break

    train_time_sec = time.perf_counter() - start_time

    # En iyi ağırlıkları geri yükle
    if early_stopper.best_state_dict is not None:
        model.load_state_dict(early_stopper.best_state_dict)

    # Checkpoint ve geçmiş dosyalarını kaydet
    ckpt_path = config.CHECKPOINTS_DIR / f"{model_name}_best.pt"
    history_path = config.CHECKPOINTS_DIR / f"{model_name}_history.csv"

    torch.save(
        {
            "model_name": model_name,
            "display_name": display_name,
            "model_state_dict": model.state_dict(),
            "best_val_loss": float(early_stopper.best_loss),
            "best_val_acc": float(best_val_acc),
            "best_epoch": int(early_stopper.best_epoch),
            "train_time_sec": float(train_time_sec),
            "class_names": config.CLASS_NAMES,
        },
        ckpt_path,
    )
    pd.DataFrame(history).to_csv(history_path, index=False)

    print(
        f"[TAMAMLANDI] {display_name} -> En iyi Epoch: {early_stopper.best_epoch} | "
        f"Best Val Loss: {early_stopper.best_loss:.4f} | "
        f"Best Val Acc: {best_val_acc*100:.2f}% | Süre: {train_time_sec:.2f} sn"
    )

    return {
        "model_name": model_name,
        "display_name": display_name,
        "checkpoint_path": str(ckpt_path),
        "history_path": str(history_path),
        "best_val_loss": float(early_stopper.best_loss),
        "best_val_acc": float(best_val_acc),
        "best_epoch": int(early_stopper.best_epoch),
        "train_time_sec": float(train_time_sec),
    }


def train_all_dl_models(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    models_to_train: list[str] = config.DL_MODELS_TO_TRAIN,
    epochs: int = config.EPOCHS,
    batch_size: int = config.BATCH_SIZE,
    learning_rate: float = config.LEARNING_RATE,
    pretrained: bool = True,
) -> dict:
    """
    Yapılandırmada tanımlı tüm derin öğrenme modellerini (ResNet50, DenseNet121) eğitir,
    en iyi CNN modelini seçer ve özet bilgiyi JSON olarak kaydeder.
    """
    config.set_seed(config.SEED)
    device = config.get_device()

    loaders = create_dataloaders(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        batch_size=batch_size,
    )
    class_weights = compute_balanced_class_weights(train_df)
    print(f"[BİLGİ] Sınıf ağırlıkları: {np.round(class_weights.numpy(), 4).tolist()}")

    results: dict[str, dict] = {}
    for model_name in models_to_train:
        res = train_single_dl_model(
            model_name=model_name,
            loaders=loaders,
            class_weights=class_weights,
            epochs=epochs,
            learning_rate=learning_rate,
            pretrained=pretrained,
            device=device,
        )
        results[model_name] = res

    # En iyi CNN modelini Doğrulama Doğruluğu (yüksek) ve Kayıp (düşük) kriterine göre seç
    best_model_name = max(
        results.keys(),
        key=lambda m: (results[m]["best_val_acc"], -results[m]["best_val_loss"]),
    )

    summary = {
        "best_cnn_model": best_model_name,
        "best_cnn_display_name": results[best_model_name]["display_name"],
        "models": results,
    }

    summary_path = config.OUTPUTS_DIR / "dl_training_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(
        f"\n[EN İYİ CNN SEÇİMİ] {summary['best_cnn_display_name']} "
        f"(Val Acc: {results[best_model_name]['best_val_acc']*100:.2f}%)"
    )
    return summary


@torch.no_grad()
def predict_dl_model(
    checkpoint_path: Path | str,
    test_loader: DataLoader,
    device: Optional[torch.device] = None,
) -> dict:
    """
    Kaydedilmiş bir DL checkpoint'ini yükler, Test seti üzerinde tahmin ve olasılıkları
    üretir ve çıkarım (inference) süresini ölçer.
    """
    device = device or config.get_device()
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_name = ckpt["model_name"]

    model: AlzheimerTransferCNN = build_model(
        model_name=model_name,
        num_classes=config.NUM_CLASSES,
        dropout_rate=config.DROPOUT_RATE,
        pretrained=False,
        device=device,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    all_probs: list[np.ndarray] = []
    all_preds: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    if device.type == "cuda":
        torch.cuda.synchronize()
    start_time = time.perf_counter()

    for images, labels in test_loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        all_probs.append(probs.cpu().numpy())
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.numpy())

    if device.type == "cuda":
        torch.cuda.synchronize()
    inference_time_sec = time.perf_counter() - start_time

    return {
        "model_name": model_name,
        "display_name": ckpt.get("display_name", get_display_name(model_name)),
        "y_true": np.concatenate(all_labels, axis=0),
        "y_pred": np.concatenate(all_preds, axis=0),
        "y_prob": np.concatenate(all_probs, axis=0),
        "train_time_sec": float(ckpt.get("train_time_sec", 0.0)),
        "inference_time_sec": float(inference_time_sec),
    }
