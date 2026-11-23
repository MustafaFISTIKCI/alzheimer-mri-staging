"""
Biyomedikal MRI Görüntü Ön İşleme Modülü (src/preprocessing.py)

Bu modül, beyin MRI kesitleri üzerinde aşağıdaki standart ön işleme hattını uygular:
1. Görüntünün yüklenmesi ve gri tonlamaya (Grayscale) dönüştürülmesi
2. Gaussian Bulanıklaştırma (Gaussian Denoising) ile yüksek frekanslı tarayıcı gürültüsünün azaltılması
3. CLAHE (Contrast Limited Adaptive Histogram Equalization) ile beyin dokusu (gri/ak madde,
   ventrikül ve hipokampus sınırları) kontrastının yerel olarak iyileştirilmesi
4. 224x224 boyutuna yeniden boyutlandırma (INTER_AREA / INTER_CUBIC)
5. 3 kanallı RGB tensöre dönüştürme ve ImageNet istatistikleri ile normalizasyon

Ayrıca akademik makale raporunun Yöntem / Deneysel Sonuçlar bölümünde kullanılmak üzere
"Orijinal MRI vs CLAHE + Gaussian Uygulanmış MRI" yan yana karşılaştırma görseli üretir.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms

import config


class MRICLAHEPreprocessor:
    """
    OpenCV tabanlı CLAHE + Gaussian Gürültü Azaltma + Yeniden Boyutlandırma ön işleyicisi.
    PyTorch Dataset ve görselleştirme fonksiyonları tarafından doğrudan kullanılır.
    """

    def __init__(
        self,
        image_size: tuple[int, int] = config.IMAGE_SIZE,
        clip_limit: float = config.CLAHE_CLIP_LIMIT,
        tile_grid_size: tuple[int, int] = config.CLAHE_TILE_GRID_SIZE,
        gaussian_kernel: tuple[int, int] = config.GAUSSIAN_KERNEL_SIZE,
        gaussian_sigma: float = config.GAUSSIAN_SIGMA,
    ) -> None:
        self.image_size = image_size
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.gaussian_kernel = gaussian_kernel
        self.gaussian_sigma = gaussian_sigma

    def _create_clahe(self) -> cv2.CLAHE:
        """Çoklu iş parçacığı güvenliği için CLAHE nesnesini gerektiğinde oluşturur."""
        return cv2.createCLAHE(
            clipLimit=self.clip_limit,
            tileGridSize=self.tile_grid_size,
        )

    @staticmethod
    def load_image_unicode_safe(image_path: str | Path) -> np.ndarray:
        """
        Windows/Unicode karakter içeren dosya yollarını da sorunsuz okuyabilen
        OpenCV görüntü yükleme yardımcısı.
        """
        path_str = str(image_path)
        file_bytes = np.fromfile(path_str, dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Görüntü okunamadı veya bozuk: {path_str}")
        return image

    @staticmethod
    def to_grayscale(image: np.ndarray) -> np.ndarray:
        """Görüntüyü tek kanallı uint8 gri tonlamaya dönüştürür."""
        if image.ndim == 2:
            gray = image
        elif image.ndim == 3 and image.shape[2] == 4:
            gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        elif image.ndim == 3 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = np.squeeze(image)

        if gray.dtype != np.uint8:
            gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return gray

    def apply_gaussian_and_clahe(self, gray_image: np.ndarray) -> np.ndarray:
        """
        Gri tonlamalı MRI kesitine sırasıyla Gaussian gürültü azaltma,
        CLAHE kontrast iyileştirme ve 224x224 yeniden boyutlandırma uygular.
        """
        # 1. Gaussian Gürültü Azaltma (CLAHE öncesi gürültü artışını engeller)
        denoised = cv2.GaussianBlur(
            gray_image,
            self.gaussian_kernel,
            sigmaX=self.gaussian_sigma,
        )

        # 2. CLAHE Kontrast İyileştirme
        clahe = self._create_clahe()
        enhanced = clahe.apply(denoised)

        # 3. Hedef Ağ Giriş Boyutuna (224x224) Yeniden Boyutlandırma
        h, w = enhanced.shape[:2]
        interp = cv2.INTER_AREA if (h > self.image_size[1] or w > self.image_size[0]) else cv2.INTER_CUBIC
        resized = cv2.resize(enhanced, self.image_size, interpolation=interp)
        return resized

    def preprocess_to_rgb(self, image_path: str | Path) -> np.ndarray:
        """
        Dosya yolundan MRI görüntüsünü okur, ön işleme hattından geçirir ve
        ImageNet tabanlı CNN modelleri için 3 kanallı RGB (224, 224, 3) uint8 dizi döndürür.
        """
        raw = self.load_image_unicode_safe(image_path)
        gray = self.to_grayscale(raw)
        processed_gray = self.apply_gaussian_and_clahe(gray)
        rgb = cv2.cvtColor(processed_gray, cv2.COLOR_GRAY2RGB)
        return rgb


def get_torch_transforms(is_train: bool = False) -> transforms.Compose:
    """
    Ön işlenmiş RGB MRI görüntüsü (224x224 uint8 NumPy dizisi) için PyTorch dönüşümleri.
    Eğitim aşamasında aşırı öğrenmeyi (overfitting) azaltmak için hafif tıbbi uyumlu
    veri artırma (küçük döndürme/yatay çevirme) uygular; Doğrulama ve Testte sadece normalize eder.
    """
    if is_train:
        return transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=7),
                transforms.ToTensor(),
                transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
        ]
    )


def preprocess_single_image(
    image_path: str | Path,
    preprocessor: Optional[MRICLAHEPreprocessor] = None,
) -> torch.Tensor:
    """
    Tek bir MRI görüntüsünü uçtan uca ön işleyip (1, 3, 224, 224) normalize edilmiş tensöre çevirir.
    """
    if preprocessor is None:
        preprocessor = MRICLAHEPreprocessor()
    rgb = preprocessor.preprocess_to_rgb(image_path)
    tensor = get_torch_transforms(is_train=False)(rgb)
    return tensor


def generate_preprocessing_comparison(
    image_path: str | Path,
    output_path: Path | str = config.OUTPUTS_DIR / "preprocessing_comparison.png",
    preprocessor: Optional[MRICLAHEPreprocessor] = None,
    sample_label: Optional[str] = None,
) -> Path:
    """
    Akademik makale raporu için "Orijinal MRI vs CLAHE + Gaussian Uygulanmış MRI"
    yan yana karşılaştırma görselini (300 DPI PNG) üretir ve kaydeder.

    Args:
        image_path: Örnek MRI görüntü dosyası yolu.
        output_path: Kaydedilecek PNG dosya yolu.
        preprocessor: MRICLAHEPreprocessor örneği.
        sample_label: Görsel başlığında gösterilecek opsiyonel sınıf adı.

    Returns:
        Kaydedilen görselin Path nesnesi.
    """
    if preprocessor is None:
        preprocessor = MRICLAHEPreprocessor()

    raw = preprocessor.load_image_unicode_safe(image_path)
    raw_gray = preprocessor.to_grayscale(raw)
    raw_resized = cv2.resize(raw_gray, preprocessor.image_size, interpolation=cv2.INTER_CUBIC)
    processed = preprocessor.apply_gaussian_and_clahe(raw_gray)

    # Piksel fark haritası (iyileştirilen doku bölgelerini vurgulamak için)
    diff_map = cv2.absdiff(processed, raw_resized)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), dpi=config.FIGURE_DPI)

    axes[0].imshow(raw_resized, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("1. Orijinal MRI (224×224)", fontsize=11, fontweight="bold", pad=10)
    axes[0].axis("off")

    axes[1].imshow(processed, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title(
        f"2. Gaussian + CLAHE MRI\n(clipLimit={preprocessor.clip_limit}, grid={preprocessor.tile_grid_size})",
        fontsize=11,
        fontweight="bold",
        pad=10,
    )
    axes[1].axis("off")

    im = axes[2].imshow(diff_map, cmap="inferno")
    axes[2].set_title("3. Mutlak Kontrast Farkı |Δ|", fontsize=11, fontweight="bold", pad=10)
    axes[2].axis("off")
    cbar = fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8)

    title_suffix = f" — Sınıf: {sample_label}" if sample_label else ""
    fig.suptitle(
        f"Beyin MRI Ön İşleme Etkisi: Orijinal vs. Gaussian Gürültü Azaltma + CLAHE{title_suffix}",
        fontsize=13,
        fontweight="bold",
        y=0.99,
    )

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(out_path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)

    print(f"[GÖRSEL KAYDEDİLDİ] Ön işleme karşılaştırması: {out_path}")
    return out_path
