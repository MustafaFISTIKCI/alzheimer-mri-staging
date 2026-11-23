# Alzheimer MRI Evreleme (4 Sınıf: DL + Hibrit ML)

Beyin MR görüntülerinden Alzheimer hastalığının 4 klinik evresini (`NonDemented`, `VeryMildDemented`, `MildDemented`, `ModerateDemented`) sınıflandıran uçtan uca **PyTorch Transfer Öğrenme (ResNet50, DenseNet121)** ve **Öznitelik Seçimi Tabanlı Hibrit Makine Öğrenmesi (PCA / Fisher Skoru + SVM / XGBoost)** projesi.

## Proje Yapısı

- `config.py` — Merkezi yollar, `SEED=42` sabitleme, CUDA/CPU seçimi ve hiperparametreler
- `data/split_dataset.py` — Alt klasör taraması, `kagglehub` indirme ve Stratified %70/%15/%15 bölünme
- `src/preprocessing.py` — Gaussian gürültü azaltma + CLAHE kontrast iyileştirme + `224×224` normalizasyon
- `src/dataset.py` — PyTorch `Dataset`, deterministik `DataLoader` ve sınıf ağırlıklandırma
- `src/models.py` — `ResNet50` ve `DenseNet121` transfer öğrenme modelleri + havuzlama öznitelik çıkarımı
- `src/train_dl.py` — AMP (FP16), `EarlyStopping`, `ReduceLROnPlateau` ile uçtan uca eğitim
- `src/hybrid_ml.py` — `StandardScaler`, PCA (%95 varyans), Fisher Skoru (`SelectKBest`), `GridSearchCV` (SVM & XGBoost)
- `src/evaluate.py` — Test değerlendirmesi, 300 DPI PNG grafikler ve `model_comparison_table.csv`
- `main.py` — Tüm hattı tek komutla veya `--steps` ile çalıştıran ana betik
- `generate_report.py` — Akademik makale formatında Word raporu (`outputs/Alzheimer_MRI_Evreleme_Makale_Raporu.docx`)
- `generate_presentation_guide.py` — Sunum asistanı ve jüri savunma rehberi (`outputs/Alzheimer_MRI_Sunum_Asistani.docx`)

## Kurulum ve Çalıştırma

```powershell
pip install -r requirements.txt
python main.py --download-kaggle --force-resplit
```

## Deneysel Sonuçlar (Test Seti — 6.600 MRI)

| Model | Accuracy (%) | Macro F1 (%) | Macro AUC | Çıkarım Süresi (sn) |
| :--- | :---: | :---: | :---: | :---: |
| ResNet50 | 99.88 | 99.89 | 1.0000 | 15.7759 |
| DenseNet121 | 99.98 | 99.99 | 1.0000 | 16.2696 |
| DenseNet121+PCA+SVM | 99.98 | 99.99 | 1.0000 | 0.5451 |
| DenseNet121+PCA+XGBoost | 99.95 | 99.96 | 1.0000 | 0.0223 |
| **DenseNet121+Fisher+SVM** | **100.00** | **100.00** | **1.0000** | **0.0752** |
| DenseNet121+Fisher+XGBoost | 99.97 | 99.97 | 1.0000 | 0.0083 |
