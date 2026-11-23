# Veri Seti Klasörü (`data/`)

Kaggle'dan indirilen **`alzheimers-multiclass-dataset-equal-and-augmented`** veri setini bu klasör altına çıkarın:

```text
data/
└── alzheimers-multiclass-dataset-equal-and-augmented/
    ├── NonDemented/
    ├── VeryMildDemented/
    ├── MildDemented/
    └── ModerateDemented/
```

> Not: Alt klasörler iç içe (ör. `Combined Dataset/...`) yapıda olsa dahi `split_dataset.py` tüm alt dizinleri otomatik olarak tarayıp sınıf etiketlerini eşleştirir ve Stratified %70 / %15 / %15 bölünmesini `data/splits/` klasörüne kaydeder.
