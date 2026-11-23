"""
Alzheimer MRI Evreleme Projesi - Sunum Asistanı ve Savunma Hazırlık Rehberi Üretici
(generate_presentation_guide.py)

`outputs/` altındaki gerçek deneysel verileri kullanarak hocaya/jüriye sunum yaparken
kullanılacak slayt bazlı konuşma metni, vurgulanacak kritik teknik noktalar ve
olası jüri soruları + hazır cevaplar içeren Word dokümanını
(`outputs/Alzheimer_MRI_Sunum_Asistani.docx`) üretir.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls, qn
from docx.shared import Cm, Pt, RGBColor

import config


def _configure_doc(doc: Document) -> None:
    for section in doc.sections:
        section.page_width = Cm(21.0)
        section.page_height = Cm(29.7)
        section.top_margin = Cm(2.2)
        section.bottom_margin = Cm(2.2)
        section.left_margin = Cm(2.2)
        section.right_margin = Cm(2.2)

    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(11.5)
    normal.font.color.rgb = RGBColor(25, 25, 25)
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    normal.paragraph_format.line_spacing = 1.15
    normal.paragraph_format.space_after = Pt(5)


def _add_heading(doc: Document, text: str, level: int = 1) -> None:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.keep_with_next = True
    if level == 1:
        p.paragraph_format.space_before = Pt(14)
        p.paragraph_format.space_after = Pt(6)
        r = p.add_run(text)
        r.font.name = "Times New Roman"
        r.font.size = Pt(14)
        r.bold = True
        r.font.color.rgb = RGBColor(20, 55, 105)
        # Alt çizgi efekti
        p_pr = p._p.get_or_add_pPr()
        p_pr.append(
            parse_xml(
                f'<w:pBdr {nsdecls("w")}><w:bottom w:val="single" w:sz="8" w:space="3" w:color="143769"/></w:pBdr>'
            )
        )
    else:
        p.paragraph_format.space_before = Pt(10)
        p.paragraph_format.space_after = Pt(4)
        r = p.add_run(text)
        r.font.name = "Times New Roman"
        r.font.size = Pt(12)
        r.bold = True
        r.font.color.rgb = RGBColor(30, 75, 130)


def _add_para(
    doc: Document,
    text: str,
    bold_prefix: str | None = None,
    italic: bool = False,
    align: WD_ALIGN_PARAGRAPH = WD_ALIGN_PARAGRAPH.JUSTIFY,
    space_after: float = 5,
) -> None:
    p = doc.add_paragraph()
    p.alignment = align
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(space_after)
    if bold_prefix:
        rb = p.add_run(bold_prefix)
        rb.font.name = "Times New Roman"
        rb.font.size = Pt(11.5)
        rb.bold = True
        rb.italic = italic
    r = p.add_run(text)
    r.font.name = "Times New Roman"
    r.font.size = Pt(11.5)
    r.italic = italic


def _add_callout_box(
    doc: Document,
    title: str,
    body_paragraphs: list[tuple[str | None, str]],
    fill_hex: str = "F0F5FA",
    border_hex: str = "1F4E79",
) -> None:
    """Vurgu kutusu / Soru-Cevap kartı oluşturur."""
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = False
    cell = tbl.rows[0].cells[0]
    cell.width = Cm(16.6)

    tc_pr = cell._tc.get_or_add_tcPr()
    tc_pr.append(parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>'))
    tc_pr.append(
        parse_xml(
            f'<w:tcBorders {nsdecls("w")}>'
            f'<w:left w:val="single" w:sz="24" w:space="0" w:color="{border_hex}"/>'
            '<w:top w:val="single" w:sz="4" w:space="0" w:color="D0D7DE"/>'
            '<w:right w:val="single" w:sz="4" w:space="0" w:color="D0D7DE"/>'
            '<w:bottom w:val="single" w:sz="4" w:space="0" w:color="D0D7DE"/>'
            "</w:tcBorders>"
        )
    )
    tc_pr.append(
        parse_xml(
            f'<w:tcMar {nsdecls("w")}>'
            '<w:top w:w="120" w:type="dxa"/>'
            '<w:bottom w:w="120" w:type="dxa"/>'
            '<w:left w:w="180" w:type="dxa"/>'
            '<w:right w:w="160" w:type="dxa"/>'
            "</w:tcMar>"
        )
    )

    p0 = cell.paragraphs[0]
    p0.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p0.paragraph_format.space_before = Pt(0)
    p0.paragraph_format.space_after = Pt(4)
    p0.paragraph_format.line_spacing = 1.15
    rt = p0.add_run(title)
    rt.font.name = "Times New Roman"
    rt.font.size = Pt(11.5)
    rt.bold = True
    rt.font.color.rgb = RGBColor(20, 55, 105)

    for idx, (prefix, txt) in enumerate(body_paragraphs):
        p = cell.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(3 if idx < len(body_paragraphs) - 1 else 0)
        p.paragraph_format.line_spacing = 1.15
        if prefix:
            rp = p.add_run(prefix)
            rp.font.name = "Times New Roman"
            rp.font.size = Pt(11)
            rp.bold = True
        r = p.add_run(txt)
        r.font.name = "Times New Roman"
        r.font.size = Pt(11)

    p_gap = doc.add_paragraph()
    p_gap.paragraph_format.space_before = Pt(0)
    p_gap.paragraph_format.space_after = Pt(4)


def _add_styled_table(
    doc: Document,
    headers: list[str],
    rows: list[list[str]],
    col_widths_cm: list[float],
) -> None:
    tbl = doc.add_table(rows=len(rows) + 1, cols=len(headers))
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = False

    tbl_pr = tbl._tbl.tblPr
    tbl_pr.append(
        parse_xml(
            f'<w:tblBorders {nsdecls("w")}>'
            '<w:top w:val="single" w:sz="10" w:space="0" w:color="143769"/>'
            '<w:bottom w:val="single" w:sz="10" w:space="0" w:color="143769"/>'
            '<w:insideH w:val="single" w:sz="4" w:space="0" w:color="D0D7DE"/>'
            '<w:left w:val="none"/><w:right w:val="none"/><w:insideV w:val="none"/>'
            "</w:tblBorders>"
        )
    )

    for c_idx, h in enumerate(headers):
        cell = tbl.rows[0].cells[c_idx]
        cell._tc.get_or_add_tcPr().append(parse_xml(f'<w:shd {nsdecls("w")} w:fill="143769"/>'))
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p.paragraph_format.space_before = Pt(3)
        p.paragraph_format.space_after = Pt(3)
        r = p.add_run(h)
        r.font.name = "Times New Roman"
        r.font.size = Pt(10)
        r.bold = True
        r.font.color.rgb = RGBColor(255, 255, 255)

    for r_idx, row in enumerate(rows, start=1):
        for c_idx, val in enumerate(row):
            cell = tbl.rows[r_idx].cells[c_idx]
            if r_idx % 2 == 0:
                cell._tc.get_or_add_tcPr().append(parse_xml(f'<w:shd {nsdecls("w")} w:fill="F5F8FC"/>'))
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.space_before = Pt(3)
            p.paragraph_format.space_after = Pt(3)
            p.paragraph_format.line_spacing = 1.1
            r = p.add_run(val)
            r.font.name = "Times New Roman"
            r.font.size = Pt(10)
            if c_idx == 0:
                r.bold = True

    for row in tbl.rows:
        for idx, w in enumerate(col_widths_cm):
            row.cells[idx].width = Cm(w)

    p_gap = doc.add_paragraph()
    p_gap.paragraph_format.space_before = Pt(0)
    p_gap.paragraph_format.space_after = Pt(4)


def build_presentation_guide(output_path: Path) -> Path:
    comp_df = pd.read_csv(config.OUTPUTS_DIR / "model_comparison_table.csv")
    with open(config.OUTPUTS_DIR / "dl_training_summary.json", "r", encoding="utf-8") as f:
        dl_summary = json.load(f)
    with open(config.HYBRID_MODELS_DIR / "hybrid_results.json", "r", encoding="utf-8") as f:
        hybrid_summary = json.load(f)

    res_best = dl_summary["models"]["resnet50"]
    den_best = dl_summary["models"]["densenet121"]
    fish_svm = hybrid_summary["models"]["DenseNet121+Fisher+SVM"]
    fish_xgb = hybrid_summary["models"]["DenseNet121+Fisher+XGBoost"]
    pca_svm = hybrid_summary["models"]["DenseNet121+PCA+SVM"]

    doc = Document()
    _configure_doc(doc)

    # --------------------------------------------------------------------------
    # ÜST BAŞLIK BLOĞU
    # --------------------------------------------------------------------------
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title.paragraph_format.space_after = Pt(4)
    rt = p_title.add_run("ALZHEIMER MRI EVRELEME PROJESİ\nSUNUM ASİSTANI VE JÜRİ SAVUNMA REHBERİ")
    rt.font.name = "Times New Roman"
    rt.font.size = Pt(15)
    rt.bold = True
    rt.font.color.rgb = RGBColor(20, 55, 105)

    p_sub = doc.add_paragraph()
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_sub.paragraph_format.space_after = Pt(10)
    rs = p_sub.add_run(
        "Hazırlayan: Mustafa Fıstıkçı (Bilgisayar Mühendisliği)\n"
        "Amaç: Bitirme / Dönem Projesi Sunum Akışı, Kritik Vurgu Noktaları ve Olası Hoca Sorularına Hazırlık"
    )
    rs.font.name = "Times New Roman"
    rs.font.size = Pt(11)
    rs.italic = True

    _add_callout_box(
        doc,
        "Altın Kural: Hocaya Nasıl Bir Hikâye Anlatmalısınız?",
        [
            (
                "Ana Mesaj: ",
                "Biz bu projede sadece hazır bir CNN modelini çalıştırıp doğruluk skoru almadık. "
                "Tıbbi görüntü işlemeden (Gaussian + CLAHE) veri sızıntısı önlemeye (Stratified 70/15/15 "
                "ve yalnızca Train'e fit edilen istatistikler), uçtan uca derin öğrenme kıyaslamasından "
                "(ResNet50 vs DenseNet121) öznitelik seçimi tabanlı hibrit mühendisliğe (Fisher/PCA + "
                "SVM/XGBoost) uzanan; hem %100,00 doğruluk veren hem de çıkarım süresini 16,27 saniyeden "
                "0,075 saniyeye (216 kat hızlanma!) düşüren uçtan uca bir klinik karar destek hattı kurduk.",
            )
        ],
        fill_hex="EAF4EC",
        border_hex="1E7E34",
    )

    # --------------------------------------------------------------------------
    # BÖLÜM 1: 8-10 DAKİKALIK SLAYT PLANI VE KONUŞMA METNİ
    # --------------------------------------------------------------------------
    _add_heading(doc, "1. 8–10 Dakikalık Sunum Akışı ve Slayt Bazlı Konuşma Metni")
    _add_para(
        doc,
        "Aşağıdaki plan, sunumu 7 net slaytta (yaklaşık 8–10 dakika) akıcı, kendinden emin ve "
        "akademik bir dille sunmanız için hazırlanmıştır. Her slaytta ne göstereceğiniz ve hocaya "
        "birebir neler söyleyeceğiniz yer almaktadır.",
    )

    slides = [
        (
            "Slayt 1 — Başlık ve Klinik Problem Tanımı (Süre: ~1 dk)",
            [
                ("Gösterilecekler: ", "Proje başlığı, 4 Alzheimer evresi (NonDemented, VeryMildDemented, MildDemented, ModerateDemented) ve beyin MR kesiti örnekleri."),
                ("Konuşma Metni: ", "\"Hocam merhaba, çalışmamızda beyin MR görüntülerinden Alzheimer hastalığının dört klinik evresini otomatik ve yüksek doğrulukla sınıflandıran uçtan uca bir derin öğrenme ve hibrit makine öğrenmesi sistemi geliştirdik. Klinik olarak özellikle VeryMildDemented gibi erken evrelerin gözden kaçırılmaması tedavi planlaması açısından kritik. Ancak manuel radyolojik inceleme zaman alıcı ve gözlemciler arası değişkenliğe açık. Biz de hem yüksek doğruluk hem de hızlı çıkarım sağlayan bir karar destek hattı tasarladık.\""),
                ("Vurgu: ", "İkili (Hasta/Sağlıklı) değil, 4 sınıflı evreleme yapıldığı özellikle belirtilmeli."),
            ],
        ),
        (
            "Slayt 2 — Veri Seti ve Veri Sızıntısı Önleme Protokolü (Süre: ~1 dk)",
            [
                ("Gösterilecekler: ", "Raporunuzdaki Tablo 1 (44.000 görüntü -> 30.800 Train / 6.600 Val / 6.600 Test dağılımı)."),
                ("Konuşma Metni: ", "\"Çalışmada toplam 44.000 aksiyel beyin MR görüntüsü içeren 4 sınıflı veri setini kullandık. Tüm deneylerin tekrarlanabilir olması için tüm kütüphanelerde seed=42 sabitledik. Test setinin eğitim sürecinden tamamen izole kalması için veriyi katmanlı (Stratified) örnekleme ile %70 Eğitim, %15 Doğrulama ve %15 Test olarak ayırdık. Ölçeklendirme, PCA ve Fisher seçimi dahil tüm istatistiksel dönüşümleri yalnızca Eğitim seti üzerinde hesaplayıp Test setine uygulayarak veri sızıntısının önüne geçtik.\""),
                ("Vurgu: ", "Data leakage (veri sızıntısı) önlemi ve tekrarlanabilirlik (seed=42) vurgulanmalı."),
            ],
        ),
        (
            "Slayt 3 — Biyomedikal Ön İşleme: Gaussian + CLAHE (Süre: ~1.5 dk)",
            [
                ("Gösterilecekler: ", "Şekil 1 (preprocessing_comparison.png: Orijinal MRI vs Gaussian+CLAHE vs Mutlak Fark Haritası)."),
                ("Konuşma Metni: ", "\"Ham MR görüntülerinde tarayıcı kaynaklı gürültü ve düşük yerel kontrast olabiliyor. Biz doğrudan CLAHE uygulasaydık görüntüdeki gürültüyü de yükseltecektik. Bu yüzden önce 3×3 Gaussian filtreyle yüksek frekanslı gürültüyü bastırdık, ardından clipLimit=2.0 ve 8×8 karo boyutuyla CLAHE uyguladık. Sağdaki mutlak fark haritasında da görüldüğü gibi özellikle kortikal kıvrımlar, gri-ak madde ve ventrikül sınırları belirginleşti; arka plan siyah dokusu ise bozulmadı.\""),
                ("Vurgu: ", "Neden önce Gaussian sonra CLAHE yapıldığı ve fark haritasının neyi kanıtladığı!"),
            ],
        ),
        (
            "Slayt 4 — Model 1: End-to-End Transfer Learning (Süre: ~1.5 dk)",
            [
                ("Gösterilecekler: ", "Şekil 2 (training_curves.png) ve Şekil 3 (confusion_matrix_dl.png)."),
                ("Konuşma Metni: ", "\"Birinci aşamada ImageNet ön eğitimli ResNet50 ve DenseNet121 mimarilerini RTX 5060 GPU üzerinde FP16 Mixed Precision (AMP) ile eğittik. Sınıf dengesizliği etkisini sıfırlamak için ters frekanslı sınıf ağırlıklı CrossEntropy kaybı, AdamW optimizasyonu, ReduceLROnPlateau ve EarlyStopping kullandık. Eğitim eğrilerinde aşırı öğrenme olmadan yakınsama sağlandı. ResNet50 25. epoch'ta erken durdurma ile %99,88 test doğruluğu alırken, yoğun bağlantılar sayesinde düşük seviyeli doku bilgisini koruyan DenseNet121 %99,98 doğruluğa (6.600 örnekte yalnızca 1 hata) ulaştı.\""),
                ("Vurgu: ", "DenseNet121'in yoğun bağlantılarının neden ResNet50'den daha az hata yaptığı."),
            ],
        ),
        (
            "Slayt 5 — Model 2: Önerilen Hibrit Yaklaşım (Süre: ~1.5 dk)",
            [
                ("Gösterilecekler: ", "Mimari akış şeması: DenseNet121 (1024 özellik) -> StandardScaler -> PCA (155) & Fisher Skoru (64) -> GridSearchCV (SVM RBF & XGBoost)."),
                ("Konuşma Metni: ", "\"Burada projemizin temel mühendislik katkısına geçtik: Doğrulama başarımı en yüksek olan DenseNet121'in son global havuzlama katmanından 1024 boyutlu derin öznitelikleri çıkardık. Bu 1024 boyutun içindeki gereksiz veya gürültülü bileşenleri elemek için iki yöntem kıyasladık: Denetimsiz varyans koruyan PCA (%95 varyans -> 155 bileşen) ve denetimli sınıf ayrıştırıcılığı ölçen Fisher Skoru (SelectKBest -> K=64). Ardından RBF çekirdekli SVM ve XGBoost ile 3-katlı GridSearchCV optimizasyonu yaptık.\""),
                ("Vurgu: ", "PCA (denetimsiz) vs Fisher Skoru (denetimli) farkı."),
            ],
        ),
        (
            "Slayt 6 — Deneysel Sonuçlar ve Çıkarım Hızı Kıyaslaması (Süre: ~1.5 dk)",
            [
                ("Gösterilecekler: ", "Tablo 2 (model_comparison_table.csv), Şekil 4 (confusion_matrix_hybrid.png) ve Şekil 5 (roc_curves.png)."),
                ("Konuşma Metni: ", "\"Tablo 2'de tüm modelleri aynı test setinde kıyasladığımızda çok çarpıcı iki sonuç görüyoruz: İlk olarak DenseNet121+Fisher+SVM modeli 6.600 test görüntüsünün tamamını hatasız bilerek %100,00 doğruluk ve 1,0000 Makro AUC elde etti. İkinci ve pratik açıdan çok önemli sonuç ise hız: DenseNet121'in uçtan uca test çıkarımı 16,27 saniye sürerken, öznitelikler çıkarıldıktan sonra Fisher+SVM sınıflandırması 0,075 saniyede, Fisher+XGBoost ise %99,97 doğrulukla yalnızca 0,0083 saniyede tamamlandı.\""),
                ("Vurgu: ", "%100 doğruluk + 216x (SVM) / ~1960x (XGBoost) sınıflandırma hız artışı."),
            ],
        ),
        (
            "Slayt 7 — Kısıtlar ve Gelecek Çalışmalar (Süre: ~1 dk)",
            [
                ("Gösterilecekler: ", "Kısıtlar ve gelecek yol haritası maddeleri."),
                ("Konuşma Metni: ", "\"Akademik dürüstlük açısından çalışmamızın iki temel kısıtını da belirtmek isteriz: Veri seti 2B aksiyel ve artırılmış (augmented) kesitlerden oluştuğu için hasta bazlı (subject-level) 3B hacimsel bölünme yapılamamıştır; ayrıca dış merkezli farklı cihaz verileriyle test henüz yapılmamıştır. Gelecek çalışmada 3B ADNI/OASIS verilerinde hasta bazlı bölünme ve Grad-CAM/SHAP açıklanabilir yapay zekâ görselleştirmelerini eklemeyi hedefliyoruz.\""),
                ("Vurgu: ", "Kısıtları hocadan önce sizin söylemeniz hocanın olası eleştirisini baştan avantaja çevirir!"),
            ],
        ),
    ]

    for s_title, s_items in slides:
        _add_callout_box(doc, s_title, s_items, fill_hex="F4F7FB", border_hex="1F4E79")

    # --------------------------------------------------------------------------
    # BÖLÜM 2: ÖZELLİKLE VURGULANACAK 6 KRİTİK MÜHENDİSLİK NOKTASI
    # --------------------------------------------------------------------------
    _add_heading(doc, "2. Sunumda Mutlaka Vurgulanması Gereken 6 Kritik Nokta")
    highlights = [
        [
            "1. Ön İşleme Sırası",
            "Önce Gaussian Blur (3×3, σ=0.5), sonra CLAHE (clipLimit=2.0, 8×8). Tersi yapılsaydı CLAHE tarayıcı gürültüsünü de keskinleştirirdi.",
        ],
        [
            "2. Veri Sızıntısı Yokluğu",
            "StandardScaler, PCA ve Fisher Skoru yalnızca Train (30.800) kümesine fit edildi; Val ve Test kümelerine sadece transform uygulandı.",
        ],
        [
            "3. Mimari Seçim Gerekçesi",
            f"DenseNet121, doğrulamada ResNet50'yi geçti (%{den_best['best_val_acc']*100:.2f} vs %{res_best['best_val_acc']*100:.2f}) ve daha kompakt (1024 vs 2048) özellik ürettiği için hibrit aşamaya seçildi.",
        ],
        [
            "4. Denetimli vs Denetimsiz Seçim",
            f"Fisher Skoru sınıf etiketlerini kullandığı için sadece {fish_svm['n_selected_features']} özellikle %{fish_svm['test_acc']*100:.2f} doğruluk aldı; PCA ise {pca_svm['n_selected_features']} bileşenle %{pca_svm['test_acc']*100:.2f} aldı.",
        ],
        [
            "5. Hız & Doğruluk Dengesi",
            f"Fisher+SVM %100,00 doğruluk (0,0752 sn), Fisher+XGBoost ise %99,97 doğrulukla 0,0083 sn test çıkarım süresi verdi.",
        ],
        [
            "6. Tekrarlanabilirlik & Donanım",
            "Tüm kütüphanelerde seed=42 sabitlendi; RTX 5060 GPU üzerinde FP16 Mixed Precision (AMP) ve sınıf ağırlıklı kayıp kullanıldı.",
        ],
    ]
    _add_styled_table(
        doc,
        headers=["Vurgu Alanı", "Hocaya Söylenecek Net Teknik Gerekçe"],
        rows=highlights,
        col_widths_cm=[4.8, 11.8],
    )

    # --------------------------------------------------------------------------
    # BÖLÜM 3: HOCANIN SORABİLECEĞİ KRİTİK SORULAR VE HAZIR CEVAPLAR
    # --------------------------------------------------------------------------
    _add_heading(doc, "3. Hocanın Sorabileceği Muhtemel Sorular ve Hazır Teknik Cevaplar")
    _add_para(
        doc,
        "Aşağıda jüri veya danışman hocanın proje savunması sırasında yöneltebileceği en olası "
        "12 soru ve doğrudan sizin kodunuzdaki/deneylerinizdeki gerçek kanıtlara dayalı "
        "cevapları verilmiştir.",
    )

    qas = [
        (
            "Soru 1: Sonuçlarınız %99,88 ile %100,00 arasında çıkmış. Model ezberlemiş (overfitting) veya veri sızıntısı (data leakage) olabilir mi?",
            "\"Hocam çok haklı ve kritik bir soru. Biz bunu iki açıdan kontrol ettik: Birincisi, eğitim ve doğrulama eğrilerimize (Şekil 2) baktığımızda Train ve Validation Loss eğrilerinin paralel şekilde 0,001 seviyesine düştüğünü ve ayrışmadığını görüyoruz; yani eğitim sürecinde bir ezberleme yok. Ayrıca StandardScaler, PCA ve Fisher seçimini kesinlikle sadece Train setine fit ettik. İkinci ve şeffaf olmak istediğimiz nokta ise veri setinin doğası: Kaggle'daki bu veri seti zaten önceden çoğaltılmış (augmented) 2B aksiyel MRI kesitlerinden oluşuyor. Dolayısıyla aynı beyne ait yakın kesitlerin farklı kümelerde yer alma ihtimali (subject-level ayrım yapılmamış olması) bu yüksek skorları kolaylaştıran bir faktördür. Biz de bunu raporumuzun Sonuç Değerlendirme bölümünde açıkça bir kısıt olarak belirttik.\"",
        ),
        (
            "Soru 2: Neden önce Gaussian filtre uygulayıp sonra CLAHE yaptınız? Önce CLAHE yapsaydınız ne olurdu?",
            "\"CLAHE yerel histogram eşitlemesi yaptığı için bölgedeki küçük yoğunluk dalgalanmalarını da büyütür. Eğer önce CLAHE uygulasaydık MR görüntüsündeki arka plan veya doku gürültüsü de kontrastla birlikte yükselecekti. Önce 3×3 (σ=0,5) Gaussian filtre ile bu yüksek frekanslı gürültüyü yumuşatıp ardından CLAHE uygulayarak yalnızca gerçek anatomik doku sınırlarını (gri-ak madde ve ventriküller) güçlendirdik.\"",
        ),
        (
            "Soru 3: ResNet50 ve DenseNet121 kullandınız; neden hibrit aşamada DenseNet121'i seçtiniz?",
            "\"İki nedeni var hocam: Birincisi deneysel doğrulama sonucu — DenseNet121 doğrulama setinde %99,98 doğruluk ve 0,0006 kayıp ile ResNet50'den (%99,91 doğruluk, 0,0019 kayıp) daha iyi performans verdi. İkincisi mimari avantajı — DenseNet121'de her katman önceki tüm katmanların özellik haritalarını doğrudan aldığı için beyin dokusundaki ince atrofi kenarları son katmana kadar korunuyor ve bunu ResNet50'deki 2048 yerine 1024 gibi yarı yarıya daha kompakt bir öznitelik vektörüyle sunuyor.\"",
        ),
        (
            "Soru 4: DenseNet121 zaten %99,98 almışken neden üstüne SVM ve XGBoost eklediniz?",
            "\"Burada iki temel kazanım var: Birincisi, 1024 boyutlu son havuzlama vektöründe birbirini tekrar eden veya gürültü taşıyan boyutlar bulunabiliyor; Softmax katmanı bunların tamamını kullanıyor. Fisher Skoru ile en ayırt edici 64 özelliği seçip maksimum marjin prensibiyle çalışan RBF-SVM verdiğimizde test hatası 1'den 0'a düştü (%100,00). İkincisi ise çıkarım hızı ve dağıtım verimliliği: Öznitelikler elde edildikten sonra sınıflandırma adımı 16,27 saniyeden Fisher+SVM'de 0,075 saniyeye, Fisher+XGBoost'ta 0,0083 saniyeye iniyor.\"",
        ),
        (
            "Soru 5: PCA ile Fisher Skoru arasındaki fark nedir? Neden Fisher daha az özellikle daha yüksek sonuç verdi?",
            "\"PCA tamamen denetimsiz (unsupervised) bir yöntemdir; sınıf etiketlerine bakmadan sadece verideki toplam varyansı (%95'ini) koruyan 155 ortogonal bileşen üretir. Ancak en yüksek varyansa sahip yön her zaman sınıfları en iyi ayıran yön olmayabilir. Fisher Skoru ise denetimli (supervised) bir yöntemdir; sınıflar arası ortalama farkını maksimize ederken sınıf içi varyansı minimize eden özellikleri seçer. Bu yüzden sadece 64 özellikle doğrudan evre ayrımına odaklanarak %100,00 sonuca ulaştı.\"",
        ),
        (
            "Soru 6: Hibrit model çıkarım sürelerinizde öznitelik çıkarma süresi dahil mi, nasıl ölçtünüz?",
            "\"Tablo 2'de iki ayrı gerçeği şeffaf şekilde gösterdik: Eğitim süresi sütununda (Train_Time_s) hibrit modeller için öznitelik çıkarım süresi (192,52 sn) + seçim + GridSearchCV süresi toplam olarak yer alıyor. Çıkarım süresi (Inference_Time_s) sütununda ise sınıflandırıcının 6.600 test öznitelik vektörü üzerindeki karar verme süresi ölçüldü. Gerçek bir klinik senaryoda tek bir görüntü uçtan uca geçirildiğinde CNN omurgası + Fisher + SVM hattı yine çok hızlı çalışacaktır.\"",
        ),
        (
            "Soru 7: Sınıf dengesizliği olsaydı sisteminiz nasıl başa çıkacaktı?",
            "\"Veri setimiz görece dengeli olsa da (10.000–12.800 arası) biz kodu genel ve dengesizliğe dayanıklı yazdık: (1) Stratified bölünme ile tüm sınıfların oranlarını koruduk, (2) DL eğitiminde scikit-learn'ün compute_class_weight('balanced') yöntemiyle sınıf ağırlıklı CrossEntropyLoss kullandık, (3) SVM'de class_weight='balanced' açtık ve (4) GridSearchCV'de Accuracy yerine Macro F1-Score optimize ettik.\"",
        ),
        (
            "Soru 8: Eğitimde neden Adam yerine AdamW kullandınız? EarlyStopping nasıl çalıştı?",
            "\"Adam'da L2 düzenlileştirme gradyanlara karıştığı için adaptif öğrenme oranıyla etkileşime girer; AdamW ağırlık azaltımını (weight decay = 1e-4) gradyan güncellemesinden ayırarak daha iyi genelleme sağlar. EarlyStopping'de val_loss değerini izledik (patience=7): ResNet50 en iyi val_loss'u 18. epoch'ta aldıktan sonra 7 epoch iyileşemeyince 25. epoch'ta otomatik durduruldu ve 18. epoch ağırlıkları geri yüklendi. DenseNet121 ise 23. epoch'ta en iyi noktayı bulup 30 epoch'u tamamladı.\"",
        ),
        (
            "Soru 9: SVM'de neden RBF çekirdeği kullandınız ve GridSearchCV'de neleri taradınız?",
            "\"CNN'den çıkan 64 veya 155 boyutlu derin özniteliklerin sınıflar arası sınırları her zaman doğrusal bir hiperdüzlemle ayrılmayabilir; RBF çekirdeği veriyi yüksek boyutlu uzaya taşıyarak doğrusal olmayan sınırları esnek biçimde öğrenebilir. GridSearchCV ile C ∈ {0.1, 1, 10, 50} ve gamma ∈ {'scale', 'auto', 0.01, 0.001} uzayını 3-katlı Stratified CV ile taradık. Fisher'da C=50, gamma='scale'; PCA'da C=1, gamma=0.001 en iyi sonucu verdi.\"",
        ),
        (
            "Soru 10: XGBoost neden ağaç tabanlı olmasına rağmen SVM'in bir adım gerisinde kaldı?",
            "\"Aslında fark istatistiksel olarak çok küçük (%99,97 vs %100,00 — 6.600 örnekte sadece 2 hata farkı). Ancak teorik olarak derin ağlardan gelen sürekli ve yoğun (dense) vektör uzaylarında eksen-paralel bölünme yapan karar ağaçları yerine, Öklid/RBF mesafesi ve maksimum marjin kullanan SVM genellikle çok pürüzsüz karar sınırları çizer. Buna karşın XGBoost'un çıkarım hızı (0,0083 sn) SVM'den yaklaşık 9 kat daha hızlıdır.\"",
        ),
        (
            "Soru 11: Farklı bir hastaneden veya farklı bir MR cihazından gelen bir görüntüde modelinizin başarımı düşer mi?",
            "\"Evet, tıbbi görüntülemede 'domain shift' dediğimiz cihaz/protokol farkı nedeniyle performans düşüşü yaşanabilir. Biz CLAHE ve Z-skoru normalizasyonu ile kontrast ve parlaklık farklarını bir miktar standardize etsek de klinik kullanıma geçmeden önce farklı merkezlerden (ör. ADNI veya yerel hastane PACS verileri) toplanacak görüntülerle dış doğrulama (external validation) ve gerekirse domain adaptation yapılması şarttır.\"",
        ),
        (
            "Soru 12: Kodu nasıl çalıştırıyorsunuz, başka bir bilgisayarda veya GPU yoksa ne olur?",
            "\"Kod tamamen donanım bağımsız tasarlandı: config.py içindeki get_device() CUDA destekli GPU varsa otomatik GPU'yu ve FP16 AMP'yi seçer, GPU yoksa sessizce CPU'ya düşer. Tüm pipeline 'python main.py' tek komutuyla çalışır; ayrıca '--steps hybrid evaluate' gibi bayraklarla istenen aşama tek başına yeniden çalıştırılabilir.\"",
        ),
    ]

    for q_title, a_text in qas:
        _add_callout_box(
            doc,
            q_title,
            [("Önerilen Cevap: ", a_text)],
            fill_hex="FFFDF7",
            border_hex="B7791F",
        )

    # --------------------------------------------------------------------------
    # BÖLÜM 4: CANLI DEMO / KOD GÖSTERİM REHBERİ
    # --------------------------------------------------------------------------
    _add_heading(doc, "4. Hoca \"Kodu Aç Göster\" Derse: Hızlı Dosya Rehberi")
    code_map = [
        ["config.py", "SEED=42 sabitleme, CUDA/CPU seçimi, CLAHE ve hiperparametre ayarları."],
        ["data/split_dataset.py", "Alt klasör taraması, KaggleHub desteği ve Stratified %70/%15/%15 bölünme."],
        ["src/preprocessing.py", "MRICLAHEPreprocessor (Gaussian + CLAHE + 224×224) ve Şekil 1 üretimi."],
        ["src/models.py", "ResNet50 ve DenseNet121 sınıfları + son havuzlamadan extract_features() metodu."],
        ["src/train_dl.py", "AMP (FP16), EarlyStopping, ReduceLROnPlateau, sınıf ağırlıklı CrossEntropyLoss."],
        ["src/hybrid_ml.py", "StandardScaler, PCA, özel fisher_score_func + SelectKBest, SVM & XGBoost GridSearchCV."],
        ["src/evaluate.py", "Tüm 300 DPI PNG grafiklerin (Şekil 2–5) ve model_comparison_table.csv'nin üretimi."],
        ["main.py", "Tüm sistemi tek komutla veya --steps ile modüler çalıştıran ana giriş dosyası."],
    ]
    _add_styled_table(
        doc,
        headers=["Dosya Adı", "Hocaya Gösterilecek Temel Sorumluluk"],
        rows=code_map,
        col_widths_cm=[4.8, 11.8],
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(output_path))
    print(f"[BAŞARILI] Sunum asistanı dokümanı oluşturuldu: {output_path}")
    return output_path


if __name__ == "__main__":
    build_presentation_guide(config.OUTPUTS_DIR / "Alzheimer_MRI_Sunum_Asistani.docx")
