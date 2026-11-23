"""
Alzheimer MRI Evreleme Projesi - Akademik Makale Formatında Word Raporu Üretici (generate_report.py)

`outputs/` klasöründeki deneysel sonuçları (`model_comparison_table.csv`,
`dl_training_summary.json`, `hybrid_models/hybrid_results.json`) ve 300 DPI
görselleri okuyarak Microsoft Word (.docx) formatında eksiksiz akademik makale
raporu (`outputs/Alzheimer_MRI_Evreleme_Makale_Raporu.docx`) üretir.

Özellikler:
- A4 sayfa düzeni, her yönden 2.5 cm kenar boşluğu
- Times New Roman 12 pt ana metin, 1.15 satır aralığı, iki yana yaslı (JUSTIFY)
- Gerçek yerel Word denklemleri (OMML: m:oMathPara / m:oMath)
- Akademik biçimli düzenlenebilir Word tabloları (Tablo 1 & Tablo 2)
- 300 DPI yüksek çözünürlüklü şekiller (Şekil 1–5)
- Birebir sıralı APA 7 kaynakça ve sıkı atıf/URL kuralları
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn
from docx.shared import Cm, Inches, Pt, RGBColor

import config


# ==============================================================================
# OMML (OFFICE MATH MARKUP LANGUAGE) YEREL WORD DENKLEM OLUŞTURUCU
# ==============================================================================
M_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"
W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def _m(tag: str) -> OxmlElement:
    """OMML (m:) ad alanında yeni bir XML öğesi oluşturur."""
    return OxmlElement(f"m:{tag}")


def _w(tag: str) -> OxmlElement:
    """WordprocessingML (w:) ad alanında yeni bir XML öğesi oluşturur."""
    return OxmlElement(f"w:{tag}")


def m_r(text: str, italic: bool = True, size_half_pt: int = 24) -> OxmlElement:
    """
    OMML matematik koşusu (<m:r>) oluşturur.
    italic=False ise düz (roman) fonksiyon/birim stili (<m:sty m:val="p"/>) uygular.
    """
    r = _m("r")
    if not italic:
        r_pr = _m("rPr")
        sty = _m("sty")
        sty.set(qn("m:val"), "p")
        r_pr.append(sty)
        r.append(r_pr)

    w_rpr = _w("rPr")
    r_fonts = _w("rFonts")
    r_fonts.set(qn("w:ascii"), "Cambria Math")
    r_fonts.set(qn("w:hAnsi"), "Cambria Math")
    sz = _w("sz")
    sz.set(qn("w:val"), str(size_half_pt))
    sz_cs = _w("szCs")
    sz_cs.set(qn("w:val"), str(size_half_pt))
    w_rpr.append(r_fonts)
    w_rpr.append(sz)
    w_rpr.append(sz_cs)
    r.append(w_rpr)

    t = _m("t")
    if text.startswith(" ") or text.endswith(" "):
        t.set(qn("xml:space"), "preserve")
    t.text = text
    r.append(t)
    return r


def m_op(text: str) -> OxmlElement:
    """Matematiksel operatör/noktalama koşusu oluşturur."""
    return m_r(text, italic=False)


def m_frac(num_elems: list[OxmlElement], den_elems: list[OxmlElement]) -> OxmlElement:
    """Kesir (<m:f>) nesnesi oluşturur."""
    f = _m("f")
    num = _m("num")
    den = _m("den")
    for el in num_elems:
        num.append(el)
    for el in den_elems:
        den.append(el)
    f.append(num)
    f.append(den)
    return f


def m_sup(base_elems: list[OxmlElement], sup_elems: list[OxmlElement]) -> OxmlElement:
    """Üst simge (<m:sSup>) nesnesi oluşturur."""
    ssup = _m("sSup")
    e = _m("e")
    sup = _m("sup")
    for el in base_elems:
        e.append(el)
    for el in sup_elems:
        sup.append(el)
    ssup.append(e)
    ssup.append(sup)
    return ssup


def m_sub(base_elems: list[OxmlElement], sub_elems: list[OxmlElement]) -> OxmlElement:
    """Alt simge (<m:sSub>) nesnesi oluşturur."""
    ssub = _m("sSub")
    e = _m("e")
    sub = _m("sub")
    for el in base_elems:
        e.append(el)
    for el in sub_elems:
        sub.append(el)
    ssub.append(e)
    ssub.append(sub)
    return ssub


def m_nary(
    symbol: str,
    sub_elems: list[OxmlElement],
    sup_elems: list[OxmlElement],
    expr_elems: list[OxmlElement],
) -> OxmlElement:
    """Toplam/çarpım gibi N-ary operatör (<m:nary>) nesnesi oluşturur."""
    nary = _m("nary")
    nary_pr = _m("naryPr")
    chr_el = _m("chr")
    chr_el.set(qn("m:val"), symbol)
    lim_loc = _m("limLoc")
    lim_loc.set(qn("m:val"), "subSup")
    nary_pr.append(chr_el)
    nary_pr.append(lim_loc)
    nary.append(nary_pr)

    sub = _m("sub")
    sup = _m("sup")
    e = _m("e")
    for el in sub_elems:
        sub.append(el)
    for el in sup_elems:
        sup.append(el)
    for el in expr_elems:
        e.append(el)
    nary.append(sub)
    nary.append(sup)
    nary.append(e)
    return nary


def m_delim(
    inner_elems: list[OxmlElement],
    beg_chr: str = "(",
    end_chr: str = ")",
) -> OxmlElement:
    """Otomatik ölçeklenen parantez/köşeli/küme parantezi (<m:d>) oluşturur."""
    d = _m("d")
    d_pr = _m("dPr")
    b = _m("begChr")
    b.set(qn("m:val"), beg_chr)
    e_chr = _m("endChr")
    e_chr.set(qn("m:val"), end_chr)
    d_pr.append(b)
    d_pr.append(e_chr)
    d.append(d_pr)

    e = _m("e")
    for el in inner_elems:
        e.append(el)
    d.append(e)
    return d


def add_omml_equation(
    doc: Document,
    math_elements: list[OxmlElement],
    eq_number: str | None = None,
) -> None:
    """
    Belgeye ortalı gerçek bir Word denklemi (m:oMathPara -> m:oMath) ekler.
    İsteğe bağlı denklem numarasını (örn. "(1)") aynı paragrafın sağ tarafına hizalı
    sekme durağıyla veya denklem etiketiyle yerleştirir.
    """
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.line_spacing = 1.15

    omath_para = _m("oMathPara")
    omath_para_pr = _m("oMathParaPr")
    jc = _m("jc")
    jc.set(qn("m:val"), "center")
    omath_para_pr.append(jc)
    omath_para.append(omath_para_pr)

    omath = _m("oMath")
    for el in math_elements:
        omath.append(el)
    if eq_number:
        # Denklem numarasını yerel denklem bloğu içinde sağda görsel netlikle belirt
        omath.append(m_op(f"      ({eq_number})"))

    omath_para.append(omath)
    p._p.append(omath_para)


# ==============================================================================
# WORD BİÇİMLENDİRME VE TABLO / ŞEKİL YARDIMCILARI
# ==============================================================================
def configure_document_styles(doc: Document) -> None:
    """A4 sayfa boyutu, 2.5 cm kenar boşlukları ve Times New Roman 12pt stilini ayarlar."""
    for section in doc.sections:
        section.page_width = Cm(21.0)
        section.page_height = Cm(29.7)
        section.top_margin = Cm(2.5)
        section.bottom_margin = Cm(2.5)
        section.left_margin = Cm(2.5)
        section.right_margin = Cm(2.5)

    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(12)
    normal.font.color.rgb = RGBColor(0, 0, 0)
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    normal.paragraph_format.line_spacing = 1.15
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY


def add_body_paragraph(
    doc: Document,
    text: str,
    bold_prefix: str | None = None,
    italic: bool = False,
    align: WD_ALIGN_PARAGRAPH = WD_ALIGN_PARAGRAPH.JUSTIFY,
    space_after: float = 6,
) -> None:
    """Times New Roman 12pt, 1.15 satır aralıklı standart akademik paragraf ekler."""
    p = doc.add_paragraph()
    p.alignment = align
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(space_after)

    if bold_prefix:
        r_pre = p.add_run(bold_prefix)
        r_pre.font.name = "Times New Roman"
        r_pre.font.size = Pt(12)
        r_pre.bold = True
        r_pre.italic = italic

    r = p.add_run(text)
    r.font.name = "Times New Roman"
    r.font.size = Pt(12)
    r.italic = italic


def add_section_heading(doc: Document, title: str, space_before: float = 14) -> None:
    """Ana bölüm başlığı ekler (Times New Roman 13pt Kalın)."""
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.space_before = Pt(space_before)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.keep_with_next = True
    r = p.add_run(title)
    r.font.name = "Times New Roman"
    r.font.size = Pt(13)
    r.bold = True


def add_subsection_heading(doc: Document, title: str) -> None:
    """Alt bölüm başlığı ekler (Times New Roman 12pt Kalın)."""
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.keep_with_next = True
    r = p.add_run(title)
    r.font.name = "Times New Roman"
    r.font.size = Pt(12)
    r.bold = True


def add_paragraph_with_hyperlink(
    doc: Document,
    before_text: str,
    url: str,
    after_text: str,
) -> None:
    """Metin içine tıklanabilir köprü (Hyperlink) içeren paragraf ekler."""
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.space_after = Pt(6)

    r1 = p.add_run(before_text)
    r1.font.name = "Times New Roman"
    r1.font.size = Pt(12)

    part = p.part
    r_id = part.relate_to(
        url,
        "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink",
        is_external=True,
    )
    hyperlink = OxmlElement("w:hyperlink")
    hyperlink.set(qn("r:id"), r_id)
    new_run = OxmlElement("w:r")
    r_pr = OxmlElement("w:rPr")
    r_fonts = OxmlElement("w:rFonts")
    r_fonts.set(qn("w:ascii"), "Times New Roman")
    r_fonts.set(qn("w:hAnsi"), "Times New Roman")
    sz = OxmlElement("w:sz")
    sz.set(qn("w:val"), "24")
    color = OxmlElement("w:color")
    color.set(qn("w:val"), "0563C1")
    u = OxmlElement("w:u")
    u.set(qn("w:val"), "single")
    r_pr.append(r_fonts)
    r_pr.append(sz)
    r_pr.append(color)
    r_pr.append(u)
    new_run.append(r_pr)
    t = OxmlElement("w:t")
    t.text = url
    new_run.append(t)
    hyperlink.append(new_run)
    p._p.append(hyperlink)

    r2 = p.add_run(after_text)
    r2.font.name = "Times New Roman"
    r2.font.size = Pt(12)


def _set_cell_shading(cell, fill_hex: str) -> None:
    shading = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    cell._tc.get_or_add_tcPr().append(shading)


def _apply_academic_table_borders(table) -> None:
    """
    Akademik makale (booktabs tarzı) tablo kenarlıkları:
    Üst kenar kalın (1.5 pt), başlık altı orta kalın (1.0 pt), alt kenar kalın (1.5 pt),
    dikey çizgiler yok, iç satırlarda ince gri ayrım çizgisi.
    """
    tbl_pr = table._tbl.tblPr
    borders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        '<w:top w:val="single" w:sz="12" w:space="0" w:color="000000"/>'
        '<w:bottom w:val="single" w:sz="12" w:space="0" w:color="000000"/>'
        '<w:insideH w:val="single" w:sz="4" w:space="0" w:color="CCCCCC"/>'
        '<w:left w:val="none"/>'
        '<w:right w:val="none"/>'
        '<w:insideV w:val="none"/>'
        "</w:tblBorders>"
    )
    tbl_pr.append(borders)

    # Başlık satırının alt çizgisini belirginleştir
    for cell in table.rows[0].cells:
        tc_pr = cell._tc.get_or_add_tcPr()
        tc_borders = parse_xml(
            f'<w:tcBorders {nsdecls("w")}>'
            '<w:top w:val="single" w:sz="12" w:space="0" w:color="000000"/>'
            '<w:bottom w:val="single" w:sz="8" w:space="0" w:color="000000"/>'
            "</w:tcBorders>"
        )
        tc_pr.append(tc_borders)


def add_academic_table(
    doc: Document,
    caption: str,
    headers: list[str],
    rows: list[list[str]],
    font_size_pt: float = 9.5,
    col_widths_cm: list[float] | None = None,
    left_align_first_col: bool = True,
) -> None:
    """Düzenlenebilir, gölgeli başlıklı ve akademik kenarlıklı Word tablosu ekler."""
    p_cap = doc.add_paragraph()
    p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_cap.paragraph_format.space_before = Pt(8)
    p_cap.paragraph_format.space_after = Pt(4)
    p_cap.paragraph_format.keep_with_next = True
    r_cap = p_cap.add_run(caption)
    r_cap.font.name = "Times New Roman"
    r_cap.font.size = Pt(10.5)
    r_cap.bold = True

    table = doc.add_table(rows=len(rows) + 1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False

    # Başlık satırı
    hdr_cells = table.rows[0].cells
    for c_idx, h_text in enumerate(headers):
        cell = hdr_cells[c_idx]
        _set_cell_shading(cell, "E6ECF5")
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT if (left_align_first_col and c_idx == 0) else WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.line_spacing = 1.05
        r = p.add_run(h_text)
        r.font.name = "Times New Roman"
        r.font.size = Pt(font_size_pt)
        r.bold = True

    # Veri satırları
    for r_idx, row_vals in enumerate(rows, start=1):
        row_cells = table.rows[r_idx].cells
        is_last = r_idx == len(rows)
        for c_idx, val in enumerate(row_vals):
            cell = row_cells[c_idx]
            if r_idx % 2 == 0:
                _set_cell_shading(cell, "F8FAFC")
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if (left_align_first_col and c_idx == 0) else WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(2)
            p.paragraph_format.line_spacing = 1.05
            r = p.add_run(str(val))
            r.font.name = "Times New Roman"
            r.font.size = Pt(font_size_pt)
            if c_idx == 0 or (is_last and "Toplam" in str(row_vals[0])):
                r.bold = True

    if col_widths_cm:
        for row in table.rows:
            for idx, w_cm in enumerate(col_widths_cm):
                row.cells[idx].width = Cm(w_cm)

    _apply_academic_table_borders(table)

    p_sp = doc.add_paragraph()
    p_sp.paragraph_format.space_before = Pt(0)
    p_sp.paragraph_format.space_after = Pt(6)


def add_academic_figure(
    doc: Document,
    image_path: Path,
    caption: str,
    width_cm: float = 15.5,
) -> None:
    """300 DPI grafiği belgeye ortalı olarak gömer ve altına akademik Şekil açıklaması ekler."""
    if not image_path.exists():
        raise FileNotFoundError(f"Görsel dosyası bulunamadı: {image_path}")

    p_img = doc.add_paragraph()
    p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_img.paragraph_format.space_before = Pt(8)
    p_img.paragraph_format.space_after = Pt(4)
    p_img.paragraph_format.keep_with_next = True
    run_img = p_img.add_run()
    run_img.add_picture(str(image_path), width=Cm(width_cm))

    p_cap = doc.add_paragraph()
    p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_cap.paragraph_format.space_before = Pt(0)
    p_cap.paragraph_format.space_after = Pt(8)
    r_cap = p_cap.add_run(caption)
    r_cap.font.name = "Times New Roman"
    r_cap.font.size = Pt(10.5)
    r_cap.italic = True


# ==============================================================================
# YEREL WORD DENKLEMLERİ (1 - 9)
# ==============================================================================
def add_eq1_gaussian(doc: Document) -> None:
    """Denklem (1): 2D Gauss Filtresi G(x, y)"""
    elems = [
        m_r("G"),
        m_delim([m_r("x"), m_op(","), m_r("y")]),
        m_op(" = "),
        m_frac(
            [m_op("1")],
            [m_op("2"), m_r("π"), m_sup([m_r("σ")], [m_op("2")])],
        ),
        m_op(" "),
        m_r("exp", italic=False),
        m_delim(
            [
                m_op("−"),
                m_frac(
                    [m_sup([m_r("x")], [m_op("2")]), m_op(" + "), m_sup([m_r("y")], [m_op("2")])],
                    [m_op("2"), m_sup([m_r("σ")], [m_op("2")])],
                ),
            ]
        ),
    ]
    add_omml_equation(doc, elems, eq_number="1")


def add_eq2_clahe(doc: Document) -> None:
    """Denklem (2): CLAHE histogram kırpma eşiği β"""
    elems = [
        m_r("β"),
        m_op(" = "),
        m_frac([m_r("M"), m_op(" · "), m_r("N")], [m_r("L")]),
        m_delim(
            [
                m_op("1 + "),
                m_frac([m_r("α")], [m_op("100")]),
                m_delim([m_sub([m_r("s")], [m_r("max", italic=False)]), m_op(" − 1")]),
            ]
        ),
    ]
    add_omml_equation(doc, elems, eq_number="2")


def add_eq3_zscore(doc: Document) -> None:
    """Denklem (3): Z-Skoru piksel normalizasyonu I_norm(x, y, c)"""
    elems = [
        m_sub([m_r("I")], [m_r("norm", italic=False)]),
        m_delim([m_r("x"), m_op(","), m_r("y"), m_op(","), m_r("c")]),
        m_op(" = "),
        m_frac(
            [
                m_r("I"),
                m_delim([m_r("x"), m_op(","), m_r("y"), m_op(","), m_r("c")]),
                m_op(" − "),
                m_sub([m_r("μ")], [m_r("c")]),
            ],
            [m_sub([m_r("σ")], [m_r("c")])],
        ),
    ]
    add_omml_equation(doc, elems, eq_number="3")


def add_eq4_resnet(doc: Document) -> None:
    """Denklem (4): ResNet artık blok denklemi"""
    elems = [
        m_r("y"),
        m_op(" = "),
        m_r("F"),
        m_delim(
            [
                m_r("x"),
                m_op(", "),
                m_delim([m_sub([m_r("W")], [m_r("i")])], beg_chr="{", end_chr="}"),
            ]
        ),
        m_op(" + "),
        m_r("x"),
    ]
    add_omml_equation(doc, elems, eq_number="4")


def add_eq5_densenet(doc: Document) -> None:
    """Denklem (5): DenseNet yoğun bağlantı denklemi"""
    elems = [
        m_sub([m_r("x")], [m_r("l")]),
        m_op(" = "),
        m_sub([m_r("H")], [m_r("l")]),
        m_delim(
            [
                m_delim(
                    [
                        m_sub([m_r("x")], [m_op("0")]),
                        m_op(", "),
                        m_sub([m_r("x")], [m_op("1")]),
                        m_op(", …, "),
                        m_sub([m_r("x")], [m_r("l"), m_op("−1")]),
                    ],
                    beg_chr="[",
                    end_chr="]",
                )
            ]
        ),
    ]
    add_omml_equation(doc, elems, eq_number="5")


def add_eq6_softmax_ce(doc: Document) -> None:
    """Denklem (6a & 6b): Softmax olasılık dağılımı ve Sınıf Ağırlıklı Çapraz Entropi Kaybı"""
    elems_softmax = [
        m_sub([m_r("p̂")], [m_r("i,c")]),
        m_op(" = "),
        m_frac(
            [m_r("exp", italic=False), m_delim([m_sub([m_r("z")], [m_r("i,c")])])],
            [
                m_nary(
                    "∑",
                    [m_r("k"), m_op("=1")],
                    [m_r("C")],
                    [m_r("exp", italic=False), m_delim([m_sub([m_r("z")], [m_r("i,k")])])],
                )
            ],
        ),
    ]
    add_omml_equation(doc, elems_softmax, eq_number="6")

    elems_ce = [
        m_sub([m_r("L")], [m_r("CE", italic=False)]),
        m_op(" = −"),
        m_frac([m_op("1")], [m_r("N")]),
        m_nary(
            "∑",
            [m_r("i"), m_op("=1")],
            [m_r("N")],
            [
                m_nary(
                    "∑",
                    [m_r("c"), m_op("=1")],
                    [m_r("C")],
                    [
                        m_sub([m_r("w")], [m_r("c")]),
                        m_op(" · "),
                        m_sub([m_r("y")], [m_r("i,c")]),
                        m_op(" "),
                        m_r("log", italic=False),
                        m_delim([m_sub([m_r("p̂")], [m_r("i,c")])]),
                    ],
                )
            ],
        ),
    ]
    add_omml_equation(doc, elems_ce, eq_number="7")


def add_eq7_pca(doc: Document) -> None:
    """Denklem (8): PCA kovaryans matrisi ve özdeğer ayrışımı"""
    elems_cov = [
        m_r("Σ", italic=False),
        m_op(" = "),
        m_frac([m_op("1")], [m_r("n"), m_op(" − 1")]),
        m_sup([m_sub([m_r("X")], [m_r("c")])], [m_r("T")]),
        m_sub([m_r("X")], [m_r("c")]),
        m_op(",   "),
        m_r("Σ", italic=False),
        m_sub([m_r("v")], [m_r("k")]),
        m_op(" = "),
        m_sub([m_r("λ")], [m_r("k")]),
        m_sub([m_r("v")], [m_r("k")]),
    ]
    add_omml_equation(doc, elems_cov, eq_number="8")


def add_eq8_fisher(doc: Document) -> None:
    """Denklem (9): Çok sınıflı Fisher Skoru F(j)"""
    elems = [
        m_r("F"),
        m_delim([m_r("j")]),
        m_op(" = "),
        m_frac(
            [
                m_nary(
                    "∑",
                    [m_r("c"), m_op("=1")],
                    [m_r("C")],
                    [
                        m_sub([m_r("n")], [m_r("c")]),
                        m_sup(
                            [m_delim([m_sub([m_r("μ")], [m_r("c,j")]), m_op(" − "), m_sub([m_r("μ")], [m_r("j")])])],
                            [m_op("2")],
                        ),
                    ],
                )
            ],
            [
                m_nary(
                    "∑",
                    [m_r("c"), m_op("=1")],
                    [m_r("C")],
                    [
                        m_sub([m_r("n")], [m_r("c")]),
                        m_sup([m_sub([m_r("σ")], [m_r("c,j")])], [m_op("2")]),
                    ],
                ),
                m_op(" + "),
                m_r("ϵ"),
            ],
        ),
    ]
    add_omml_equation(doc, elems, eq_number="9")


def add_eq9_metrics(doc: Document) -> None:
    """Denklem (10-13): Accuracy, Precision, Recall ve F1-Score formülleri"""
    add_omml_equation(
        doc,
        [
            m_r("Accuracy", italic=False),
            m_op(" = "),
            m_frac(
                [m_r("TP", italic=False), m_op(" + "), m_r("TN", italic=False)],
                [
                    m_r("TP", italic=False),
                    m_op(" + "),
                    m_r("TN", italic=False),
                    m_op(" + "),
                    m_r("FP", italic=False),
                    m_op(" + "),
                    m_r("FN", italic=False),
                ],
            ),
        ],
        eq_number="10",
    )
    add_omml_equation(
        doc,
        [
            m_r("Precision", italic=False),
            m_op(" = "),
            m_frac(
                [m_r("TP", italic=False)],
                [m_r("TP", italic=False), m_op(" + "), m_r("FP", italic=False)],
            ),
            m_op(",    "),
            m_r("Recall", italic=False),
            m_op(" = "),
            m_frac(
                [m_r("TP", italic=False)],
                [m_r("TP", italic=False), m_op(" + "), m_r("FN", italic=False)],
            ),
        ],
        eq_number="11",
    )
    add_omml_equation(
        doc,
        [
            m_r("F1-Score", italic=False),
            m_op(" = "),
            m_frac(
                [m_op("2 · "), m_r("Precision", italic=False), m_op(" · "), m_r("Recall", italic=False)],
                [m_r("Precision", italic=False), m_op(" + "), m_r("Recall", italic=False)],
            ),
        ],
        eq_number="12",
    )


# ==============================================================================
# VERİ OKUMA VE MAKALE RAPORU OLUŞTURMA
# ==============================================================================
def load_experiment_artifacts() -> tuple[pd.DataFrame, dict, dict, dict[str, dict[str, int]]]:
    """outputs/ ve data/splits/ altındaki deneysel verileri okur."""
    comp_df = pd.read_csv(config.OUTPUTS_DIR / "model_comparison_table.csv")

    with open(config.OUTPUTS_DIR / "dl_training_summary.json", "r", encoding="utf-8") as f:
        dl_summary = json.load(f)

    with open(config.HYBRID_MODELS_DIR / "hybrid_results.json", "r", encoding="utf-8") as f:
        hybrid_summary = json.load(f)

    split_counts: dict[str, dict[str, int]] = {}
    for split_name in ("train", "val", "test"):
        split_df = pd.read_csv(config.SPLITS_DIR / f"{split_name}.csv")
        split_counts[split_name] = split_df["label_name"].value_counts().to_dict()

    return comp_df, dl_summary, hybrid_summary, split_counts


def build_report(output_docx_path: Path) -> Path:
    """Tüm kurallara uygun akademik makale raporunu üretir ve .docx olarak kaydeder."""
    comp_df, dl_summary, hybrid_summary, split_counts = load_experiment_artifacts()

    doc = Document()
    configure_document_styles(doc)

    # --------------------------------------------------------------------------
    # BAŞLIK VE YAZAR BİLGİLERİ
    # --------------------------------------------------------------------------
    p_tr_title = doc.add_paragraph()
    p_tr_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_tr_title.paragraph_format.space_before = Pt(4)
    p_tr_title.paragraph_format.space_after = Pt(6)
    p_tr_title.paragraph_format.line_spacing = 1.15
    r_tr = p_tr_title.add_run(
        "Evrişimli Sinir Ağları ve Öznitelik Seçimi Tabanlı Hibrit Makine Öğrenmesi "
        "Yaklaşımları ile Beyin MR Görüntülerinden Alzheimer Hastalığı Evrelemesi"
    )
    r_tr.font.name = "Times New Roman"
    r_tr.font.size = Pt(14)
    r_tr.bold = True

    p_en_title = doc.add_paragraph()
    p_en_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_en_title.paragraph_format.space_before = Pt(0)
    p_en_title.paragraph_format.space_after = Pt(10)
    p_en_title.paragraph_format.line_spacing = 1.15
    r_en = p_en_title.add_run(
        "Staging of Alzheimer’s Disease from Brain MRI Images Using Convolutional Neural "
        "Networks and Feature Selection-Based Hybrid Machine Learning Approaches"
    )
    r_en.font.name = "Times New Roman"
    r_en.font.size = Pt(12.5)
    r_en.italic = True
    r_en.bold = True

    # Öğrenci / Yazar Bilgi Şablonu
    p_author = doc.add_paragraph()
    p_author.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_author.paragraph_format.space_before = Pt(2)
    p_author.paragraph_format.space_after = Pt(12)
    p_author.paragraph_format.line_spacing = 1.15
    author_lines = [
        ("Yazar: ", True), ("Mustafa Fıstıkçı\n", False),
        ("Bölüm: ", True), ("Bilgisayar Mühendisliği\n", False),
        ("Üniversite / Kurum: ", True), ("[Üniversite Adı Giriniz] — Bitirme / Dönem Projesi Raporu\n", False),
        ("Danışman: ", True), ("[Danışman Unvanı ve Adı Soyadı Giriniz]", False),
    ]
    for txt, is_b in author_lines:
        ra = p_author.add_run(txt)
        ra.font.name = "Times New Roman"
        ra.font.size = Pt(11)
        ra.bold = is_b

    # --------------------------------------------------------------------------
    # 1- ÖZET (KESİNLİKLE ATIF YOK)
    # --------------------------------------------------------------------------
    add_section_heading(doc, "1- Özet", space_before=6)
    add_body_paragraph(
        doc,
        "Alzheimer hastalığı, beyinde geri dönüşümsüz nörodejeneratif hasara yol açarak bellek, "
        "bilişsel işlevler ve günlük yaşam aktivitelerinde ilerleyici bozulmaya neden olan en yaygın "
        "demans türüdür. Hastalığın erken ve ara evrelerde doğru biçimde tespit edilmesi, klinik "
        "müdahalelerin zamanında planlanması açısından kritik önem taşımaktadır. Bu çalışmada, "
        "beyin Manyetik Rezonans Görüntüleme (MRI) kesitleri kullanılarak dört farklı klinik evrenin "
        "(NonDemented, VeryMildDemented, MildDemented ve ModerateDemented) yüksek doğrulukla "
        "sınıflandırılması amacıyla uçtan uca derin öğrenme ve öznitelik seçimi tabanlı hibrit "
        "makine öğrenmesi yaklaşımları karşılaştırmalı olarak geliştirilmiştir. Toplam 44.000 MRI "
        "görüntüsü katmanlı örnekleme ile %70 eğitim (30.800), %15 doğrulama (6.600) ve %15 test "
        "(6.600) kümelerine ayrılmıştır. Görüntü kalitesini artırmak amacıyla tüm kesitlere "
        "Gaussian gürültü azaltma filtresi ve Kontrast Sınırlı Uyarlamalı Histogram Eşitleme "
        "(CLAHE) ön işleme hattı uygulanmıştır. Model 1 kapsamında ResNet50 ve DenseNet121 "
        "mimarileri transfer öğrenme, karma hassasiyetli eğitim (AMP), sınıf ağırlıklı çapraz "
        "entropi kaybı ve erken durdurma mekanizmalarıyla eğitilmiştir. Model 2 kapsamında ise "
        "doğrulama başarımı en yüksek olan DenseNet121 modelinin son havuzlama katmanından "
        "çıkarılan 1024 boyutlu derin öznitelikler, Temel Bileşenler Analizi (PCA, 155 bileşen) ve "
        "Fisher Skoru (SelectKBest, 64 öznitelik) yöntemleriyle indirgendikten sonra Radyal Tabanlı "
        "Fonksiyon (RBF) çekirdekli Destek Vektör Makineleri (SVM) ve XGBoost algoritmalarıyla "
        "sınıflandırılmıştır. Deneysel sonuçlar, test kümesinde ResNet50 modelinin %99,88, "
        "DenseNet121 modelinin %99,98 genel doğruluk elde ettiğini; hibrit yaklaşımlardan "
        "DenseNet121+Fisher+SVM modelinin ise %100,00 doğruluk, 1,0000 Makro AUC ve yalnızca "
        "0,0752 saniyelik toplam test çıkarım süresiyle hem en yüksek başarımı hem de yüksek "
        "hesaplama verimliliğini sunduğunu ortaya koymaktadır.",
    )
    add_body_paragraph(
        doc,
        "Alzheimer Hastalığı Evrelemesi, Beyin MRI, CLAHE, Transfer Öğrenme, DenseNet121, "
        "Fisher Skoru, PCA, Destek Vektör Makineleri, XGBoost.",
        bold_prefix="Anahtar Kelimeler: ",
        space_after=8,
    )

    add_body_paragraph(
        doc,
        "Alzheimer’s disease is the most prevalent form of dementia, causing irreversible "
        "neurodegenerative damage and progressive deterioration in memory, cognitive functions, and "
        "daily living activities. Accurate staging in early and intermediate phases is vital for "
        "timely clinical intervention and patient management. In this study, end-to-end deep "
        "learning and feature selection-based hybrid machine learning pipelines are systematically "
        "developed and compared for four-class staging (NonDemented, VeryMildDemented, "
        "MildDemented, and ModerateDemented) using brain Magnetic Resonance Imaging (MRI) slices. "
        "A balanced multi-class dataset comprising 44,000 MRI images was partitioned via stratified "
        "sampling into 70% training (30,800), 15% validation (6,600), and 15% test (6,600) subsets. "
        "To suppress acquisition noise and enhance local tissue boundaries, all images underwent a "
        "standardized preprocessing pipeline combining Gaussian denoising and Contrast Limited "
        "Adaptive Histogram Equalization (CLAHE). In Model 1, ResNet50 and DenseNet121 backbones "
        "were fine-tuned end-to-end using Automatic Mixed Precision (AMP), class-weighted "
        "cross-entropy loss, and early stopping. In Model 2, 1024-dimensional deep representations "
        "extracted from the global pooling layer of the best-performing CNN (DenseNet121) were "
        "compressed using Principal Component Analysis (PCA, 155 components) and Fisher Score "
        "selection (64 features), followed by grid-searched RBF-kernel Support Vector Machines "
        "(SVM) and XGBoost classifiers. Experimental evaluations on the independent test set "
        "demonstrate that ResNet50 and DenseNet121 achieve 99.88% and 99.98% accuracy, "
        "respectively, whereas the proposed DenseNet121+Fisher+SVM hybrid model attains 100.00% "
        "accuracy and 1.0000 Macro AUC with a total test inference time of only 0.0752 seconds.",
        bold_prefix="Abstract: ",
        italic=True,
    )
    add_body_paragraph(
        doc,
        "Alzheimer’s Disease Staging, Brain MRI, CLAHE, Transfer Learning, DenseNet121, "
        "Fisher Score, PCA, Support Vector Machines, XGBoost.",
        bold_prefix="Keywords: ",
        italic=True,
        space_after=10,
    )

    # --------------------------------------------------------------------------
    # 2- GİRİŞ
    # --------------------------------------------------------------------------
    add_section_heading(doc, "2- Giriş")
    add_body_paragraph(
        doc,
        "Alzheimer hastalığı (AH), serebral kortekste ve özellikle hafıza oluşumundan sorumlu "
        "hipokampus bölgesinde ilerleyici nöron kaybı, amiloid plak birikimi ve nörofibriler yumak "
        "oluşumuyla karakterize edilen kronik bir nörodejeneratif hastalıktır [1]. Hastalığın "
        "erken evrelerinde ortaya çıkan hafif bilişsel bozuklukların sağlıklı yaşlanma "
        "süreçlerinden ayırt edilmesi klinik tanı açısından büyük önem taşır. Yapısal Manyetik "
        "Rezonans Görüntüleme (MRI), iyonlaştırıcı radyasyon içermemesi ve gri madde, ak madde ile "
        "beyin omurilik sıvısı sınırlarını yüksek uzamsal çözünürlükte görüntüleyebilmesi "
        "sayesinde AH tanı ve evreleme protokollerinde temel biyobelirteç aracı olarak kabul "
        "edilmektedir [2]. Ancak erken ve çok hafif demans evrelerindeki ventriküler genişleme ve "
        "kortikal atrofi değişimlerinin radyologlar tarafından görsel incelemeyle tutarlı biçimde "
        "ayırt edilmesi zaman alıcı ve gözlemciler arası değişkenliğe açık bir süreçtir.",
    )
    add_body_paragraph(
        doc,
        "Son yıllarda derin öğrenme ve özellikle Evrişimli Sinir Ağları (CNN), tıbbi görüntü "
        "analizinde insan uzman düzeyine yaklaşan veya onu aşan otomatik öznitelik çıkarım "
        "yeteneği sayesinde yaygın biçimde benimsenmiştir [3], [4]. ResNet50 gibi artık bağlantılı "
        "mimariler derin ağlarda gradyan sönümlemesi sorununu aşarken [5], DenseNet121 gibi yoğun "
        "bağlantılı mimariler tüm katmanlar arası öznitelik yeniden kullanımını maksimize ederek "
        "doku sınırlarındaki ince anatomik örüntüleri etkin şekilde koruyabilmektedir [6]. Buna "
        "karşın, ham MRI kesitlerindeki tarayıcı kaynaklı gürültü ve düşük yerel kontrast, derin "
        "ağların erken evre atrofi ipuçlarını yakalamasını zorlaştırabilmektedir. Bu nedenle "
        "Kontrast Sınırlı Uyarlamalı Histogram Eşitleme (CLAHE) gibi yerel kontrast iyileştirme "
        "yöntemlerinin biyomedikal ön işleme hatlarına dahil edilmesi kritik bir gereklilik haline "
        "gelmiştir [7].",
    )
    add_body_paragraph(
        doc,
        "Mevcut literatürde uçtan uca CNN modelleri yaygın olarak kullanılsa da, son tam bağlı "
        "(fully connected) Softmax katmanları yüksek boyutlu öznitelik uzayında gereksiz veya "
        "gürültülü boyutları filtreleyememekte, bu da özellikle küçük tıbbi veri alt gruplarında "
        "genelleme kararlılığını ve çıkarım hızını sınırlandırabilmektedir. Bu çalışmada mevcut "
        "yöntemlerin söz konusu eksikliklerini gidermek üzere iki aşamalı bütünleşik bir mimari "
        "önerilmektedir: (i) Gaussian gürültü azaltma + CLAHE ön işleme hattıyla zenginleştirilmiş "
        "MRI görüntüleri üzerinde ResNet50 ve DenseNet121 modellerinin sınıf ağırlıklı transfer "
        "öğrenme ile eğitilmesi, (ii) en başarılı CNN omurgasından elde edilen yüksek seviyeli "
        "özniteliklerin PCA ve Fisher Skoru ile ayrıştırıcı bir alt uzaya indirgenerek SVM ve "
        "XGBoost sınıflandırıcılarıyla birleştirilmesi. Önerilen hibrit yaklaşım, derin "
        "temsillerin semantik gücünü klasik makine öğrenmesi algoritmalarının marjin "
        "maksimizasyonu ve düşük çıkarım gecikmesi avantajlarıyla bir araya getirmektedir.",
    )

    # --------------------------------------------------------------------------
    # 2.5 - İLGİLİ ÇALIŞMALAR
    # --------------------------------------------------------------------------
    add_section_heading(doc, "2.5 - İlgili Çalışmalar")
    add_body_paragraph(
        doc,
        "Alzheimer hastalığının beyin MRI verilerinden otomatik olarak sınıflandırılması ve "
        "evrelenmesi üzerine son on yılda çok sayıda derin öğrenme ve hibrit makine öğrenmesi "
        "çalışması yürütülmüştür. İlk dönem çalışmalardan birinde Islam ve Zhang [8], OASIS veri "
        "seti üzerinde dört sınıflı AH evrelemesi için çoklu CNN topluluk (ensemble) mimarisi "
        "önermiş ve %93,18 doğruluk oranına ulaşmıştır. Benzer şekilde Hon ve Khan [9], ImageNet "
        "üzerinde ön eğitimli VGG16 ve Inception-V4 ağlarını transfer öğrenme ile MRI kesitlerine "
        "uyarlamış; sınırlı örnek sayısına rağmen %96,25 düzeyinde ikili sınıflandırma başarımı "
        "raporlamıştır.",
    )
    add_body_paragraph(
        doc,
        "Öznitelik çıkarımı ve çoklu modalite temelli yaklaşımlarda Suk ve ark. [10], MRI ve PET "
        "görüntülerinden derin Boltzmann makineleri ile hiyerarşik öznitelikler öğrenip SVM "
        "sınıflandırıcı ile birleştirerek AD/MCI ayrımında klasik el yapımı özniteliklere göre "
        "anlamlı iyileşme sağlamıştır. Payan ve Montana [11], seyrek otokodlayıcılar ve 3B "
        "evrişimli ağlar kullanarak ADNI veri setinde %89,47 üç sınıflı doğruluk bildirmiştir. "
        "Sarraf ve Tofighi [12] ise LeNet-5 ve GoogleNet mimarilerini fMRI ve yapısal MRI "
        "verilerine uygulayarak %96’nın üzerinde sınıflandırma performansı elde etmiştir.",
    )
    add_body_paragraph(
        doc,
        "Daha güncel kapsamlı bir değerlendirmede Wen ve ark. [13], literatürdeki CNN tabanlı AH "
        "çalışmalarının önemli bir bölümünde veri sızıntısı (data leakage) ve tekrarlanabilirlik "
        "eksiklikleri bulunduğuna dikkat çekmiş; eğitim, doğrulama ve test kümelerinin katmanlı "
        "ve izole biçimde ayrılmasının bilimsel geçerlilik için zorunlu olduğunu vurgulamıştır. "
        "Önceki çalışmalardan farklı olarak bizim çalışmamızda: (1) veri sızıntısını önleyen "
        "katmanlı bölünme ve yalnızca eğitim setine oturtulan istatistiksel ölçeklendirme/seçim "
        "protokolü uygulanmış, (2) ResNet50 [5] ve DenseNet121 [6] omurgaları aynı ön işleme "
        "hattında adil biçimde kıyaslanmış, (3) varyans koruyan PCA ile denetimli Fisher Skoru "
        "öznitelik seçimi hem SVM hem de XGBoost sınıflandırıcılarıyla çapraz değerlendirilerek "
        "doğruluk ve çıkarım hızı dengesi kapsamlı biçimde ortaya konmuştur.",
    )

    # --------------------------------------------------------------------------
    # 3- YÖNTEM
    # --------------------------------------------------------------------------
    add_section_heading(doc, "3- Yöntem")

    add_subsection_heading(doc, "3.1. Veri Seti ve Katmanlı (Stratified) Bölünme")
    add_paragraph_with_hyperlink(
        doc,
        "Bu çalışmada deneysel değerlendirmeler için Kaggle platformunda açık erişimli olarak "
        "sunulan çok sınıflı ve dengeli Alzheimer MRI veri seti (",
        "https://www.kaggle.com/datasets/aryansinghal10/alzheimers-multiclass-dataset-equal-and-augmented",
        ") kullanılmıştır. Veri seti, klinik demans şiddetine göre dört sınıftan oluşan toplam "
        "44.000 adet aksiyel beyin MRI kesiti içermektedir: NonDemented (12.800 görüntü), "
        "VeryMildDemented (11.200 görüntü), MildDemented (10.000 görüntü) ve ModerateDemented "
        "(10.000 görüntü). Tüm deneylerin tekrarlanabilirliğini sağlamak amacıyla Python, NumPy "
        "ve PyTorch kütüphanelerinde rastgelelik tohumu seed = 42 olarak sabitlenmiştir. Veri "
        "sızıntısını engellemek ve sınıf oranlarını her alt kümede korumak için iki aşamalı "
        "katmanlı örnekleme (Stratified Train/Validation/Test) uygulanarak veri %70 Eğitim "
        "(30.800), %15 Doğrulama (6.600) ve %15 Test (6.600) olacak şekilde bölünmüştür. "
        "Oluşturulan veri dağılımı Tablo 1’de sunulmaktadır.",
    )

    # Tablo 1: Veri seti dağılımı
    t1_headers = ["Sınıf Etiketi", "Eğitim (%70)", "Doğrulama (%15)", "Test (%15)", "Toplam"]
    t1_rows = []
    tot_tr = tot_va = tot_te = 0
    for cname in config.CLASS_NAMES:
        tr_c = int(split_counts["train"].get(cname, 0))
        va_c = int(split_counts["val"].get(cname, 0))
        te_c = int(split_counts["test"].get(cname, 0))
        tot_c = tr_c + va_c + te_c
        tot_tr += tr_c
        tot_va += va_c
        tot_te += te_c
        t1_rows.append([cname, f"{tr_c:,}".replace(",", "."), f"{va_c:,}".replace(",", "."), f"{te_c:,}".replace(",", "."), f"{tot_c:,}".replace(",", ".")])
    t1_rows.append(
        [
            "Toplam",
            f"{tot_tr:,}".replace(",", "."),
            f"{tot_va:,}".replace(",", "."),
            f"{tot_te:,}".replace(",", "."),
            f"{(tot_tr + tot_va + tot_te):,}".replace(",", "."),
        ]
    )
    add_academic_table(
        doc,
        caption="Tablo 1. Veri setinin sınıflara göre katmanlı (Stratified) Eğitim, Doğrulama ve Test dağılımı.",
        headers=t1_headers,
        rows=t1_rows,
        font_size_pt=10,
        col_widths_cm=[5.2, 2.7, 2.9, 2.6, 2.6],
    )

    add_subsection_heading(doc, "3.2. Görüntü Ön İşleme Hattı (Gaussian + CLAHE + Normalizasyon)")
    add_body_paragraph(
        doc,
        "Beyin MRI kesitleri tek kanallı gri tonlamalı yoğunluk matrisine dönüştürüldükten sonra "
        "kontrast iyileştirme öncesinde yüksek frekanslı tarayıcı gürültüsünün abartılmasını "
        "önlemek amacıyla 3×3 çekirdek boyutunda ve σ = 0,5 standart sapmalı iki boyutlu "
        "Gaussian alçak geçiren filtreden geçirilmiştir. İki boyutlu Gaussian çekirdeği "
        "Denklem (1)’de tanımlanmaktadır:",
    )
    add_eq1_gaussian(doc)

    add_body_paragraph(
        doc,
        "Gürültü bastırma adımının ardından gri madde, ak madde ve ventrikül sınırlarındaki yerel "
        "kontrastı artırmak için CLAHE algoritması uygulanmıştır [7]. Klasik histogram eşitlemeden "
        "farklı olarak CLAHE, görüntüyü M×N boyutunda ayrık karolara (tileGridSize = 8×8) böler ve "
        "gürültü amplifikasyonunu engellemek amacıyla histogram tepelerini kırpma eşiği β ile "
        "sınırlandırır (clipLimit = 2,0). Teorik kırpma eşiği Denklem (2)’deki gibi ifade edilir:",
    )
    add_eq2_clahe(doc)

    add_body_paragraph(
        doc,
        "Burada L gri ton seviyesi sayısını, α kırpma faktörünü ve s_max izin verilen maksimum "
        "eğimi temsil etmektedir. Ön işlenen görüntüler 224×224 piksel çözünürlüğe yeniden "
        "boyutlandırılmış, ImageNet ön eğitimli ağlarla uyumluluk için 3 kanallı RGB biçimine "
        "dönüştürülmüş ve ImageNet kanal istatistikleri (μ = [0,485, 0,456, 0,406], "
        "σ = [0,229, 0,224, 0,225]) kullanılarak Denklem (3)’teki Z-skoru normalizasyonu ile "
        "ölçeklendirilmiştir [14]. Eğitim aşamasında aşırı öğrenmeyi azaltmak için yalnızca "
        "eğitim kümesine rastgele yatay çevirme (p = 0,5) ve ±7° küçük açısal döndürme "
        "uygulanmıştır. Ön işleme etkisinin görsel karşılaştırması Şekil 1’de verilmiştir.",
    )
    add_eq3_zscore(doc)

    add_academic_figure(
        doc,
        image_path=config.OUTPUTS_DIR / "preprocessing_comparison.png",
        caption="Şekil 1. ModerateDemented sınıfına ait örnek beyin MRI kesitinde ham görüntü, Gaussian + CLAHE ön işleme çıktısı ve mutlak piksel kontrast fark haritası.",
        width_cm=15.6,
    )

    add_subsection_heading(doc, "3.3. Derin Öğrenme Transfer Modelleri (Model 1: ResNet50 ve DenseNet121)")
    add_body_paragraph(
        doc,
        "Model 1 kapsamında ImageNet üzerinde ön eğitimli ResNet50 ve DenseNet121 omurgaları "
        "uçtan uca ince ayar (fine-tuning) ile eğitilmiştir. ResNet50 mimarisi, derin ağlarda "
        "bozulma (degradation) problemini çözmek için girdi özelliğini doğrudan çıktıya ekleyen "
        "artık (residual) bağlantılar kullanır ve son global ortalama havuzlama katmanında 2048 "
        "boyutlu öznitelik vektörü üretir [5]. Artık bloğun matematiksel tanımı Denklem (4)’tedir:",
    )
    add_eq4_resnet(doc)

    add_body_paragraph(
        doc,
        "DenseNet121 mimarisi ise her katmanın kendisinden önceki tüm katmanların öznitelik "
        "haritalarını girdi olarak aldığı yoğun bağlantı (dense connectivity) yapısına sahiptir "
        "ve son havuzlama katmanında 1024 boyutlu kompakt bir temsil çıkarır [6]. l-inci "
        "katmanın dönüşüm denklemi Denklem (5)’te verilmiştir:",
    )
    add_eq5_densenet(doc)

    add_body_paragraph(
        doc,
        "Her iki omurganın sonuna Dropout (p = 0,4), 256 nöronlu tam bağlı ReLU katmanı, ikinci "
        "Dropout (p = 0,2) ve 4 sınıflı çıkış katmanı eklenmiştir. Sınıf dengesizliğinin kayıp "
        "üzerindeki etkisini dengelemek için eğitim setinden hesaplanan ters frekanslı sınıf "
        "ağırlıkları w_c kullanılarak Softmax olasılıkları (Denklem 6) ve sınıf ağırlıklı çok "
        "sınıflı Çapraz Entropi Kaybı (Denklem 7) optimize edilmiştir [13]:",
    )
    add_eq6_softmax_ce(doc)

    add_subsection_heading(doc, "3.4. Hibrit Makine Öğrenmesi Yaklaşımı (Model 2: Özellik Seçimi + SVM / XGBoost)")
    add_body_paragraph(
        doc,
        "Model 2 aşamasında, doğrulama kümesinde en yüksek başarımı elde eden CNN modelinin "
        "(DenseNet121) son global ortalama havuzlama katmanından deterministik olarak 1024 "
        "boyutlu derin öznitelik vektörleri çıkarılmış ve yalnızca eğitim seti üzerinde "
        "StandardScaler ile sıfır ortalama, birim varyansa dönüştürülmüştür. Boyut indirgeme ve "
        "öznitelik seçimi için iki tamamlayıcı yöntem incelenmiştir. İlk olarak, Temel Bileşenler "
        "Analizi (PCA) ile verideki toplam varyansın %95’ini koruyan ortogonal özvektörler "
        "hesaplanmıştır (Denklem 8) [15]. İkinci olarak, denetimli bir öznitelik seçici olan "
        "Fisher Skoru (SelectKBest) kullanılarak sınıflar arası dağılımı maksimize ederken sınıf "
        "içi varyansı minimize eden en ayırt edici K = 64 öznitelik seçilmiştir (Denklem 9) [16]:",
    )
    add_eq7_pca(doc)
    add_eq8_fisher(doc)

    add_body_paragraph(
        doc,
        "Seçilen PCA ve Fisher öznitelik alt uzayları üzerinde iki farklı klasik makine öğrenmesi "
        "sınıflandırıcısı eğitilmiş ve 3-katlı katmanlı çapraz doğrulama (GridSearchCV, Macro-F1 "
        "ölçütlü) ile hiperparametreleri optimize edilmiştir: (i) Radyal Tabanlı Fonksiyon (RBF) "
        "çekirdekli, sınıf dengeli ve olasılık kalibrasyonlu Destek Vektör Makineleri (SVM) [17] "
        "ve (ii) gradyan artırmalı karar ağaçları temelli XGBoost sınıflandırıcısı [18].",
    )

    add_subsection_heading(doc, "3.5. Değerlendirme Metrikleri")
    add_body_paragraph(
        doc,
        "Tüm modellerin bağımsız test kümesi üzerindeki başarımı Genel Doğruluk (Accuracy), sınıf "
        "bazlı doğruluklar, Makro Kesinlik (Macro Precision), Makro Duyarlılık (Macro Recall), "
        "Makro F1-Skoru (Denklem 10–12), One-vs-Rest (OvR) ROC-AUC skorları ve toplam "
        "eğitim/çıkarım süreleri üzerinden nicel olarak kıyaslanmıştır:",
    )
    add_eq9_metrics(doc)

    # --------------------------------------------------------------------------
    # 4- DENEYSEL SONUÇLAR
    # --------------------------------------------------------------------------
    add_section_heading(doc, "4- Deneysel Sonuçlar")

    res_info = dl_summary["models"]["resnet50"]
    den_info = dl_summary["models"]["densenet121"]
    pca_svm = hybrid_summary["models"]["DenseNet121+PCA+SVM"]
    pca_xgb = hybrid_summary["models"]["DenseNet121+PCA+XGBoost"]
    fish_svm = hybrid_summary["models"]["DenseNet121+Fisher+SVM"]
    fish_xgb = hybrid_summary["models"]["DenseNet121+Fisher+XGBoost"]

    add_body_paragraph(
        doc,
        f"Tüm deneyler NVIDIA GeForce RTX 5060 GPU üzerinde PyTorch CUDA (FP16 Automatic Mixed "
        f"Precision aktif) ortamında gerçekleştirilmiştir. Model 1 eğitiminde AdamW optimizasyon "
        f"algoritması [19] (başlangıç öğrenme oranı lr = 1×10⁻⁴, ağırlık azaltımı 1×10⁻⁴), "
        f"batch boyutu 32, maksimum 30 epoch, ReduceLROnPlateau (faktör 0,5, sabır 3) ve "
        f"EarlyStopping (sabır 7) kullanılmıştır. Eğitim sürecinde ResNet50 modeli erken durdurma "
        f"mekanizması ile 25. epoch’ta sonlanmış ve en iyi doğrulama başarımını {res_info['best_epoch']}. "
        f"epoch’ta (Val Loss: {res_info['best_val_loss']:.4f}, Val Acc: %{res_info['best_val_acc']*100:.2f}) "
        f"yakalamıştır. DenseNet121 modeli ise 30 epoch boyunca eğitilerek en iyi doğrulama "
        f"performansına {den_info['best_epoch']}. epoch’ta (Val Loss: {den_info['best_val_loss']:.4f}, "
        f"Val Acc: %{den_info['best_val_acc']*100:.2f}) ulaşmış ve hibrit aşama için temel "
        f"öznitelik çıkarıcı olarak seçilmiştir [5], [6]. Modellerin eğitim ve doğrulama "
        f"eğrileri Şekil 2’de sunulmaktadır.",
    )

    add_academic_figure(
        doc,
        image_path=config.OUTPUTS_DIR / "training_curves.png",
        caption="Şekil 2. ResNet50 ve DenseNet121 modellerinin epoch bazlı Eğitim/Doğrulama Doğruluk ve Cross-Entropy Kayıp eğrileri.",
        width_cm=15.6,
    )

    add_body_paragraph(
        doc,
        f"Hibrit ML (Model 2) aşamasında, eğitilen DenseNet121 modelinden 30.800 eğitim, 6.600 "
        f"doğrulama ve 6.600 test örneği için 1024 boyutlu özniteliklerin çıkarılması toplam "
        f"192,52 saniye sürmüştür. PCA dönüşümü boyutu 1024’ten {pca_svm['n_selected_features']} "
        f"bileşene (%95,04 açıklanan varyans) indirgemiş [15], Fisher Skoru seçimi ise optimal "
        f"öznitelik sayısını doğrulama başarımı %99,98 ile K = {fish_svm['n_selected_features']} "
        f"olarak belirlemiştir [16]. GridSearchCV sonucunda en iyi hiperparametreler "
        f"DenseNet121+PCA+SVM için C = {pca_svm['best_params']['C']}, gamma = {pca_svm['best_params']['gamma']} [17]; "
        f"DenseNet121+Fisher+SVM için C = {fish_svm['best_params']['C']}, gamma = '{fish_svm['best_params']['gamma']}'; "
        f"DenseNet121+PCA+XGBoost için n_estimators = {pca_xgb['best_params']['n_estimators']}, "
        f"max_depth = {pca_xgb['best_params']['max_depth']}, lr = {pca_xgb['best_params']['learning_rate']} [18]; "
        f"DenseNet121+Fisher+XGBoost için ise n_estimators = {fish_xgb['best_params']['n_estimators']}, "
        f"max_depth = {fish_xgb['best_params']['max_depth']}, lr = {fish_xgb['best_params']['learning_rate']} "
        f"olarak bulunmuştur. Tüm modellerin bağımsız test seti (6.600 görüntü) üzerindeki "
        f"karşılaştırmalı sonuçları Tablo 2’de özetlenmiştir.",
    )

    # Tablo 2: CSV'den otomatik okunan tüm model sonuçları
    t2_headers = [
        "Model",
        "Genel\nAcc (%)",
        "NonDem\n(%)",
        "VeryMild\n(%)",
        "Mild\n(%)",
        "Moderate\n(%)",
        "Makro\nPrec (%)",
        "Makro\nRec (%)",
        "Makro\nF1 (%)",
        "Makro\nAUC",
        "Eğitim\n(sn)",
        "Çıkarım\n(sn)",
    ]
    t2_rows = []
    for _, row in comp_df.iterrows():
        t2_rows.append(
            [
                row["Model"],
                f"{row['Accuracy (%)']:.2f}",
                f"{row['Acc_NonDemented (%)']:.2f}",
                f"{row['Acc_VeryMildDemented (%)']:.2f}",
                f"{row['Acc_MildDemented (%)']:.2f}",
                f"{row['Acc_ModerateDemented (%)']:.2f}",
                f"{row['Macro_Precision (%)']:.2f}",
                f"{row['Macro_Recall (%)']:.2f}",
                f"{row['Macro_F1 (%)']:.2f}",
                f"{row['Macro_AUC']:.4f}",
                f"{row['Train_Time_s']:.2f}",
                f"{row['Inference_Time_s']:.4f}",
            ]
        )
    add_academic_table(
        doc,
        caption="Tablo 2. Tüm derin öğrenme ve hibrit makine öğrenmesi modellerinin test seti (6.600 MRI) üzerindeki karşılaştırmalı performansı.",
        headers=t2_headers,
        rows=t2_rows,
        font_size_pt=8.5,
        col_widths_cm=[3.4, 1.15, 1.15, 1.15, 1.05, 1.2, 1.15, 1.15, 1.15, 1.15, 1.15, 1.15],
    )

    add_body_paragraph(
        doc,
        "Tablo 2 ve Şekil 3 incelendiğinde, uçtan uca modellerde ResNet50’nin toplam 8 hatalı "
        "sınıflandırma (NonDemented ve VeryMildDemented arasında 7, NonDemented-MildDemented "
        "arasında 1) ile %99,88 doğruluk elde ettiği; DenseNet121’in ise yalnızca 1 "
        "NonDemented örneğini VeryMildDemented olarak etiketleyerek %99,98 doğruluk ve %99,99 "
        "Makro F1 skoruna ulaştığı görülmektedir. Bu bulgu, yoğun katman bağlantılarının erken "
        "evre hafif atrofi sınırlarını ayırt etmede artık mimariye göre daha hassas temsil "
        "ürettiğini doğrulamaktadır [6].",
    )

    add_academic_figure(
        doc,
        image_path=config.OUTPUTS_DIR / "confusion_matrix_dl.png",
        caption="Şekil 3. Model 1 (ResNet50 ve DenseNet121) yaklaşımlarının test seti üzerindeki sayısal ve yüzdelik karmaşıklık matrisleri.",
        width_cm=15.6,
    )

    add_body_paragraph(
        doc,
        "Hibrit modellerin karmaşıklık matrisleri (Şekil 4) incelendiğinde, özellikle denetimli "
        "Fisher Skoru ile seçilen 64 öznitelik üzerinde eğitilen DenseNet121+Fisher+SVM "
        "modelinin 6.600 test görüntüsünün tamamını hatasız sınıflandırarak %100,00 doğruluk, "
        "%100,00 Makro F1 ve 1,0000 Makro AUC değerine ulaştığı görülmektedir [16], [17]. "
        "DenseNet121+PCA+SVM yalnızca 1 ModerateDemented örneğinde hata yaparken (%99,98), "
        "DenseNet121+Fisher+XGBoost yalnızca 2 örnekte yanılmıştır (%99,97) [18]. Süre açısından "
        "karşılaştırıldığında, DenseNet121 test seti çıkarımı 16,2696 saniye sürerken, öznitelik "
        "çıkarımı sonrasındaki sınıflandırma adımı DenseNet121+Fisher+SVM için 0,0752 saniyeye, "
        "DenseNet121+Fisher+XGBoost için ise 0,0083 saniyeye düşmüştür. Ayrıca Şekil 5’teki "
        "One-vs-Rest ROC eğrileri hem en iyi derin öğrenme modelinin hem de en iyi hibrit modelin "
        "tüm evrelerde 1,0000 sınıf bazlı ve makro AUC değerleriyle kusursuz ayırt edicilik "
        "sergilediğini ortaya koymaktadır.",
    )

    add_academic_figure(
        doc,
        image_path=config.OUTPUTS_DIR / "confusion_matrix_hybrid.png",
        caption="Şekil 4. Model 2 (DenseNet121 + PCA/Fisher + SVM/XGBoost) hibrit sınıflandırıcılarının test seti karmaşıklık matrisleri.",
        width_cm=15.6,
    )

    add_academic_figure(
        doc,
        image_path=config.OUTPUTS_DIR / "roc_curves.png",
        caption="Şekil 5. En iyi derin öğrenme modeli (DenseNet121) ve en iyi hibrit model (DenseNet121+Fisher+SVM) için 4 sınıflı One-vs-Rest (OvR) ROC eğrileri ve AUC skorları.",
        width_cm=15.6,
    )

    # --------------------------------------------------------------------------
    # 5- SONUÇ DEĞERLENDİRME (KESİNLİKLE ATIF YOK)
    # --------------------------------------------------------------------------
    add_section_heading(doc, "5- Sonuç Değerlendirme")
    add_body_paragraph(
        doc,
        "Bu çalışmada beyin MRI görüntülerinden dört aşamalı Alzheimer hastalığı evrelemesi için "
        "Gaussian gürültü azaltma ve CLAHE ön işleme temelli uçtan uca derin öğrenme (ResNet50, "
        "DenseNet121) ile öznitelik seçimi tabanlı hibrit makine öğrenmesi (PCA/Fisher + "
        "SVM/XGBoost) yaklaşımları sistematik olarak geliştirilmiş ve değerlendirilmiştir. Elde "
        "edilen bulgular, yoğun bağlantılı DenseNet121 mimarisinin beyin dokusundaki ince "
        "yapısal farklılıkları ResNet50’ye kıyasla daha yüksek hassasiyetle yakaladığını (%99,98 "
        "vs. %99,88) göstermiştir. Dahası, DenseNet121’den çıkarılan 1024 boyutlu derin "
        "temsillerin Fisher Skoru ile en ayırt edici 64 özniteliğe indirgenip RBF çekirdekli SVM "
        "ile sınıflandırılması, test kümesindeki tüm örneklerde %100,00 doğruluk sağlayarak "
        "gereksiz öznitelik gürültüsünün elenmesinin genelleme başarımını artırdığını "
        "kanıtlamıştır. XGBoost tabanlı hibrit modeller ise %99,95–%99,97 doğruluk bandını "
        "korurken milisaniye seviyesinde çıkarım süreleriyle gerçek zamanlı klinik karar destek "
        "sistemleri için son derece elverişli bir alternatif sunmuştur.",
    )
    add_body_paragraph(
        doc,
        "Çalışmanın güçlü yönlerinin yanında dikkate alınması gereken bazı kısıtları da "
        "bulunmaktadır. İlk olarak, kullanılan veri seti 2 boyutlu aksiyel MRI kesitlerinden ve "
        "çoğaltılmış (augmented) örneklerden oluştuğu için hasta bazlı (subject-level) üç boyutlu "
        "hacimsel bilgi tam olarak modellenememiş ve kesitler arası olası korelasyonlar "
        "dışlanamamıştır. İkinci olarak, deneyler tek bir çok sınıflı veri kaynağı üzerinde "
        "yürütülmüştür; farklı MRI tarayıcı protokollerine ve çok merkezli klinik popülasyonlara "
        "yönelik dış doğrulama henüz gerçekleştirilmemiştir. Gelecek çalışmalarda, hasta bazlı "
        "katmanlı bölünme uygulanan çok merkezli 3B MRI hacimleri üzerinde Vision Transformer ve "
        "3B-DenseNet hibrit mimarilerinin incelenmesi, Grad-CAM veya SHAP tabanlı açıklanabilir "
        "yapay zekâ haritalarının radyolojik doğrulamaya dahil edilmesi ve hafifletilmiş model "
        "ağırlıklarının klinik PACS sistemlerine entegrasyonu hedeflenmektedir.",
    )

    # --------------------------------------------------------------------------
    # 6. KAYNAKLAR (APA 7 FORMATI, METİN İÇİ SIRAYA BİREBİR UYUMLU, URL YOK)
    # --------------------------------------------------------------------------
    add_section_heading(doc, "6. Kaynaklar")

    references = [
        "[1] McKhann, G. M., Knopman, D. S., Chertkow, H., Hyman, B. T., Jack, C. R., Kawas, C. H., Klunk, W. E., Koroshetz, W. J., Manly, J. J., Mayeux, R., Mohs, R. C., Morris, J. C., Rossor, M. N., Scheltens, P., Carrillo, M. C., Thies, B., Weintraub, S., & Phelps, C. H. (2011). The diagnosis of dementia due to Alzheimer’s disease: Recommendations from the National Institute on Aging-Alzheimer’s Association workgroups on diagnostic guidelines for Alzheimer’s disease. Alzheimer’s & Dementia, 7(3), 263–269.",
        "[2] Jack, C. R., Bernstein, M. A., Fox, N. C., Thompson, P., Alexander, G., Harvey, D., Borowski, B., Britson, P. J., Whitwell, J. L., Ward, C., Dale, A. M., Felmlee, J. P., Gunter, J. L., Hill, D. L. G., Killiany, R., Schuff, N., Fox-Bosetti, S., Lin, C., Studholme, C., ... Weiner, M. W. (2008). The Alzheimer’s Disease Neuroimaging Initiative (ADNI): MRI methods. Journal of Magnetic Resonance Imaging, 27(4), 685–691.",
        "[3] LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. Nature, 521(7553), 436–444.",
        "[4] Litjens, G., Kooi, T., Bejnordi, B. E., Setio, A. A. A., Ciompi, F., Ghafoorian, M., van der Laan, J. A. W. M., van Ginneken, B., & Sánchez, C. I. (2017). A survey on deep learning in medical image analysis. Medical Image Analysis, 42, 60–88.",
        "[5] He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 770–778.",
        "[6] Huang, G., Liu, Z., Van Der Maaten, L., & Weinberger, K. Q. (2017). Densely connected convolutional networks. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 4700–4708.",
        "[7] Zuiderveld, K. (1994). Contrast limited adaptive histogram equalization. In P. S. Heckbert (Ed.), Graphics Gems IV (pp. 474–485). Academic Press.",
        "[8] Islam, J., & Zhang, Y. (2018). Brain MRI analysis for Alzheimer’s disease diagnosis using an ensemble system of deep convolutional neural networks. Brain Informatics, 5(2), 1–14.",
        "[9] Hon, M., & Khan, N. M. (2017). Towards Alzheimer’s disease classification through transfer learning. Proceedings of the IEEE International Conference on Bioinformatics and Biomedicine (BIBM), 1166–1169.",
        "[10] Suk, H.-I., Lee, S.-W., & Shen, D. (2014). Hierarchical feature representation and multimodal fusion with deep learning for AD/MCI diagnosis. NeuroImage, 101, 569–582.",
        "[11] Payan, A., & Montana, G. (2015). Predicting Alzheimer’s disease: A neuroimaging study with 3D convolutional neural networks. arXiv preprint arXiv:1502.02506.",
        "[12] Sarraf, S., & Tofighi, G. (2016). DeepAD: Alzheimer’s disease classification via deep convolutional neural networks using MRI and fMRI. bioRxiv, 070441.",
        "[13] Wen, J., Thibeau-Sutre, E., Diaz-Melo, M., Samper-González, J., Routier, A., Bottani, S., Dormont, D., Durrleman, S., Burgos, N., & Colliot, O. (2020). Convolutional neural networks for classification of Alzheimer’s disease: Overview and reproducible evaluation. Medical Image Analysis, 63, 101694.",
        "[14] Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., & Fei-Fei, L. (2009). ImageNet: A large-scale hierarchical image database. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 248–255.",
        "[15] Jolliffe, I. T., & Cadima, J. (2016). Principal component analysis: A review and recent developments. Philosophical Transactions of the Royal Society A: Mathematical, Physical and Engineering Sciences, 374(2065), 20150202.",
        "[16] Gu, Q., Li, Z., & Han, J. (2011). Generalized Fisher score for feature selection. Proceedings of the Twenty-Seventh Conference on Uncertainty in Artificial Intelligence (UAI), 266–273.",
        "[17] Cortes, C., & Vapnik, V. (1995). Support-vector networks. Machine Learning, 20(3), 273–297.",
        "[18] Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 785–794.",
        "[19] Loshchilov, I., & Hutter, F. (2019). Decoupled weight decay regularization. International Conference on Learning Representations (ICLR).",
    ]

    for ref in references:
        p_ref = doc.add_paragraph()
        p_ref.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p_ref.paragraph_format.left_indent = Cm(0.75)
        p_ref.paragraph_format.first_line_indent = Cm(-0.75)
        p_ref.paragraph_format.line_spacing = 1.15
        p_ref.paragraph_format.space_before = Pt(0)
        p_ref.paragraph_format.space_after = Pt(4)
        r_ref = p_ref.add_run(ref)
        r_ref.font.name = "Times New Roman"
        r_ref.font.size = Pt(11)

    output_docx_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(output_docx_path))
    return output_docx_path


def verify_generated_docx(docx_path: Path) -> None:
    """Oluşturulan .docx dosyasının akademik, atıf, URL, denklem, tablo ve şekil kurallarını doğrular."""
    doc = Document(str(docx_path))
    texts = [p.text for p in doc.paragraphs]

    def _idx_of(prefix: str) -> int:
        for i, t in enumerate(texts):
            if t.strip().startswith(prefix):
                return i
        raise AssertionError(f"Bölüm başlığı bulunamadı: {prefix}")

    i_ozet = _idx_of("1- Özet")
    i_giris = _idx_of("2- Giriş")
    i_ilgili = _idx_of("2.5 - İlgili Çalışmalar")
    i_yontem = _idx_of("3- Yöntem")
    i_sonuc_deg = _idx_of("5- Sonuç Değerlendirme")
    i_kaynak = _idx_of("6. Kaynaklar")

    # 1. Özet ve Sonuç Değerlendirme bölümlerinde atıf olmamalı
    sec1_text = "\n".join(texts[i_ozet:i_giris])
    sec5_text = "\n".join(texts[i_sonuc_deg:i_kaynak])
    assert not re.search(r"\[\d+\]", sec1_text), "HATA: 1- Özet içinde atıf bulundu!"
    assert not re.search(r"\[\d+\]", sec5_text), "HATA: 5- Sonuç Değerlendirme içinde atıf bulundu!"

    # 2. Bölüm 2-4 arası atıf sırası kontrolü
    body_text = "\n".join(texts[i_giris:i_sonuc_deg])
    seen_cites: list[int] = []
    for m in re.finditer(r"\[(\d+)\]", body_text):
        num = int(m.group(1))
        if num not in seen_cites:
            seen_cites.append(num)
    expected_seq = list(range(1, len(seen_cites) + 1))
    assert seen_cites == expected_seq, f"HATA: Metin içi atıf sırası bozuk: {seen_cites} != {expected_seq}"

    # Kaynak sayısı kontrolü
    ref_paras = [t for t in texts[i_kaynak + 1 :] if t.strip().startswith("[")]
    assert len(ref_paras) == len(seen_cites), f"HATA: Kaynakça sayısı ({len(ref_paras)}) atıf sayısıyla ({len(seen_cites)}) uyuşmuyor!"

    # 3. URL kontrolü (yalnızca 3- Yöntem içinde tek Kaggle URL'si olmalı)
    full_xml = doc._element.xml
    urls = re.findall(r"https?://[^\s<\"']+", full_xml)
    kaggle_url = "https://www.kaggle.com/datasets/aryansinghal10/alzheimers-multiclass-dataset-equal-and-augmented"
    non_schema_urls = [u for u in urls if not u.startswith("http://schemas.") and not u.startswith("http://www.w3.org/") and not u.startswith("http://purl.org/")]
    assert non_schema_urls == [kaggle_url], f"HATA: Beklenmeyen URL bulundu: {non_schema_urls}"

    # 4. Yerel Word denklemleri (m:oMathPara ve m:oMath), tablolar ve gömülü görseller kontrolü
    omath_para_count = len(doc._element.xpath(".//m:oMathPara"))
    omath_count = len(doc._element.xpath(".//m:oMath"))
    table_count = len(doc.tables)
    inline_shape_count = len(doc.inline_shapes)

    assert omath_para_count >= 12 and omath_count >= 12, f"HATA: Yetersiz OMML denklem sayısı: {omath_para_count}"
    assert table_count == 2, f"HATA: Beklenen tablo sayısı 2, bulunan: {table_count}"
    assert inline_shape_count == 5, f"HATA: Beklenen şekil sayısı 5, bulunan: {inline_shape_count}"

    print(
        f"[DOĞRULAMA BAŞARILI] {docx_path.name}\n"
        f"  - OMML Denklem (m:oMathPara / m:oMath): {omath_para_count}\n"
        f"  - Yerel Word Tablosu: {table_count}\n"
        f"  - Gömülü 300 DPI Şekil: {inline_shape_count}\n"
        f"  - Sıralı Atıflar: [1]..[{len(seen_cites)}] (Özet & Sonuç atıf sayısı: 0)\n"
        f"  - Kaggle Dışı URL Sayısı: 0"
    )


def main() -> None:
    out_path = config.OUTPUTS_DIR / "Alzheimer_MRI_Evreleme_Makale_Raporu.docx"
    generated = build_report(out_path)
    verify_generated_docx(generated)


if __name__ == "__main__":
    main()
