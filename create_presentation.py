#!/usr/bin/env python3
"""
Создание PowerPoint презентации для исследования
"""

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.dml.color import RGBColor

def create_presentation():
    prs = Presentation()

    # Установка размера слайдов (16:9)
    prs.slide_width = Inches(10)
    prs.slide_height = Inches(5.625)

    # === СЛАЙД 1: Титульный ===
    slide_layout = prs.slide_layouts[6]  # Blank
    slide = prs.slides.add_slide(slide_layout)

    # Заголовок
    title_box = slide.shapes.add_textbox(Inches(0.5), Inches(1.5), Inches(9), Inches(1.5))
    title_frame = title_box.text_frame
    title_frame.text = "ШИРОКОЗНАЧНАЯ ГЛАГОЛЬНАЯ ЛЕКСИКА В ГЕРМАНСКИХ ЯЗЫКАХ"
    title_para = title_frame.paragraphs[0]
    title_para.font.size = Pt(32)
    title_para.font.bold = True
    title_para.alignment = PP_ALIGN.CENTER

    # Подзаголовок
    subtitle_box = slide.shapes.add_textbox(Inches(0.5), Inches(3), Inches(9), Inches(1))
    subtitle_frame = subtitle_box.text_frame
    subtitle_frame.text = "Траектории семантического развития"
    subtitle_para = subtitle_frame.paragraphs[0]
    subtitle_para.font.size = Pt(24)
    subtitle_para.alignment = PP_ALIGN.CENTER

    # Дополнительная информация
    info_box = slide.shapes.add_textbox(Inches(0.5), Inches(4), Inches(9), Inches(1))
    info_frame = info_box.text_frame
    info_frame.text = "Компьютерный анализ четырёх германских языков (IV–XIV вв.)"
    info_para = info_frame.paragraphs[0]
    info_para.font.size = Pt(18)
    info_para.alignment = PP_ALIGN.CENTER

    # === СЛАЙД 2: Теоретические основы ===
    slide = prs.slides.add_slide(slide_layout)

    title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(9), Inches(0.6))
    title_frame = title_box.text_frame
    title_frame.text = "Что такое широкозначная лексика?"
    title_para = title_frame.paragraphs[0]
    title_para.font.size = Pt(28)
    title_para.font.bold = True

    content_box = slide.shapes.add_textbox(Inches(0.5), Inches(1), Inches(9), Inches(4))
    content_frame = content_box.text_frame
    content_frame.word_wrap = True

    p = content_frame.paragraphs[0]
    p.text = "Широкозначность = высокая степень семантической обобщённости слова"
    p.font.size = Pt(18)
    p.space_after = Pt(12)

    p = content_frame.add_paragraph()
    p.text = "В древних языках выполняет особую номинативную функцию"
    p.font.size = Pt(18)
    p.space_after = Pt(12)

    p = content_frame.add_paragraph()
    p.text = "Профилирует элементы референциальных сценариев базовой деятельности"
    p.font.size = Pt(18)
    p.space_after = Pt(12)

    p = content_frame.add_paragraph()
    p.text = "Связана с процессами: РАСШИРЕНИЕ, СУЖЕНИЕ, СМЕЩЕНИЕ значения"
    p.font.size = Pt(20)
    p.font.bold = True
    p.space_after = Pt(12)

    p = content_frame.add_paragraph()
    p.text = "Ключевая идея: Глагол 'делать' — пример широкой семантической зоны"
    p.font.size = Pt(18)
    p.font.italic = True

    # === СЛАЙД 3: Цель и задачи ===
    slide = prs.slides.add_slide(slide_layout)

    title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(9), Inches(0.6))
    title_frame = title_box.text_frame
    title_frame.text = "Цель и задачи исследования"
    title_para = title_frame.paragraphs[0]
    title_para.font.size = Pt(28)
    title_para.font.bold = True

    content_box = slide.shapes.add_textbox(Inches(0.5), Inches(1), Inches(9), Inches(4))
    content_frame = content_box.text_frame

    p = content_frame.paragraphs[0]
    p.text = "ЦЕЛЬ: Анализ траекторий семантического развития глагола 'делать'"
    p.font.size = Pt(20)
    p.font.bold = True
    p.space_after = Pt(16)

    p = content_frame.add_paragraph()
    p.text = "Задачи:"
    p.font.size = Pt(20)
    p.font.bold = True
    p.space_after = Pt(8)

    tasks = [
        "1. Выявить семантические кластеры употребления глагола в 4 языках",
        "2. Проследить диахроническую эволюцию (IV→XIV вв.)",
        "3. Сравнить trajectories развития в типологическом аспекте",
        "4. Визуализировать семантические сдвиги"
    ]

    for task in tasks:
        p = content_frame.add_paragraph()
        p.text = task
        p.font.size = Pt(18)
        p.space_after = Pt(8)
        p.level = 1

    p = content_frame.add_paragraph()
    p.text = "Гипотеза: Общий тренд от конкретного к абстрактному с различиями"
    p.font.size = Pt(18)
    p.font.italic = True
    p.space_after = Pt(8)

    # === СЛАЙД 4: Материал исследования ===
    slide = prs.slides.add_slide(slide_layout)

    title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(9), Inches(0.6))
    title_frame = title_box.text_frame
    title_frame.text = "Корпусы данных"
    title_para = title_frame.paragraphs[0]
    title_para.font.size = Pt(28)
    title_para.font.bold = True

    # Таблица
    table_box = slide.shapes.add_textbox(Inches(0.5), Inches(1), Inches(9), Inches(3.5))
    table_frame = table_box.text_frame
    table_frame.word_wrap = True

    header = "Язык | Период | Корпус | Вхождений\n"
    line1 = "Готский | IV в. | Gothic Bible | 140\n"
    line2 = "Др.-англ. | VIII-XI | Calgary | 221\n"
    line3 = "Др.-в.-нем. | X-XIII | Indiana MHG | 433\n"
    line4 = "Др.-исланд. | XI-XIV | Old Norse | 41+904\n"

    p = table_frame.paragraphs[0]
    p.text = header + line1 + line2 + line3 + line4
    p.font.size = Pt(16)
    p.font.name = 'Courier New'

    summary_box = slide.shapes.add_textbox(Inches(0.5), Inches(4.7), Inches(9), Inches(0.6))
    summary_frame = summary_box.text_frame
    summary_frame.text = "Итого: 4 языка, ~800 лет истории, 1739 вхождений"
    summary_para = summary_frame.paragraphs[0]
    summary_para.font.size = Pt(18)
    summary_para.font.bold = True

    # === СЛАЙД 5: Этимология ===
    slide = prs.slides.add_slide(slide_layout)

    title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(9), Inches(0.6))
    title_frame = title_box.text_frame
    title_frame.text = "От праиндоевропейского к германским языкам"
    title_para = title_frame.paragraphs[0]
    title_para.font.size = Pt(28)
    title_para.font.bold = True

    content_box = slide.shapes.add_textbox(Inches(0.5), Inches(1), Inches(9), Inches(4))
    content_frame = content_box.text_frame

    p = content_frame.paragraphs[0]
    p.text = "И.-е. корень: *dʰeh₁- 'класть, помещать'"
    p.font.size = Pt(20)
    p.space_after = Pt(8)

    p = content_frame.add_paragraph()
    p.text = "↓"
    p.font.size = Pt(24)
    p.alignment = PP_ALIGN.CENTER
    p.space_after = Pt(8)

    p = content_frame.add_paragraph()
    p.text = "Прото-германский: *dōjan / *dōn"
    p.font.size = Pt(20)
    p.alignment = PP_ALIGN.CENTER
    p.space_after = Pt(8)

    p = content_frame.add_paragraph()
    p.text = "↓"
    p.font.size = Pt(24)
    p.alignment = PP_ALIGN.CENTER
    p.space_after = Pt(8)

    languages = ["Готский τaujan", "Др.-англ. don", "Др.-в.-нем. tuon", "Др.-исл. dœ + gera"]

    p = content_frame.add_paragraph()
    p.text = "\n".join(languages)
    p.font.size = Pt(18)
    p.alignment = PP_ALIGN.CENTER
    p.space_after = Pt(12)

    p = content_frame.add_paragraph()
    p.text = "Типовая семантика:\n1. Расположение, помещение объекта\n2. Изготовление, производство\n3. Создание, учреждение"
    p.font.size = Pt(16)

    # === СЛАЙД 6: Методология ===
    slide = prs.slides.add_slide(slide_layout)

    title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(9), Inches(0.6))
    title_frame = title_box.text_frame
    title_frame.text = "Методы исследования"
    title_para = title_frame.paragraphs[0]
    title_para.font.size = Pt(28)
    title_para.font.bold = True

    content_box = slide.shapes.add_textbox(Inches(0.5), Inches(1), Inches(9), Inches(4))
    content_frame = content_box.text_frame

    methods = [
        ("1. Сравнительно-исторический анализ", [
            "Этимологическая реконструкция",
            "Фонетические и морфологические соответствия"
        ]),
        ("2. Дистрибутивная семантика (NLP)", [
            "TF-IDF векторизация контекстов",
            "K-Means кластеризация (6 кластеров)",
            "t-SNE + PCA визуализация"
        ]),
        ("3. Ручная категоризация", [
            "Divine (божественное)",
            "Human (человеческое)",
            "Teaching (учение)",
            "Good deeds (добрые дела)",
            "Other (прочее)"
        ])
    ]

    for i, (method, subpoints) in enumerate(methods):
        p = content_frame.paragraphs[0] if i == 0 else content_frame.add_paragraph()
        p.text = method
        p.font.size = Pt(20)
        p.font.bold = True
        p.space_after = Pt(8)

        for subpoint in subpoints:
            p = content_frame.add_paragraph()
            p.text = "• " + subpoint
            p.font.size = Pt(16)
            p.space_after = Pt(4)
            p.level = 1

        p = content_frame.add_paragraph()
        p.space_after = Pt(12)

    p = content_frame.add_paragraph()
    p.text = "Инструменты: Python 3, scikit-learn, matplotlib"
    p.font.size = Pt(16)
    p.font.italic = True

    # === СЛАЙД 7: Готский ===
    slide = prs.slides.add_slide(slide_layout)

    title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(9), Inches(0.6))
    title_frame = title_box.text_frame
    title_frame.text = "Готский τaujan: самый религиозный (IV век)"
    title_para = title_frame.paragraphs[0]
    title_para.font.size = Pt(28)
    title_para.font.bold = True

    content_box = slide.shapes.add_textbox(Inches(0.5), Inches(1), Inches(9), Inches(4.5))
    content_frame = content_box.text_frame

    facts = [
        "Период: IV век (300-400 н.э.)",
        "Корпус: Готская Библия (Wulfila)",
        "Вхождений: 140",
        "Divine контексты: 43 (31%) — САМЫЙ ВЫСОКИЙ!"
    ]

    for fact in facts:
        p = content_frame.paragraphs[0] if len(content_frame.paragraphs) == 1 else content_frame.add_paragraph()
        p.text = fact
        p.font.size = Pt(18)
        p.space_after = Pt(6)

    p = content_frame.add_paragraph()
    p.text = "\nУникальная черта: Метафора дерева (30 контекстов)"
    p.font.size = Pt(20)
    p.font.bold = True
    p.space_after = Pt(6)

    p = content_frame.add_paragraph()
    p.text = '"Доброе дерево творит добрые плоды"'
    p.font.size = Pt(16)
    p.font.italic = True
    p.space_after = Pt(12)

    p = content_frame.add_paragraph()
    p.text = "Ключевые кластеры:"
    p.font.size = Pt(18)
    p.font.bold = True
    p.space_after = Pt(6)

    clusters = ["Divine (божественное) — 43", "Teaching (учение) — 17", "Good deeds (добрые дела) — 12"]

    for cluster in clusters:
        p = content_frame.add_paragraph()
        p.text = "• " + cluster
        p.font.size = Pt(16)
        p.level = 1

    # === СЛАЙД 8: Визуализация готского ===
    slide = prs.slides.add_slide(slide_layout)

    title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(9), Inches(0.6))
    title_frame = title_box.text_frame
    title_frame.text = "Кластеризация готского τaujan"
    title_para = title_frame.paragraphs[0]
    title_para.font.size = Pt(28)
    title_para.font.bold = True

    # Placeholder для изображения
    try:
        img_path = '/Users/nora/src/Starostin/gothic/taujan_full_analysis.png'
        slide.shapes.add_picture(img_path, Inches(0.5), Inches(1), width=Inches(9))
    except:
        placeholder_box = slide.shapes.add_textbox(Inches(1), Inches(2), Inches(8), Inches(2))
        placeholder_frame = placeholder_box.text_frame
        placeholder_frame.text = "[Вставить изображение: gothic/taujan_full_analysis.png]"
        placeholder_para = placeholder_frame.paragraphs[0]
        placeholder_para.font.size = Pt(20)
        placeholder_para.alignment = PP_ALIGN.CENTER

    # === СЛАЙД 9: Древнеанглийский ===
    slide = prs.slides.add_slide(slide_layout)

    title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(9), Inches(0.6))
    title_frame = title_box.text_frame
    title_frame.text = "Древнеанглийский don: светский корпус (VIII-XI вв.)"
    title_para = title_frame.paragraphs[0]
    title_para.font.size = Pt(28)
    title_para.font.bold = True

    content_box = slide.shapes.add_textbox(Inches(0.5), Inches(1), Inches(9), Inches(4.5))
    content_frame = content_box.text_frame

    facts = [
        "Период: VIII-XI века (700-1100)",
        "Корпус: Calgary Corpus (смешанный)",
        "Вхождений: 221",
        "Divine контексты: 29 (13%) — ниже, чем в готском"
    ]

    for fact in facts:
        p = content_frame.paragraphs[0] if len(content_frame.paragraphs) == 1 else content_frame.add_paragraph()
        p.text = fact
        p.font.size = Pt(18)
        p.space_after = Pt(6)

    p = content_frame.add_paragraph()
    p.text = "\nУникальная черта: Сравнительные конструкции (49 контекстов)"
    p.font.size = Pt(20)
    p.font.bold = True
    p.space_after = Pt(6)

    p = content_frame.add_paragraph()
    p.text = '"swa swa swa" — "так же, как"'
    p.font.size = Pt(16)
    p.font.italic = True
    p.space_after = Pt(12)

    p = content_frame.add_paragraph()
    p.text = "Ключевые кластеры:"
    p.font.size = Pt(18)
    p.font.bold = True
    p.space_after = Pt(6)

    clusters = ["Relative clauses (73)", "Similes (49)", "Narrative (38)"]

    for cluster in clusters:
        p = content_frame.add_paragraph()
        p.text = "• " + cluster
        p.font.size = Pt(16)
        p.level = 1

    p = content_frame.add_paragraph()
    p.text = "\nПространственные предлоги: on (на) — 21 контекст"
    p.font.size = Pt(16)

    # === СЛАЙД 10: Визуализация древнеанглийского ===
    slide = prs.slides.add_slide(slide_layout)

    title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(9), Inches(0.6))
    title_frame = title_box.text_frame
    title_frame.text = "Кластеризация древнеанглийского don"
    title_para = title_frame.paragraphs[0]
    title_para.font.size = Pt(28)
    title_para.font.bold = True

    try:
        img_path = '/Users/nora/src/Starostin/old_english/don_full_analysis.png'
        slide.shapes.add_picture(img_path, Inches(0.5), Inches(1), width=Inches(9))
    except:
        placeholder_box = slide.shapes.add_textbox(Inches(1), Inches(2), Inches(8), Inches(2))
        placeholder_frame = placeholder_box.text_frame
        placeholder_frame.text = "[Вставить изображение: old_english/don_full_analysis.png]"
        placeholder_para = placeholder_frame.paragraphs[0]
        placeholder_para.font.size = Pt(20)
        placeholder_para.alignment = PP_ALIGN.CENTER

    # === СЛАЙД 11: Древневерхнемецкий ===
    slide = prs.slides.add_slide(slide_layout)

    title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(9), Inches(0.6))
    title_frame = title_box.text_frame
    title_frame.text = "Древневерхнемецкий tuon: религиозный ренессанс (X-XIII вв.)"
    title_para = title_frame.paragraphs[0]
    title_para.font.size = Pt(28)
    title_para.font.bold = True

    content_box = slide.shapes.add_textbox(Inches(0.5), Inches(1), Inches(9), Inches(4.5))
    content_frame = content_box.text_frame

    facts = [
        "Период: X-XIII века (1050-1350)",
        "Корпус: Indiana MHG (теологические тексты)",
        "Вхождений: 433 — САМЫЙ БОЛЬШОЙ!",
        "Divine контексты: 125 (29%) — вновь высокий!"
    ]

    for fact in facts:
        p = content_frame.paragraphs[0] if len(content_frame.paragraphs) == 1 else content_frame.add_paragraph()
        p.text = fact
        p.font.size = Pt(18)
        p.space_after = Pt(6)

    p = content_frame.add_paragraph()
    p.text = "\nУникальная черта: Перфект getan (108 вхождений, 25%)"
    p.font.size = Pt(20)
    p.font.bold = True
    p.space_after = Pt(6)

    p = content_frame.add_paragraph()
    p.text = '"getan" — совершить, сделать (совершённый вид)'
    p.font.size = Pt(16)
    p.font.italic = True
    p.space_after = Pt(12)

    p = content_frame.add_paragraph()
    p.text = "Ключевые кластеры:"
    p.font.size = Pt(18)
    p.font.bold = True
    p.space_after = Pt(6)

    clusters = ["'wir tuon' (мы делаем) — 71", "'ich tuon' (я делаю) — 56", "getan (совершённый) — 108"]

    for cluster in clusters:
        p = content_frame.add_paragraph()
        p.text = "• " + cluster
        p.font.size = Pt(16)
        p.level = 1

    # === СЛАЙД 12: Визуализация древневерхнемецкого ===
    slide = prs.slides.add_slide(slide_layout)

    title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(9), Inches(0.6))
    title_frame = title_box.text_frame
    title_frame.text = "Кластеризация древневерхнемецкого tuon"
    title_para = title_frame.paragraphs[0]
    title_para.font.size = Pt(28)
    title_para.font.bold = True

    try:
        img_path = '/Users/nora/src/Starostin/middle_high_german/tuon_analysis.png'
        slide.shapes.add_picture(img_path, Inches(0.5), Inches(1), width=Inches(9))
    except:
        placeholder_box = slide.shapes.add_textbox(Inches(1), Inches(2), Inches(8), Inches(2))
        placeholder_frame = placeholder_box.text_frame
        placeholder_frame.text = "[Вставить изображение: middle_high_german/tuon_analysis.png]"
        placeholder_para = placeholder_frame.paragraphs[0]
        placeholder_para.font.size = Pt(20)
        placeholder_para.alignment = PP_ALIGN.CENTER

    # === СЛАЙД 13: Древнескандинавский — введение ===
    slide = prs.slides.add_slide(slide_layout)

    title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(9), Inches(0.6))
    title_frame = title_box.text_frame
    title_frame.text = "Древнескандинавский: уникальный случай двух глаголов!"
    title_para = title_frame.paragraphs[0]
    title_para.font.size = Pt(28)
    title_para.font.bold = True

    content_box = slide.shapes.add_textbox(Inches(0.5), Inches(1), Inches(9), Inches(4.5))
    content_frame = content_box.text_frame

    facts = [
        "Период: XI-XIV века (1100-1400)",
        "Корпус: Old Norse Parsed (саги, законы)",
        "ДВА глагола:",
        "  1. dœ (от *dōjan*) — 22 вхождения",
        "  2. gera (от *garwijan*) — 904 вхождения!"
    ]

    for fact in facts:
        p = content_frame.paragraphs[0] if len(content_frame.paragraphs) == 1 else content_frame.add_paragraph()
        p.text = fact
        p.font.size = Pt(18) if "ДВА" not in fact else Pt(20)
        if "ДВА" in fact:
            p.font.bold = True
        p.space_after = Pt(6)
        if fact.startswith("  "):
            p.level = 1

    p = content_frame.add_paragraph()
    p.text = "\nУникальная черта: Лексикальная дифференциация"
    p.font.size = Pt(20)
    p.font.bold = True
    p.space_after = Pt(6)

    p = content_frame.add_paragraph()
    p.text = "dœ = архаический, редко используемый (13.6% divine)"
    p.font.size = Pt(16)
    p.space_after = Pt(4)

    p = content_frame.add_paragraph()
    p.text = "gera = продуктивный, частотный (10.3% divine, 11.9% command)"
    p.font.size = Pt(16)
    p.space_after = Pt(12)

    p = content_frame.add_paragraph()
    p.text = "Divine контексты: самый низкий процент (7-13%)"
    p.font.size = Pt(16)
    p.space_after = Pt(4)

    p = content_frame.add_paragraph()
    p.text = "Тип текстов: Светские саги (Egils saga, Sturlunga)"
    p.font.size = Pt(16)

    # === СЛАЙДЫ 14-15: Визуализации dœ и gera ===
    for slide_num, (title, img_path, placeholder) in enumerate([
        ("Кластеризация dœ (*dōjan*)", '/Users/nora/src/Starostin/old_norse/doe_separate_analysis.png',
         "[Вставить: old_norse/doe_separate_analysis.png]"),
        ("Кластеризация gera (*garwijan*)", '/Users/nora/src/Starostin/old_norse/gera_analysis.png',
         "[Вставить: old_norse/gera_analysis.png]")
    ], 1):

        slide = prs.slides.add_slide(slide_layout)

        title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(9), Inches(0.6))
        title_frame = title_box.text_frame
        title_frame.text = title
        title_para = title_frame.paragraphs[0]
        title_para.font.size = Pt(28)
        title_para.font.bold = True

        try:
            slide.shapes.add_picture(img_path, Inches(0.5), Inches(1), width=Inches(9))
        except:
            placeholder_box = slide.shapes.add_textbox(Inches(1), Inches(2), Inches(8), Inches(2))
            placeholder_frame = placeholder_box.text_frame
            placeholder_frame.text = placeholder
            placeholder_para = placeholder_frame.paragraphs[0]
            placeholder_para.font.size = Pt(20)
            placeholder_para.alignment = PP_ALIGN.CENTER

    # === СЛАЙД 16: Сравнительная таблица ===
    slide = prs.slides.add_slide(slide_layout)

    title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(9), Inches(0.6))
    title_frame = title_box.text_frame
    title_frame.text = "Диахроническая эволюция (IV→XIV вв.)"
    title_para = title_frame.paragraphs[0]
    title_para.font.size = Pt(28)
    title_para.font.bold = True

    table_box = slide.shapes.add_textbox(Inches(0.5), Inches(1), Inches(9), Inches(4))
    table_frame = table_box.text_frame

    header = "Параметр | Готский (IV) | Др.-англ. | MHG | Др.-исл.\n"
    line1 = "Вхождений | 140 | 221 | 433 | 22+904\n"
    line2 = "Divine % | 31% | 13% | 29% | 7-13%\n"
    line3 = "Главная форма | taujiþ | dyde | tuon/getan | dyr/gera\n"
    line4 = "Уникальное | Метафора | Сравнения | getan | Два глагола!\n"
    line5 = "Тип текстов | Религиоз. | Смеш. | Религиоз. | Светский\n"

    p = table_frame.paragraphs[0]
    p.text = header + line1 + line2 + line3 + line4 + line5
    p.font.size = Pt(14)
    p.font.name = 'Courier New'

    summary_box = slide.shapes.add_textbox(Inches(0.5), Inches(5), Inches(9), Inches(0.5))
    summary_frame = summary_box.text_frame
    summary_frame.text = "Тренд: Секуляризация со временем (31% → 7%) с флуктуациями"
    summary_para = summary_frame.paragraphs[0]
    summary_para.font.size = Pt(18)
    summary_para.font.bold = True

    # === СЛАЙД 17: Главные выводы ===
    slide = prs.slides.add_slide(slide_layout)

    title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(9), Inches(0.6))
    title_frame = title_box.text_frame
    title_frame.text = "Главные выводы"
    title_para = title_frame.paragraphs[0]
    title_para.font.size = Pt(28)
    title_para.font.bold = True

    content_box = slide.shapes.add_textbox(Inches(0.5), Inches(1), Inches(9), Inches(4.5))
    content_frame = content_box.text_frame

    conclusions = [
        ("1. Диахроническая эволюция", "От религиозного к светскому: 31% → 7% (IV→XIV вв.)"),
        ("2. Универсальность", "Контекстуальная зависимость, широкая полисемия, адаптивность, стабильность"),
        ("3. Уникальные особенности", "Готский: метафора | Др.-англ.: сравнения | MHG: getan | Др.-исл.: два глагола"),
        ("4. Лексикальная дифференциация", "Др.-исландский: dœ (исчез) vs gera (выжил)")
    ]

    for title, text in conclusions:
        p = content_frame.paragraphs[0] if len([p for p in content_frame.paragraphs if p.text]) == 0 else content_frame.add_paragraph()
        p.text = title
        p.font.size = Pt(18)
        p.font.bold = True
        p.space_after = Pt(4)

        p = content_frame.add_paragraph()
        p.text = text
        p.font.size = Pt(14)
        p.space_after = Pt(10)
        p.level = 1

    # === СЛАЙД 18: Заключение ===
    slide = prs.slides.add_slide(slide_layout)

    title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(9), Inches(0.6))
    title_frame = title_box.text_frame
    title_frame.text = "Заключение и перспективы"
    title_para = title_frame.paragraphs[0]
    title_para.font.size = Pt(28)
    title_para.font.bold = True

    content_box = slide.shapes.add_textbox(Inches(0.5), Inches(1), Inches(9), Inches(4))
    content_frame = content_box.text_frame

    p = content_frame.paragraphs[0]
    p.text = "Достижения:"
    p.font.size = Pt(20)
    p.font.bold = True
    p.space_after = Pt(6)

    achievements = [
        "✓ Унифицированная методика для 4 языков",
        "✓ 1739 analysed occurrences",
        "✓ 24 визуализации (PCA, t-SNE)",
        "✓ Диахрония: 800 лет (300–1400 н.э.)"
    ]

    for achievement in achievements:
        p = content_frame.add_paragraph()
        p.text = achievement
        p.font.size = Pt(16)
        p.space_after = Pt(4)
        p.level = 1

    p = content_frame.add_paragraph()
    p.text = "\nВклад в науку:"
    p.font.size = Pt(20)
    p.font.bold = True
    p.space_after = Pt(6)

    contributions = [
        "• Демонстрация возможностей NLP для исторической лингвистики",
        "• Визуализация семантических траекторий",
        "• Эмпирическая база для сравнительных исследований"
    ]

    for contribution in contributions:
        p = content_frame.add_paragraph()
        p.text = contribution
        p.font.size = Pt(14)
        p.space_after = Pt(4)
        p.level = 1

    p = content_frame.add_paragraph()
    p.text = "\nПерспективы: расширить корпус, включить другие языки, создать базу данных"
    p.font.size = Pt(14)
    p.font.italic = True

    # === СЛАЙД 19: Спасибо ===
    slide = prs.slides.add_slide(slide_layout)

    thank_you_box = slide.shapes.add_textbox(Inches(1), Inches(2), Inches(8), Inches(1.5))
    thank_you_frame = thank_you_box.text_frame
    thank_you_frame.text = "Спасибо за внимание!"
    thank_you_para = thank_you_frame.paragraphs[0]
    thank_you_para.font.size = Pt(44)
    thank_you_para.font.bold = True
    thank_you_para.alignment = PP_ALIGN.CENTER

    questions_box = slide.shapes.add_textbox(Inches(1), Inches(3.5), Inches(8), Inches(1))
    questions_frame = questions_box.text_frame
    questions_frame.text = "Буду рад ответить на ваши вопросы"
    questions_para = questions_frame.paragraphs[0]
    questions_para.font.size = Pt(24)
    questions_para.alignment = PP_ALIGN.CENTER

    # Сохранение
    prs.save('/Users/nora/src/Starostin/PRESENTATION.pptx')
    print("✓ Презентация создана: PRESENTATION.pptx")
    print(f"  Всего слайдов: {len(prs.slides)}")

if __name__ == '__main__':
    create_presentation()
