# Analysis of Old English *dōn* ( Calgary Corpus)

## Проект: PIE *dhe-* → OE *dōn*

---

## Описание

Компьютерный анализ семантической структуры глагола *dōn* «делать, творить» в древнеанглийском языке (VII–XI вв.) на основе Toronto Corpus of Old English (Calgary).

**Исследовательская задача:** Сравнить развитие PIE *dhe-* в древнеанглийском и древневерхненемецком для подтверждения архаичности широкозначности.

---

## Структура директории

```
calgary_analysis/
├── README.md                      # Этот файл
├── results.md                     # Полный анализ OE *dōn*
├── comparison.md                  # Сравнение OE vs OHG
├── don_hdbscan_analysis.py        # Скрипт анализа
├── don_hdbscan_umap.png           # Визуализация
└── don_hdbscan_results.txt        # Сырые данные
```

---

## Основные результаты

### Количественные показатели

| Показатель | Значение |
|-----------|----------|
| **Период** | VII–XI вв. |
| **Корпус** | Toronto Corpus of Old English (Calgary) |
| **Всего вхождений** | 226 |
| **Словоформ** | ~5 (don, de, deð, dyde, gedon) |
| **Кластеров** | 2 (низкого качества) |
| **Шумовых точек** | 173 (76,6%) |
| **Silhouette score** | ~0,28 (плохой) |

### Семантическая структура

**Кластер 0 (10,2%):** Perfektive-Divine Action
- *gedon* «совершенное действие Бога»

**Кластер 1 (13,3%):** Prayer-Supplication
- *dō me halne* «сделай меня целым»

**Переходная зона (76,6%):**
- Континуум значений (без четких границ)

---

## Сравнение: OE vs OHG

| Показатель | OE *dōn* | OHG *tuōn* |
|-----------|----------|-----------|
| **Шум** | 76,6% | 48,4% |
| **Сакральная сфера** | Молитвы | Покаяние |
| **Тип** | Performatives | Коллокации |
| **Вывод** | **Шире по семантике** | Более структурирован |

---

## Главный вывод

✓ **Широкозначность — архаичная черта!**

**Аргументы:**
1. 76,6% шума в OE (VII–XI вв.)
2. 48,4% шума в OHG (IX в.)
3. Оба языка показывают континуум значений

**Интерпретация:** Широкозначность проявляется **уже в ранний период**, что подтверждает:
- Либо **archaic наследие** от PIE *dhe-*
- Либо **раннее расширение** в прогерманский период

---

## Сакральная специализация

### Old English: Performatives in Prayer

```
dō me halne = "сделай меня целым" (молитва)
dō mē þīne mildheortnesse = "сделай мне милость"
```

### Old High German: Collocations

```
riuwa tuōn = "творить покаяние" (коллокация)
```

**Различие:**
- OE: Direct address (императивы в молитвах)
- OHG: Lexical compound (существительное + глагол)

---

## Методология

### Pipeline

```
Calgary Corpus (90 текстов)
    ↓
Извлечение *dōn* (226 контекстов, окно ±5)
    ↓
TF-IDF векторизация (226 × 3097)
    ↓
HDBSCAN кластеризация
    ↓
UMAP визуализация
```

### Parameters

- **Context window:** ±5 tokens
- **TF-IDF:** unigrams + bigrams
- **HDBSCAN:** min_cluster_size=23
- **UMAP:** n_neighbors=15

---

## Corpus Description

**Toronto Corpus of Old English (Calgary)**

| Параметр | Значение |
|----------|----------|
| **Название** | Toronto Corpus of Old English |
| **Объём** | ~90 текстов |
| **Язык** | Древнеанглийский (Old English) |
| **Диалекты** | West Saxon, Anglian, Kentish |
| **Период** | VII–XI вв. |
| **Жанры** | Поэзия, проза, религиозные тексты |

**Основные тексты:**
- *Beowulf* (эпическая поэма)
- *The Dream of the Rood* (религиозная поэма)
- *Caedmon's Hymn* (религиозная поэзия)
- Гомилии Ælfric
- Библейские переводы

---

## Запуск анализа

```bash
cd /Users/nora/src/Starostin
python3 calgary_analysis/don_hdbscan_analysis.py
```

**Ожидаемый результат:**
```
======================================================================
DON ANALYSIS - OLD ENGLISH CALGARY CORPUS
======================================================================

1. Loading corpus...
Found 85 text files
Found 226 occurrences of don/don-related forms

2. Creating TF-IDF matrix...
TF-IDF matrix shape: (226, 3097)

3. HDBSCAN clustering...
Number of clusters: 2
Number of noise points: 173

4. UMAP visualization...
Visualization saved to: don_hdbscan_umap.png

6. Saving results...
Results saved to: don_hdbscan_results.txt
```

---

## Сравнительная таблица германских языков

| Язык | Глагол | Статус | Сакральная сфера |
|------|--------|--------|------------------|
| **Готский** (IV в.) | — | ✗ Утрачен | — |
| **Древнескандинавский** | — | ✗ Утрачен | — |
| **Древнеанглийский** | *dōn* | ✓ Сохранен | ✓ Молитвы (performatives) |
| **Древневерхненемецкий** | *tuōn* | ✓ Сохранен | ✓ Покаяние (*riuwa*) |

---

## Библиография

**Корпус:**
- Toronto Corpus of Old English (Calgary). University of Toronto.

**Литература:**
- Kroonen, G. (2013). *Etymological Dictionary of Proto-Germanic*. Brill.
- Bosworth, J., & Toller, T. N. (1898). *An Anglo-Saxon Dictionary*. Oxford.
- Mitchell, B. (1985). *Old English Syntax*. Oxford University Press.

---

**Дата:** Март 2026
**Корпус:** Calgary Corpus of Old English
**Анализ:** *dōn* (226 occurrences)
