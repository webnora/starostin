#!/usr/bin/env python3
"""
Анализ векторного пространства глагола taujan в Готской Библии
"""

import re
from collections import Counter, defaultdict
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Кодировка для вывода
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans']

def parse_gothic_corpus(filename):
    """Парсит корпус, возвращает список токенов с контекстами"""
    print("📖 Читаю корпус...")

    with open(filename, 'r', encoding='utf-8') as f:
        text = f.read()

    # Разбиваем на строки, пропускаем комментарии
    lines = []
    for line in text.split('\n'):
        line = line.strip()
        # Пропускаем пустые строки и комментарии
        if line and not line.startswith('#'):
            lines.append(line)

    print(f"✓ Загружено {len(lines)} строк")

    return lines

def extract_taujan_contexts(lines, window_size=5):
    """Извлекает контексты для всех форм taujan"""

    # Все формы taujan (презенс, претерит, пассив, абзаут)
    taujan_patterns = [
        r'\btaujan\b', r'\btaujis\b', r'\btauji[þþ]\b', r'\btauja[iï]d\b',
        r'\btaujai[þþ]\b', r'\btaujand\b', r'\btaujands\b',
        r'\bgataujan\b', r'\bgatauj\b', r'\bgatauji[þþ]\b',
        r'\bgataujands\b', r'\bgataujand\b'
    ]

    combined_pattern = '|'.join(taujan_patterns)

    contexts = []
    total_matches = 0

    for line in lines:
        # Находим все вхождения форм taujan
        for match in re.finditer(combined_pattern, line, re.IGNORECASE):
            total_matches += 1
            form = match.group()

            # Получаем контекст
            start = max(0, match.start() - window_size * 10)
            end = min(len(line), match.end() + window_size * 10)

            context_before = line[start:match.start()].strip()
            context_after = line[match.end():end].strip()

            # Токенизируем контекст
            tokens_before = re.findall(r'\b\w+\b', context_before)
            tokens_after = re.findall(r'\b\w+\b', context_after)

            contexts.append({
                'form': form,
                'full_context': line,
                'before': tokens_before[-window_size:] if tokens_before else [],
                'after': tokens_after[:window_size] if tokens_after else [],
                'line_num': lines.index(line) + 1
            })

    print(f"✓ Найдено {total_matches} вхождений форм taujan")

    return contexts

def build_vocabulary(contexts, min_freq=2):
    """Строит словарь из контекстов"""

    word_freq = Counter()

    for ctx in contexts:
        # Считаем слова в контексте
        for word in ctx['before'] + ctx['after']:
            word_freq[word.lower()] += 1

    # Фильтруем редкие слова
    vocab = [word for word, freq in word_freq.most_common() if freq >= min_freq]

    print(f"✓ Словарь: {len(vocab)} слов (min_freq={min_freq})")
    print(f"  Топ-20 слов: {[w for w, _ in word_freq.most_common(20)]}")

    return vocab, word_freq

def build_cooccurrence_matrix(contexts, vocab):
    """Строит матрицу совстречаемости"""

    word_to_idx = {word: i for i, word in enumerate(vocab)}
    vocab_size = len(vocab)

    # Матрица совстречаемости
    cooccur = np.zeros((vocab_size, vocab_size))

    for ctx in contexts:
        # Все слова в контексте
        context_words = [w.lower() for w in ctx['before'] + ctx['after']]

        # Увеличиваем счетчик для каждой пары
        for i, w1 in enumerate(context_words):
            if w1 in word_to_idx:
                for j, w2 in enumerate(context_words):
                    if i != j and w2 in word_to_idx:
                        cooccur[word_to_idx[w1], word_to_idx[w2]] += 1

    print(f"✓ Матрица совстречаемости: {cooccur.shape}")

    return cooccur, word_to_idx

def compute_ppmi(cooccur):
    """Вычисляет PPMI (Positive Pointwise Mutual Information)"""

    # Всего совстречаемостей
    total = cooccur.sum()

    # Маргинальные вероятности
    row_sums = cooccur.sum(axis=1)
    col_sums = cooccur.sum(axis=0)

    ppmi = np.zeros_like(cooccur)

    for i in range(cooccur.shape[0]):
        for j in range(cooccur.shape[1]):
            if cooccur[i, j] > 0:
                pmi = np.log2((cooccur[i, j] * total) / (row_sums[i] * col_sums[j]))
                ppmi[i, j] = max(0, pmi)

    print(f"✓ PPMI вычислен")

    return ppmi

def visualize_embeddings(ppmi, vocab, word_freq, n_words=50):
    """Визуализирует эмбеддинги"""

    # Берем топ-n слов
    top_words = [w for w, _ in word_freq.most_common(n_words)]

    # Получаем их индексы
    indices = [i for i, w in enumerate(vocab) if w in top_words]

    # Матрица для этих слов
    ppmi_subset = ppmi[np.ix_(indices, indices)]

    # PCA
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(ppmi_subset)

    # Визуализация
    fig, ax = plt.subplots(figsize=(16, 12))

    # Выделяем taujan и связанные слова
    taujan_related = ['taujan', 'tauja', 'taujiþ', 'waurstw', 'gataujan',
                      'waurkjan', 'qen', 'dom', 'þiud', 'frawaurht']

    colors = []
    sizes = []

    for i, word in enumerate(top_words):
        if word in taujan_related or 'tauj' in word:
            colors.append('red')
            sizes.append(300)
        elif word in ['jah', 'iþ', 'ak', 'uhan', 'ni']:
            colors.append('lightgray')
            sizes.append(50)
        else:
            colors.append('steelblue')
            sizes.append(100)

    # Scatter plot
    ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
               c=colors, s=sizes, alpha=0.6, edgecolors='black', linewidth=0.5)

    # Добавляем подписи
    for i, word in enumerate(top_words):
        ax.annotate(word, (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                   fontsize=10, ha='center', va='center',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                            edgecolor='gray', alpha=0.7))

    ax.set_title(f'Векторное пространство слов из контекстов τAUJAN\n(PCA, топ-{n_words} слов)',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('PC1', fontsize=12)
    ax.set_ylabel('PC2', fontsize=12)
    ax.grid(True, alpha=0.3)

    # Легенда
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', label='τaujan и связанные'),
        Patch(facecolor='steelblue', label='Содержательные слова'),
        Patch(facecolor='lightgray', label='Союзы/частицы')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plt.savefig('/Users/nora/src/Starostin/taujan_pca.png', dpi=300, bbox_inches='tight')
    print(f"✓ Визуализация сохранена: taujan_pca.png")

    plt.show()

def find_similar_words(ppmi, vocab, word_freq, target_word, top_n=10):
    """Находит слова, похожие на target_word"""

    if target_word not in vocab:
        print(f"⚠ Слово '{target_word}' не в словаре")
        return []

    word_to_idx = {word: i for i, word in enumerate(vocab)}
    target_idx = word_to_idx[target_word]

    # Косинусное сходство
    target_vector = ppmi[target_idx]

    similarities = []
    for i, word in enumerate(vocab):
        if i != target_idx:
            # Косинусное сходство
            sim = np.dot(target_vector, ppmi[i]) / (np.linalg.norm(target_vector) * np.linalg.norm(ppmi[i]))
            if not np.isnan(sim):
                similarities.append((word, sim))

    # Сортируем
    similarities.sort(key=lambda x: x[1], reverse=True)

    print(f"\n🔍 Слова, похожие на '{target_word}':")
    for word, sim in similarities[:top_n]:
        freq = word_freq[word]
        print(f"  {word:15s} (sim={sim:.3f}, freq={freq})")

    return similarities[:top_n]

def analyze_taujan_forms(contexts):
    """Анализирует разные формы taujan"""

    form_stats = Counter()

    for ctx in contexts:
        form_stats[ctx['form']] += 1

    print(f"\n📊 Частотность форм τaujan:")
    for form, count in form_stats.most_common():
        print(f"  {form:15s}: {count:3d}")

def main():
    print("="*60)
    print("🔬 Анализ векторного пространства τAUJAN")
    print("="*60)

    # 1. Загружаем корпус
    lines = parse_gothic_corpus('/Users/nora/src/Starostin/gotica.txt')

    # 2. Извлекаем контексты taujan
    contexts = extract_taujan_contexts(lines, window_size=5)

    if not contexts:
        print("❌ Не найдено контекстов taujan")
        return

    # 3. Анализируем формы
    analyze_taujan_forms(contexts)

    # 4. Строим словарь
    vocab, word_freq = build_vocabulary(contexts, min_freq=3)

    # 5. Строим матрицу совстречаемости
    cooccur, word_to_idx = build_cooccurrence_matrix(contexts, vocab)

    # 6. Вычисляем PPMI
    ppmi = compute_ppmi(cooccur)

    # 7. Визуализируем
    visualize_embeddings(ppmi, vocab, word_freq, n_words=60)

    # 8. Находим похожие слова
    if 'waurstw' in vocab:
        find_similar_words(ppmi, vocab, word_freq, 'waurstw', top_n=15)

    if 'frawaurht' in vocab:
        find_similar_words(ppmi, vocab, word_freq, 'frawaurht', top_n=15)

    # 9. Примеры контекстов
    print(f"\n📝 Примеры контекстов τaujan:")
    for ctx in contexts[:5]:
        print(f"\n  [{ctx['line_num']}] {ctx['form']}")
        before = ' '.join(ctx['before'][-3:])
        after = ' '.join(ctx['after'][:3])
        print(f"    ... {before} [{ctx['form']}] {after} ...")

    print("\n✓ Анализ завершён!")

if __name__ == '__main__':
    main()
