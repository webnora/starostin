#!/usr/bin/env python3
"""
Глубокий анализ глагола taujan:
1. Семантическая кластеризация по смыслам
2. t-SNE визуализация
3. Анализ контекстов (существительные, прилагательные, дополнения)
"""

import re
from collections import Counter, defaultdict
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings('ignore')

# Кодировка для вывода
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def parse_gothic_corpus(filename):
    """Парсит корпус"""
    print("📖 Читаю корпус...")
    with open(filename, 'r', encoding='utf-8') as f:
        text = f.read()

    lines = []
    for line in text.split('\n'):
        line = line.strip()
        if line and not line.startswith('#'):
            lines.append(line)

    print(f"✓ Загружено {len(lines)} строк")
    return lines

def extract_taujan_contexts(lines, window_size=10):
    """Извлекает расширенные контексты для всех форм taujan"""

    taujan_patterns = [
        r'\btaujan\b', r'\btaujis\b', r'\btauji[þþ]\b', r'\btauja[iï]d\b',
        r'\btaujai[þþ]\b', r'\btaujand\b', r'\btaujands\b',
        r'\bgataujan\b', r'\bgatauj\b', r'\bgatauji[þþ]\b',
        r'\bgataujands\b', r'\bgataujand\b', r'\btaujaid\b'
    ]

    combined_pattern = '|'.join(taujan_patterns)
    contexts = []

    for line_num, line in enumerate(lines, 1):
        for match in re.finditer(combined_pattern, line, re.IGNORECASE):
            form = match.group()

            # Расширенный контекст
            start = max(0, match.start() - 200)
            end = min(len(line), match.end() + 200)

            context_before = line[start:match.start()].strip()
            context_after = line[match.end():end].strip()
            full_line = line

            # Токенизация
            tokens_before = re.findall(r'\b[\w-]+\b', context_before)
            tokens_after = re.findall(r'\b[\w-]+\b', context_after)
            tokens_line = re.findall(r'\b[\w-]+\b', full_line)

            contexts.append({
                'form': form,
                'full_line': full_line,
                'before': tokens_before,
                'after': tokens_after,
                'line_tokens': tokens_line,
                'line_num': line_num
            })

    print(f"✓ Найдено {len(contexts)} вхождений форм taujan")
    return contexts

def categorize_contexts_manual(contexts):
    """Ручная категоризация по контекстуальным маркерам"""

    categories = {
        'divine_action': [],      # Божественное действие (Бог, Иисус)
        'human_action': [],       # Человеческое действие
        'teaching': [],           # Учение/заповеди
        'good_works': [],         # Добрые дела
        'evil_works': [],         # Злые дела/грех
        'creation': [],           # Творение/создание
        'miracles': [],           # Чудеса
        'obedience': [],          # Повиновение/исполнение воли
        'other': []
    }

    # Маркеры для категорий
    markers = {
        'divine_action': ['gud', 'attin', 'frauja', 'Iesus', 'Xristus', 'ahma', 'sunus'],
        'teaching': ['waurd', 'lais', 'witoþ', 'qenþ', 'rodida', 'anabiud'],
        'good_works': ['waila', 'godo', 'armaion', 'gudabaubo', 'libains'],
        'evil_works': ['ubilo', 'frawaurht', 'horin', 'waila', 'iddeig', 'wiþra'],
        'miracles': ['taikne', 'galaubein', 'galaubid', 'macleig', 'usleiþ'],
        'obedience': ['wiljan', 'anabaus', 'skalks', 'fraletan', 'qimi'],
        'creation': ['waurstw', 'akran', 'bagms', 'waurkjan', 'gaskapjan']
    }

    for ctx in contexts:
        text_line = ' '.join(ctx['line_tokens']).lower()
        categorized = False

        for cat, cats_markers in markers.items():
            if any(marker.lower() in text_line for marker in cats_markers):
                categories[cat].append(ctx)
                categorized = True
                break

        if not categorized:
            categories['other'].append(ctx)

    print("\n📊 Распределение по категориям:")
    for cat, ctxs in categories.items():
        print(f"  {cat:20s}: {len(ctxs):3d}")

    return categories

def analyze_pos_patterns(contexts):
    """Анализ частей речи в контекстах"""

    # Готские слова-маркеры (упрощённые шаблоны)
    nouns = defaultdict(Counter)
    adjectives = defaultdict(Counter)
    verbs = defaultdict(Counter)
    prepositions = Counter()

    # Списки готских слов по частям речи (упрощённо)
    noun_suffixes = ['s', 'a', 'o', 'e', 'ja', 'ons', 'ins', 'us']
    adj_suffixes = ['s', 'ba', 'iska', 'aga']
    prep_words = ['in', 'ana', 'fra', 'du', 'us', 'faura', 'bi', 'und', 'miþ', 'uh', 'þanuh']

    for ctx in contexts:
        form = ctx['form']
        all_tokens = ctx['before'][-5:] + ctx['after'][:5]

        for token in all_tokens:
            token_lower = token.lower()

            # Препозиции
            if token_lower in prep_words:
                prepositions[token_lower] += 1

            # Существительные (по контексту taujan)
            # Наиболее частотные слова рядом с taujan
            elif len(token) > 3:
                # Существительные, которые часто являются объектом taujan
                nouns[form][token_lower] += 1

    print("\n🔤 Наиболее частотные слова рядом с формами taujan:")
    for form in sorted(nouns.keys()):
        print(f"\n  {form}:")
        for word, count in nouns[form].most_common(10):
            print(f"    {word:20s}: {count}")

    print(f"\n📍 Препозиции в контекстах taujan:")
    for prep, count in prepositions.most_common():
        print(f"  {prep:10s}: {count}")

    return nouns, prepositions

def extract_semantic_features(contexts):
    """Извлекает семантические признаки для кластеризации"""

    # TF-IDF на контекстах
    corpus = []
    for ctx in contexts:
        # Берём токены вокруг taujan
        tokens = ctx['before'][-8:] + ctx['after'][:8]
        corpus.append(' '.join(tokens))

    vectorizer = TfidfVectorizer(max_features=100, min_df=2)
    tfidf_matrix = vectorizer.fit_transform(corpus)

    print(f"✓ TF-IDF матрица: {tfidf_matrix.shape}")
    print(f"  Особенности (words): {len(vectorizer.get_feature_names_out())}")

    return tfidf_matrix.toarray(), vectorizer

def cluster_contexts(contexts, n_clusters=5):
    """Кластеризация контекстов"""

    # Получаем признаки
    features, vectorizer = extract_semantic_features(contexts)

    # K-means кластеризация
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(features)

    # Agglomerative кластеризация для иерархии
    agg = AgglomerativeClustering(n_clusters=n_clusters)
    agg_clusters = agg.fit_predict(features)

    print(f"\n🎯 K-Means кластеры (n={n_clusters}):")
    cluster_dist = Counter(clusters)
    for i in range(n_clusters):
        print(f"  Кластер {i}: {cluster_dist[i]} контекстов")

    # Анализ ключевых слов для каждого кластера
    feature_names = vectorizer.get_feature_names_out()
    cluster_centers = kmeans.cluster_centers_

    print(f"\n🔑 Ключевые слова для кластеров:")
    for i in range(n_clusters):
        # Топ-10 слов для центра кластера
        top_indices = cluster_centers[i].argsort()[-10:][::-1]
        top_words = [feature_names[idx] for idx in top_indices]
        print(f"\n  Кластер {i}: {', '.join(top_words)}")

    return clusters, features, vectorizer

def visualize_tsne(contexts, features, cluster_labels):
    """Визуализация через t-SNE"""

    # t-SNE
    print("\n🎨 Вычисляю t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    embeddings_2d = tsne.fit_transform(features)

    # Создаём фигуру с несколькими подграфиками
    fig = plt.figure(figsize=(20, 12))

    # 1. t-SNE с кластерами
    ax1 = plt.subplot(2, 3, 1)
    colors = plt.cm.tab10(np.linspace(0, 1, max(cluster_labels) + 1))

    for i in range(len(contexts)):
        ax1.scatter(embeddings_2d[i, 0], embeddings_2d[i, 1],
                   c=[colors[cluster_labels[i]]], s=60, alpha=0.7,
                   edgecolors='black', linewidth=0.5)

    ax1.set_title('t-SNE: Кластеры контекстов taujan', fontsize=12, fontweight='bold')
    ax1.set_xlabel('t-SNE 1')
    ax1.set_ylabel('t-SNE 2')
    ax1.grid(True, alpha=0.3)

    # 2. t-SNE с формами глагола
    ax2 = plt.subplot(2, 3, 2)
    form_colors = {}
    color_idx = 0
    form_to_idx = {}

    for ctx in contexts:
        form = ctx['form']
        if form not in form_to_idx:
            form_to_idx[form] = color_idx
            color_idx += 1

    form_palette = plt.cm.Set3(np.linspace(0, 1, len(form_to_idx)))

    for i, ctx in enumerate(contexts):
        form = ctx['form']
        color = form_palette[form_to_idx[form]]
        ax2.scatter(embeddings_2d[i, 0], embeddings_2d[i, 1],
                   c=[color], s=60, alpha=0.7, edgecolors='black', linewidth=0.5)

    ax2.set_title('t-SNE: По формам глагола', fontsize=12, fontweight='bold')
    ax2.set_xlabel('t-SNE 1')
    ax2.set_ylabel('t-SNE 2')
    ax2.grid(True, alpha=0.3)

    # Легенда для форм
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=form_palette[form_to_idx[form]], label=form)
                      for form in sorted(form_to_idx.keys())]
    ax2.legend(handles=legend_elements, loc='best', fontsize=8)

    # 3. Распределение по кластерам
    ax3 = plt.subplot(2, 3, 3)
    cluster_counts = Counter(cluster_labels)
    ax3.bar(cluster_counts.keys(), cluster_counts.values(),
           color=[colors[i] for i in cluster_counts.keys()], edgecolor='black')
    ax3.set_title('Распределение по кластерам', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Кластер')
    ax3.set_ylabel('Количество')
    ax3.grid(True, alpha=0.3, axis='y')

    # 4. Частотность форм
    ax4 = plt.subplot(2, 3, 4)
    form_dist = Counter([ctx['form'] for ctx in contexts])
    forms = list(form_dist.keys())
    counts = list(form_dist.values())
    ax4.bar(range(len(forms)), counts, color='steelblue', edgecolor='black')
    ax4.set_xticks(range(len(forms)))
    ax4.set_xticklabels(forms, rotation=45, ha='right')
    ax4.set_title('Частотность форм taujan', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Количество')
    ax4.grid(True, alpha=0.3, axis='y')

    # 5. Примеры контекстов для каждого кластера
    ax5 = plt.subplot(2, 3, 5)
    ax5.axis('off')

    example_text = "Примеры контекстов:\n\n"

    for cluster_id in sorted(set(cluster_labels)):
        cluster_contexts = [ctx for i, ctx in enumerate(contexts) if cluster_labels[i] == cluster_id]
        example_text += f"Кластер {cluster_id}:\n"

        # Берём 2 примера
        for ctx in cluster_contexts[:2]:
            before = ' '.join(ctx['before'][-3:])
            after = ' '.join(ctx['after'][:3:])
            example_text += f"  ...{before} [{ctx['form']}] {after}...\n"
        example_text += "\n"

    ax5.text(0, 1, example_text, fontsize=8, family='monospace',
            verticalalignment='top', transform=ax5.transAxes)

    # 6. PCA для сравнения
    ax6 = plt.subplot(2, 3, 6)
    pca = PCA(n_components=2)
    pca_embeddings = pca.fit_transform(features)

    for i in range(len(contexts)):
        ax6.scatter(pca_embeddings[i, 0], pca_embeddings[i, 1],
                   c=[colors[cluster_labels[i]]], s=60, alpha=0.7,
                   edgecolors='black', linewidth=0.5)

    ax6.set_title(f'PCA: Сравнение (variance={pca.explained_variance_ratio_.sum():.2f})',
                 fontsize=12, fontweight='bold')
    ax6.set_xlabel('PC1')
    ax6.set_ylabel('PC2')
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/Users/nora/src/Starostin/taujan_tsne_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Визуализация сохранена: taujan_tsne_analysis.png")

    plt.show()

def analyze_semantic_clusters(contexts, cluster_labels):
    """Детальный анализ семантических кластеров"""

    print("\n" + "="*60)
    print("🔍 ДЕТАЛЬНЫЙ АНАЛИЗ КЛАСТЕРОВ")
    print("="*60)

    for cluster_id in sorted(set(cluster_labels)):
        cluster_contexts = [ctx for i, ctx in enumerate(contexts) if cluster_labels[i] == cluster_id]

        print(f"\n{'='*60}")
        print(f"КЛАСТЕР {cluster_id} ({len(cluster_contexts)} контекстов)")
        print(f"{'='*60}")

        # Формы в этом кластере
        forms = [ctx['form'] for ctx in cluster_contexts]
        form_dist = Counter(forms)
        print(f"\nФормы:")
        for form, count in form_dist.most_common():
            print(f"  {form:15s}: {count}")

        # Все слова в контекстах
        all_words = []
        for ctx in cluster_contexts:
            all_words.extend([w.lower() for w in ctx['before'][-5:] + ctx['after'][:5]])

        word_dist = Counter(all_words)
        print(f"\nКлючевые слова:")
        for word, count in word_dist.most_common(15):
            print(f"  {word:20s}: {count}")

        # Примеры
        print(f"\nПримеры контекстов:")
        for ctx in cluster_contexts[:3]:
            before = ' '.join(ctx['before'][-5:])
            after = ' '.join(ctx['after'][:5])
            print(f"  [{ctx['line_num']}] ...{before} [{ctx['form']}] {after}...")

def find_patterns_in_categories(categories):
    """Находит паттерны в категориях"""

    print("\n" + "="*60)
    print("📂 АНАЛИЗ КАТЕГОРИЙ")
    print("="*60)

    for cat_name, cat_contexts in categories.items():
        if len(cat_contexts) == 0:
            continue

        print(f"\n{'='*60}")
        print(f"{cat_name.upper()} ({len(cat_contexts)} контекстов)")
        print(f"{'='*60}")

        # Формы
        forms = Counter([ctx['form'] for ctx in cat_contexts])
        print(f"\nФормы:")
        for form, count in forms.most_common():
            print(f"  {form:15s}: {count}")

        # Ключевые слова
        all_words = []
        for ctx in cat_contexts:
            all_words.extend([w.lower() for w in ctx['before'][-5:] + ctx['after'][:5]])

        word_dist = Counter(all_words)
        print(f"\nКлючевые слова:")
        for word, count in word_dist.most_common(10):
            print(f"  {word:20s}: {count}")

        # Примеры
        print(f"\nПримеры:")
        for ctx in cat_contexts[:2]:
            before = ' '.join(ctx['before'][-4:])
            after = ' '.join(ctx['after'][:4])
            print(f"  [{ctx['line_num']}] ...{before} [{ctx['form']}] {after}...")

def main():
    print("="*60)
    print("🔬 ГЛУБОКИЙ АНАЛИЗ τAUJAN")
    print("="*60)

    # 1. Загружаем корпус
    lines = parse_gothic_corpus('/Users/nora/src/Starostin/gotica.txt')

    # 2. Извлекаем контексты
    contexts = extract_taujan_contexts(lines, window_size=10)

    # 3. Ручная категоризация по маркерам
    categories = categorize_contexts_manual(contexts)

    # 4. Анализ частей речи
    nouns, preps = analyze_pos_patterns(contexts)

    # 5. Кластеризация
    clusters, features, vectorizer = cluster_contexts(contexts, n_clusters=5)

    # 6. Визуализация t-SNE
    visualize_tsne(contexts, features, clusters)

    # 7. Детальный анализ кластеров
    analyze_semantic_clusters(contexts, clusters)

    # 8. Анализ категорий
    find_patterns_in_categories(categories)

    # Сохраняем результаты в файл
    print("\n💾 Сохраняю результаты в файл...")

    with open('/Users/nora/src/Starostin/taujan_results.txt', 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("РЕЗУЛЬТАТЫ АНАЛИЗА τAUJAN\n")
        f.write("="*60 + "\n\n")

        # Формы
        f.write("ЧАСТОТНОСТЬ ФОРМ:\n")
        forms = Counter([ctx['form'] for ctx in contexts])
        for form, count in forms.most_common():
            f.write(f"  {form:15s}: {count}\n")
        f.write("\n")

        # Кластеры
        f.write("\nКЛАСТЕРЫ:\n")
        for cluster_id in sorted(set(clusters)):
            cluster_contexts = [ctx for i, ctx in enumerate(contexts) if clusters[i] == cluster_id]
            f.write(f"\nКластер {cluster_id} ({len(cluster_contexts)}):\n")

            all_words = []
            for ctx in cluster_contexts:
                all_words.extend([w.lower() for w in ctx['before'][-5:] + ctx['after'][:5]])

            word_dist = Counter(all_words)
            for word, count in word_dist.most_common(10):
                f.write(f"  {word:20s}: {count}\n")

        f.write("\n" + "="*60 + "\n")
        f.write("ВСЕ КОНТЕКСТЫ:\n")
        f.write("="*60 + "\n\n")

        for ctx in contexts:
            before = ' '.join(ctx['before'][-5:])
            after = ' '.join(ctx['after'][:5])
            f.write(f"[{ctx['line_num']}] {ctx['form']:15s} ...{before} [{ctx['form']}] {after}...\n")

    print("✓ Результаты сохранены: taujan_results.txt")

    print("\n" + "="*60)
    print("✅ АНАЛИЗ ЗАВЕРШЁН!")
    print("="*60)
    print("\nСозданные файлы:")
    print("  • taujan_tsne_analysis.png - визуализация")
    print("  • taujan_results.txt - результаты в текстовом виде")

if __name__ == '__main__':
    main()
