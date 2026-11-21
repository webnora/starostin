#!/usr/bin/env python3
"""
Анализ глагола DON в Calgary Corpus (Древнеанглийский)
Все формы: don, do, dyde, dydon, dydest, deð, doð
Методы: HDBSCAN кластеризация + UMAP визуализация
Согласно требованиям из prompt.md
"""

import re
import os
import random
from collections import Counter, defaultdict
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import umap
import hdbscan
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Установка seed для воспроизводимости
random.seed(42)
np.random.seed(42)

def load_calgary_corpus(directory):
    """Загрузка всех файлов из Calgary Corpus"""
    print("📖 Загрузка Calgary Corpus...")

    all_texts = []
    files = [f for f in os.listdir(directory) if f.endswith('.txt') and not f.startswith('.')]
    print(f"  Найдено {len(files)} файлов")

    for filename in sorted(files):
        filepath = os.path.join(directory, filename)
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
                all_texts.append(text)
        except Exception as e:
            print(f"  Ошибка в {filename}: {e}")
            continue

    combined_text = '\n\n'.join(all_texts)
    print(f"✓ Загружено {len(all_texts)} файлов")
    print(f"  Всего символов: {len(combined_text)}")
    return combined_text

def extract_don_contexts(text, window=15):
    """Извлечение контекстов ВСЕХ форм глагола don"""

    # Все формы глагола don в древнеанглийском
    don_patterns = [
        r'\bdon\b',           # don (инфинитив)
        r'\bdo\b',            # do (императив/настоящее)
        r'\bdyde\b',          # dyde (прошедшее 3SG)
        r'\bdydon\b',         # dydon (прошедшее PL)
        r'\bdydest\b',        # dydest (прошедшее 2SG)
        r'\bdeð\b',           # deð (настоящее 3SG)
        r'\bdoð\b',           # doð (настоящее 3SG, альт.)
        r'\bdidede\b',        # didede (альт. форма)
        r'\bdidon\b',         # didon (альт. форма)
    ]

    # Разбиваем текст на предложения по #
    sentences = text.split('#')

    contexts = []
    sent_num = 0

    combined_pattern = '|'.join(don_patterns)

    for sentence in sentences:
        sent_num += 1
        sentence = sentence.strip()
        if len(sentence) < 10:
            continue

        # Разбиваем на слова
        words = re.findall(r'[a-zA-Zàáæðéèíóòúýþÿ-]+', sentence)

        for word_idx, word in enumerate(words):
            clean_word = word.lower()

            if re.match(combined_pattern, clean_word):
                form = clean_word

                before_words = [w.lower() for w in words[:word_idx]]
                after_words = [w.lower() for w in words[word_idx+1:]]

                contexts.append({
                    'form': form,
                    'before': before_words[-window:],
                    'after': after_words[:window],
                    'sentence': sentence,
                    'sent_num': sent_num
                })
                break  # Только первое вхождение don в предложении

    print(f"✓ Найдено {len(contexts)} контекстов глагола don (все формы)")
    return contexts

def analyze_clusters(contexts, cluster_labels, feature_names, vectorizer):
    """Детальный анализ каждого кластера"""

    print("\n" + "="*70)
    print("🔍 ДЕТАЛЬНЫЙ АНАЛИЗ КЛАСТЕРОВ")
    print("="*70)

    # Игнорируем шум (label -1)
    unique_labels = set(cluster_labels)
    if -1 in unique_labels:
        unique_labels.remove(-1)
        noise_count = sum(1 for label in cluster_labels if label == -1)
        print(f"\n🔊 Шум (не классифицировано): {noise_count} контекстов")

    print(f"\n📊 Найдено кластеров: {len(unique_labels)}")

    cluster_info = {}

    for cluster_id in sorted(unique_labels):
        print(f"\n{'='*70}")
        print(f"КЛАСТЕР {cluster_id}")
        print(f"{'='*70}")

        # Находим индексы контекстов в этом кластере
        cluster_indices = [i for i, label in enumerate(cluster_labels) if label == cluster_id]
        cluster_contexts = [contexts[i] for i in cluster_indices]

        # 1. Размер кластера
        size = len(cluster_contexts)
        print(f"\n1. РАЗМЕР КЛАСТЕРА: {size} контекстов ({size/len(contexts)*100:.1f}%)")

        # 2. Топ-20 слов по среднему TF-IDF
        cluster_tfidf = vectorizer.transform([' '.join(ctx['before'][-10:] + ctx['after'][:10]) for ctx in cluster_contexts])

        # Средний TF-IDF по кластеру
        mean_tfidf = np.mean(cluster_tfidf.toarray(), axis=0)

        # Топ-20 слов
        top20_idx = mean_tfidf.argsort()[-20:][::-1]
        top20_words = [(feature_names[i], mean_tfidf[i]) for i in top20_idx]

        print(f"\n2. ТОП-20 СЛОВ ПО СРЕДНЕМУ TF-IDF:")
        for rank, (word, score) in enumerate(top20_words, 1):
            print(f"   {rank:2d}. {word:20s} : {score:.4f}")

        # 3. 10 случайных примеров
        sample_size = min(10, len(cluster_contexts))
        sampled = random.sample(cluster_contexts, sample_size)

        print(f"\n3. {sample_size} СЛУЧАЙНЫХ ПРИМЕРОВ КОНТЕКСТОВ:")
        for i, ctx in enumerate(sampled, 1):
            before = ' '.join(ctx['before'][-5:])
            after = ' '.join(ctx['after'][:5:])
            print(f"   [{i}] ...{before} [{ctx['form']}] {after}...")

        cluster_info[cluster_id] = {
            'size': size,
            'top_words': top20_words,
            'contexts': cluster_contexts
        }

    # Если есть шум
    if -1 in set(cluster_labels):
        print(f"\n{'='*70}")
        print("ШУМОВЫЕ КОНТЕКСТЫ (не попали в кластеры):")
        noise_indices = [i for i, label in enumerate(cluster_labels) if label == -1]
        print(f"Всего: {len(noise_indices)}")

        if len(noise_indices) > 0:
            sample = min(10, len(noise_indices))
            sampled = [contexts[i] for i in random.sample(noise_indices, sample)]
            print(f"\n{sample} случайных примеров:")
            for i, ctx in enumerate(sampled, 1):
                before = ' '.join(ctx['before'][-5:])
                after = ' '.join(ctx['after'][:5:])
                print(f"   [{i}] ...{before} [{ctx['form']}] {after}...")

    return cluster_info

def visualize_umap(features, clusters, cluster_info, contexts):
    """Визуализация UMAP"""

    print("\n🎨 Вычисляю UMAP проекцию...")

    # UMAP с параметрами
    reducer = umap.UMAP(
        n_neighbors=min(15, len(contexts) - 1),
        min_dist=0.1,
        n_components=2,
        metric='euclidean',
        random_state=42
    )

    embedding = reducer.fit_transform(features)

    # Создаём фигуру
    fig = plt.figure(figsize=(20, 12))

    # 1. UMAP по кластерам
    ax1 = plt.subplot(2, 3, 1)

    unique_labels = set(clusters)
    if -1 in unique_labels:
        unique_labels.remove(-1)

    # Цвета для кластеров
    n_clusters = len(unique_labels)
    colors = plt.cm.tab20(np.linspace(0, 1, max(21, n_clusters + 1)))

    # Рисуем точки
    for label in unique_labels:
        mask = clusters == label
        ax1.scatter(embedding[mask, 0], embedding[mask, 1],
                   c=[colors[label]], s=80, alpha=0.7,
                   edgecolors='black', linewidth=0.5, label=f'Кластер {label}')

    # Шум
    noise_mask = clusters == -1
    if noise_mask.any():
        ax1.scatter(embedding[noise_mask, 0], embedding[noise_mask, 1],
                   c='lightgray', s=50, alpha=0.5, label='Шум')

    ax1.set_title('UMAP: Кластеры don (HDBSCAN) - Calgary (все формы)', fontweight='bold', fontsize=11)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)

    # 2. Распределение размеров кластеров
    ax2 = plt.subplot(2, 3, 2)

    cluster_sizes = [cluster_info[c]['size'] for c in sorted(unique_labels)]
    cluster_ids = list(sorted(unique_labels))

    bars = ax2.bar(cluster_ids, cluster_sizes, color=[colors[i] for i in cluster_ids],
                   edgecolor='black')
    ax2.set_title('Размеры кластеров', fontweight='bold', fontsize=12)
    ax2.set_xlabel('Кластер')
    ax2.set_ylabel('Количество контекстов')
    ax2.grid(True, alpha=0.3, axis='y')

    # Добавляем значения над барами
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=10)

    # 3. Частотность форм
    ax3 = plt.subplot(2, 3, 3)

    form_dist = Counter([ctx['form'] for ctx in contexts])
    forms = list(form_dist.keys())
    counts = list(form_dist.values())

    ax3.barh(range(len(forms)), counts, color='steelblue', edgecolor='black')
    ax3.set_yticks(range(len(forms)))
    ax3.set_yticklabels(forms)
    ax3.set_title('Частотность форм don', fontweight='bold', fontsize=12)
    ax3.set_xlabel('Количество')
    ax3.grid(True, alpha=0.3, axis='x')

    # 4. Статистика кластеризации
    ax4 = plt.subplot(2, 3, 4)
    ax4.axis('off')

    stats_text = "СТАТИСТИКА КЛАСТЕРИЗАЦИИ:\n\n"
    stats_text += f"Всего контекстов: {len(contexts)}\n"
    stats_text += f"Найдено кластеров: {len(unique_labels)}\n"
    noise_count = sum(1 for c in clusters if c == -1)
    stats_text += f"Шум: {noise_count} ({noise_count/len(contexts)*100:.1f}%)\n\n"

    stats_text += "РАЗМЕРЫ КЛАСТЕРОВ:\n"
    for cluster_id in sorted(unique_labels):
        size = cluster_info[cluster_id]['size']
        pct = size / len(contexts) * 100
        stats_text += f"  Кластер {cluster_id}: {size} ({pct:.1f}%)\n"

    stats_text += f"\nФОРМЫ DON (топ-15):\n"
    for form, count in form_dist.most_common(15):
        stats_text += f"  {form}: {count}\n"

    ax4.text(0, 1, stats_text, fontsize=9, family='monospace',
            verticalalignment='top', transform=ax4.transAxes)

    # 5. Топ слова для каждого кластера
    ax5 = plt.subplot(2, 3, 5)
    ax5.axis('off')

    top_words_text = "ТОП-10 СЛОВ ДЛЯ КАЖДОГО КЛАСТЕРА:\n\n"

    for cluster_id in sorted(unique_labels):
        top_words = cluster_info[cluster_id]['top_words'][:10]
        top_words_text += f"Кластер {cluster_id} ({cluster_info[cluster_id]['size']}):\n"
        for rank, (word, score) in enumerate(top_words, 1):
            top_words_text += f"  {rank}. {word:15s} ({score:.3f})\n"
        top_words_text += "\n"

    ax5.text(0, 1, top_words_text, fontsize=7, family='monospace',
            verticalalignment='top', transform=ax5.transAxes)

    # 6. Распределение по категориям
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')

    # Ручная категоризация
    categories = {
        'divine': ['dryhten', 'god', 'halig', 'heofon', ' crist', 'engel', 'heaven'],
        'teaching': ['lar', 'secgan', 'spelling', 'bodian', 'cyðan', ' læran'],
        'command': ['biddan', 'bebeodan', 'hatan', 'scyran', 'swarian'],
        'good_deeds': ['riht', 'soð', 'fremian', 'gife', 'mildheortnes'],
        'evil_deeds': ['yfel', 'facen', 'swica', 'leas', 'unriht'],
        'battle': ['siger', 'winn', 'guð', 'wæpen', 'feoht', 'here'],
        'comparison': ['swa', 'swilce', 'swa þe']
    }

    cat_counts = {}
    for cat, markers in categories.items():
        count = 0
        for ctx in contexts:
            text = ' '.join(ctx['before'] + ctx['after']).lower()
            if any(marker in text for marker in markers):
                count += 1
        cat_counts[cat] = count

    cat_text = "РУЧНАЯ КАТЕГОРИЗАЦИЯ:\n\n"
    for cat, count in sorted(cat_counts.items(), key=lambda x: x[1], reverse=True):
        cat_text += f"{cat:15s}: {count}\n"

    cat_text += f"\nНе категоризовано: {len(contexts) - sum(cat_counts.values())}"

    ax6.text(0, 1, cat_text, fontsize=9, family='monospace',
            verticalalignment='top', transform=ax6.transAxes)

    plt.tight_layout()
    plt.savefig('/Users/nora/src/Starostin/old_english/don_full_calgary_umap.png', dpi=300, bbox_inches='tight')
    print("✓ Визуализация сохранена: don_full_calgary_umap.png")

    plt.show()

def save_results(contexts, cluster_info, clusters):
    """Сохранение результатов"""

    print("\n💾 Сохранение результатов...")

    with open('/Users/nora/src/Starostin/old_english/don_full_calgary_results.txt', 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("АНАЛИЗ ГЛАГОЛА DON (Calgary Corpus, ВСЕ ФОРМЫ)\n")
        f.write("Методы: HDBSCAN + UMAP\n")
        f.write("="*70 + "\n\n")

        # Формы
        f.write("ЧАСТОТНОСТЬ ФОРМ:\n")
        form_dist = Counter([ctx['form'] for ctx in contexts])
        for form, count in form_dist.most_common():
            f.write(f"  {form:15s}: {count}\n")

        # Кластеры
        unique_labels = set(clusters)
        if -1 in unique_labels:
            unique_labels.remove(-1)

        f.write(f"\n\nНАЙДЕНО КЛАСТЕРОВ: {len(unique_labels)}\n")

        for cluster_id in sorted(unique_labels):
            f.write(f"\n{'='*70}\n")
            f.write(f"КЛАСТЕР {cluster_id}\n")
            f.write(f"{'='*70}\n\n")

            cluster_contexts = [ctx for i, ctx in enumerate(contexts) if clusters[i] == cluster_id]

            f.write(f"Размер: {len(cluster_contexts)} ({len(cluster_contexts)/len(contexts)*100:.1f}%)\n\n")

            # Топ-20 слов
            f.write("ТОП-20 СЛОВ (TF-IDF):\n")
            top_words = cluster_info[cluster_id]['top_words']
            for rank, (word, score) in enumerate(top_words, 1):
                f.write(f"  {rank:2d}. {word:20s} : {score:.4f}\n")

            # Все контексты
            f.write(f"\nВСЕ КОНТЕКСТЫ КЛАСТЕРА:\n")
            for ctx in cluster_contexts:
                before = ' '.join(ctx['before'][-5:])
                after = ' '.join(ctx['after'][:5:])
                f.write(f"  [{ctx['sent_num']:4d}] ...{before} [{ctx['form']}] {after}...\n")

        # Шум
        if -1 in set(clusters):
            noise_count = sum(1 for c in clusters if c == -1)
            f.write(f"\n\n{'='*70}\n")
            f.write("ШУМ (не классифицировано)\n")
            f.write(f"{'='*70}\n\n")
            f.write(f"Всего: {noise_count} контекстов\n\n")

            noise_contexts = [ctx for i, ctx in enumerate(contexts) if clusters[i] == -1]
            f.write("Случайные примеры:\n")
            for ctx in random.sample(noise_contexts, min(30, len(noise_contexts))):
                before = ' '.join(ctx['before'][-5:])
                after = ' '.join(ctx['after'][:5:])
                f.write(f"  ...{before} [{ctx['form']}] {after}...\n")

    print("✓ Результаты сохранены: don_full_calgary_results.txt")

def main():
    print("="*70)
    print("🔬 АНАЛИЗ DON (Calgary, ВСЕ ФОРМЫ) - HDBSCAN + UMAP")
    print("="*70)

    # 1. Загружаем корпус
    text = load_calgary_corpus('/Users/nora/src/Starostin/calgary')

    if len(text) < 1000:
        print("❌ Корпус слишком мал")
        return

    # 2. Извлекаем контексты don (все формы)
    contexts = extract_don_contexts(text, window=15)

    if len(contexts) < 10:
        print("❌ Слишком мало контекстов")
        return

    # Частота форм
    form_dist = Counter([ctx['form'] for ctx in contexts])
    print(f"\n📊 Частотность форм don (все формы):")
    for form, count in form_dist.most_common(15):
        print(f"  {form:15s}: {count}")

    # 3. TF-IDF векторизация
    print("\n🔢 TF-IDF векторизация...")
    corpus = [' '.join(ctx['before'][-10:] + ctx['after'][:10]) for ctx in contexts]

    vectorizer = TfidfVectorizer(
        max_features=150,
        min_df=2,
        token_pattern=r"[a-zA-Zàáæðéèíóòúýþ]+"
    )
    features = vectorizer.fit_transform(corpus).toarray()
    feature_names = vectorizer.get_feature_names_out()

    print(f"   Признаков: {features.shape}")

    # 4. HDBSCAN кластеризация
    print("\n🎯 HDBSCAN кластеризация...")

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=8,        # Уменьшен для нахождения мелких кластеров
        min_samples=2,             # Уменьшен
        metric='euclidean',
        cluster_selection_method='eom'
    )

    clusters = clusterer.fit_predict(features)

    unique_clusters = set(clusters)
    if -1 in unique_clusters:
        unique_clusters.remove(-1)

    noise_count = sum(1 for c in clusters if c == -1)
    print(f"   Найдено кластеров: {len(unique_clusters)}")
    print(f"   Шум: {noise_count} ({noise_count/len(clusters)*100:.1f}%)")

    # Распределение по кластерам
    cluster_dist = Counter(clusters)
    for cluster_id in sorted(unique_clusters):
        print(f"   Кластер {cluster_id}: {cluster_dist[cluster_id]} контекстов")

    # 5. Анализ кластеров
    cluster_info = analyze_clusters(contexts, clusters, feature_names, vectorizer)

    # 6. Визуализация UMAP
    visualize_umap(features, clusters, cluster_info, contexts)

    # 7. Сохранение результатов
    save_results(contexts, cluster_info, clusters)

    print("\n" + "="*70)
    print("✅ АНАЛИЗ ЗАВЕРШЁН!")
    print("="*70)
    print("\nСозданные файлы:")
    print("  • don_full_calgary_umap.png - визуализация UMAP")
    print("  • don_full_calgary_results.txt - результаты с топ-словами")

if __name__ == '__main__':
    main()
