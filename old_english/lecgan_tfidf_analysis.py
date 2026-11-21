#!/usr/bin/env python3
"""
Анализ глагола LECGAN (класть, положить) в Calgary Corpus (Древнеанглийский)
Методы: TF-IDF + HDBSCAN + UMAP
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
            continue

    combined_text = '\n\n'.join(all_texts)
    print(f"✓ Загружено {len(all_texts)} файлов")
    return combined_text

def extract_lecgan_contexts(text, window=15):
    """Извлечение контекстов всех форм глагола lecgan (класть, положить)"""

    # Все формы глагола lecgan в древнеанглийском
    # Слабый глагол I класса с аблаутом
    lecgan_patterns = [
        # Инфинитив
        r'\blecgan\b',
        # Настоящее время
        r'\blecg(e|æ)\b',           # 1/3 SG (lecg- > liceg- в некоторых позициях)
        r'\blecgest\b',             # 2 SG
        r'\blecg(e|æ)ð\b',          # 3 SG
        r'\blecgiað\b',             # PL
        r'\blecgia[ðþ]\b',          # PL (альт.)
        r'\bliceg(e|æ)ð\b',         # 3 SG (с палатализацией)
        r'\blicge[st]\b',           # 2 SG (с палатализацией)
        # Прошедшее время (аблаут: ea > eo > ie)
        r'\bleag\b',                # 1/3 SG
        r'\blige\b',                # 1/3 SG (альт. форма)
        r'\blege\b',                # 1/3 SG (альт. форма)
        r'\bleag(on)?\b',           # 1/3 SG
        r'\blægon\b',               # PL
        r'\bleagon\b',              # PL (альт.)
        # Императив
        r'\blecg\b',                # SG
        r'\blecgia[ðþ]\b',          # PL
        # Причастия
        r'\bge?le[gd]\b',           # причастие (gelaid)
        r'\bge?lag\b',              # причастие (альт. форма)
        r'\bge?legæn\b',            # причастие (редкая форма)
    ]

    sentences = text.split('#')
    contexts = []
    sent_num = 0

    patterns = [re.compile(p, re.IGNORECASE) for p in lecgan_patterns]

    for sentence in sentences:
        sent_num += 1
        sentence = sentence.strip()
        if len(sentence) < 10:
            continue

        words = re.findall(r'[a-zA-Zàáæðéèíóòúýþÿ-]+', sentence)

        for word_idx, word in enumerate(words):
            clean_word = word.lower()

            matched = False
            for pattern in patterns:
                if pattern.match(clean_word):
                    matched = True
                    break

            if matched:
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
                break

    print(f"✓ Найдено {len(contexts)} контекстов глагола lecgan (все формы)")
    return contexts

def analyze_clusters(contexts, cluster_labels, feature_names, vectorizer):
    """Детальный анализ каждого кластера"""

    print("\n" + "="*70)
    print("🔍 ДЕТАЛЬНЫЙ АНАЛИЗ КЛАСТЕРОВ")
    print("="*70)

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

        cluster_indices = [i for i, label in enumerate(cluster_labels) if label == cluster_id]
        cluster_contexts = [contexts[i] for i in cluster_indices]

        size = len(cluster_contexts)
        print(f"\n1. РАЗМЕР КЛАСТЕРА: {size} контекстов ({size/len(contexts)*100:.1f}%)")

        cluster_tfidf = vectorizer.transform([' '.join(ctx['before'][-10:] + ctx['after'][:10]) for ctx in cluster_contexts])
        mean_tfidf = np.mean(cluster_tfidf.toarray(), axis=0)

        top20_idx = mean_tfidf.argsort()[-20:][::-1]
        top20_words = [(feature_names[i], mean_tfidf[i]) for i in top20_idx]

        print(f"\n2. ТОП-20 СЛОВ ПО СРЕДНЕМУ TF-IDF:")
        for rank, (word, score) in enumerate(top20_words, 1):
            print(f"   {rank:2d}. {word:20s} : {score:.4f}")

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

    reducer = umap.UMAP(
        n_neighbors=min(15, len(contexts) - 1),
        min_dist=0.1,
        n_components=2,
        metric='euclidean',
        random_state=42
    )

    embedding = reducer.fit_transform(features)

    fig = plt.figure(figsize=(20, 12))

    # 1. UMAP по кластерам
    ax1 = plt.subplot(2, 3, 1)

    unique_labels = set(clusters)
    if -1 in unique_labels:
        unique_labels.remove(-1)

    n_clusters = len(unique_labels)
    colors = plt.cm.tab20(np.linspace(0, 1, max(21, n_clusters + 1)))

    for label in unique_labels:
        mask = clusters == label
        ax1.scatter(embedding[mask, 0], embedding[mask, 1],
                   c=[colors[label]], s=80, alpha=0.7,
                   edgecolors='black', linewidth=0.5, label=f'Кластер {label}')

    noise_mask = clusters == -1
    if noise_mask.any():
        ax1.scatter(embedding[noise_mask, 0], embedding[noise_mask, 1],
                   c='lightgray', s=50, alpha=0.5, label='Шум')

    ax1.set_title('UMAP: Кластеры lecgan (HDBSCAN) - Calgary (все формы)', fontweight='bold', fontsize=11)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)

    # 2. Распределение размеров кластеров
    ax2 = plt.subplot(2, 3, 2)

    cluster_sizes = [cluster_info[c]['size'] for c in sorted(unique_labels)]
    cluster_ids = list(sorted(unique_labels))

    if cluster_sizes:
        bars = ax2.bar(cluster_ids, cluster_sizes, color=[colors[i] for i in cluster_ids],
                       edgecolor='black')
        ax2.set_title('Размеры кластеров', fontweight='bold', fontsize=12)
        ax2.set_xlabel('Кластер')
        ax2.set_ylabel('Количество контекстов')
        ax2.grid(True, alpha=0.3, axis='y')

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
    ax3.set_title('Частотность форм lecgan', fontweight='bold', fontsize=12)
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

    if unique_labels:
        stats_text += "РАЗМЕРЫ КЛАСТЕРОВ:\n"
        for cluster_id in sorted(unique_labels):
            size = cluster_info[cluster_id]['size']
            pct = size / len(contexts) * 100
            stats_text += f"  Кластер {cluster_id}: {size} ({pct:.1f}%)\n"

    stats_text += f"\nФОРМЫ LECGAN (топ-15):\n"
    for form, count in form_dist.most_common(15):
        stats_text += f"  {form}: {count}\n"

    ax4.text(0, 1, stats_text, fontsize=9, family='monospace',
            verticalalignment='top', transform=ax4.transAxes)

    # 5. Топ слова для каждого кластера
    ax5 = plt.subplot(2, 3, 5)
    ax5.axis('off')

    if unique_labels:
        top_words_text = "ТОП-10 СЛОВ ДЛЯ КАЖДОГО КЛАСТЕРА:\n\n"

        for cluster_id in sorted(unique_labels):
            top_words = cluster_info[cluster_id]['top_words'][:10]
            top_words_text += f"Кластер {cluster_id} ({cluster_info[cluster_id]['size']}):\n"
            for rank, (word, score) in enumerate(top_words, 1):
                top_words_text += f"  {rank}. {word:15s} ({score:.3f})\n"
            top_words_text += "\n"

        ax5.text(0, 1, top_words_text, fontsize=7, family='monospace',
                verticalalignment='top', transform=ax5.transAxes)

    # 6. Описание семантики
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')

    semantics_text = "СЕМАНТИКА LECGAN:\n\n"
    semantics_text += "ЛЕКГАН (класть, положить)\n"
    semantics_text += "Конкретный глагол физического действия\n\n"
    semantics_text += "Основные значения:\n"
    semantics_text += "  • Класть, помещать\n"
    semantics_text += "  • Полагать, основывать\n"
    semantics_text += "  • Созидать (редко)\n"
    semantics_text += "  • Укладывать (спать)\n\n"

    semantics_text += "Сравнение с DON:\n"
    semantics_text += "  • DON: широкий спектр действий\n"
    semantics_text += "  • LECGAN: специфическое действие\n\n"

    semantics_text += "Ожидается:\n"
    semantics_text += "  • Физические объекты\n"
    semantics_text += "  • Пространственные отношения\n"
    semantics_text += "  • Строительные контексты\n"
    semantics_text += "  • Метафорические употребления"

    ax6.text(0, 1, semantics_text, fontsize=9, family='monospace',
            verticalalignment='top', transform=ax6.transAxes)

    plt.tight_layout()
    plt.savefig('/Users/nora/src/Starostin/old_english/lecgan_tfidf_umap.png', dpi=300, bbox_inches='tight')
    print("✓ Визуализация сохранена: lecgan_tfidf_umap.png")
    plt.close()

def save_results(contexts, cluster_info, clusters):
    """Сохранение результатов"""

    print("\n💾 Сохранение результатов...")

    with open('/Users/nora/src/Starostin/old_english/lecgan_tfidf_results.txt', 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("АНАЛИЗ ГЛАГОЛА LECGAN (Calgary Corpus, ВСЕ ФОРМЫ)\n")
        f.write("Методы: TF-IDF + HDBSCAN + UMAP\n")
        f.write("="*70 + "\n\n")

        f.write("ЧАСТОТНОСТЬ ФОРМ:\n")
        form_dist = Counter([ctx['form'] for ctx in contexts])
        for form, count in form_dist.most_common():
            f.write(f"  {form:15s}: {count}\n")

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

            f.write("ТОП-20 СЛОВ (TF-IDF):\n")
            top_words = cluster_info[cluster_id]['top_words']
            for rank, (word, score) in enumerate(top_words, 1):
                f.write(f"  {rank:2d}. {word:20s} : {score:.4f}\n")

            f.write(f"\nВСЕ КОНТЕКСТЫ КЛАСТЕРА:\n")
            for ctx in cluster_contexts:
                before = ' '.join(ctx['before'][-5:])
                after = ' '.join(ctx['after'][:5:])
                f.write(f"  [{ctx['sent_num']:4d}] ...{before} [{ctx['form']}] {after}...\n")

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

    print("✓ Результаты сохранены: lecgan_tfidf_results.txt")

def main():
    print("="*70)
    print("🔬 АНАЛИЗ LECGAN (Calgary, ВСЕ ФОРМЫ) - TF-IDF + HDBSCAN + UMAP")
    print("="*70)

    text = load_calgary_corpus('/Users/nora/src/Starostin/calgary')

    if len(text) < 1000:
        print("❌ Корпус слишком мал")
        return

    contexts = extract_lecgan_contexts(text, window=15)

    if len(contexts) < 10:
        print("❌ Слишком мало контекстов")
        return

    form_dist = Counter([ctx['form'] for ctx in contexts])
    print(f"\n📊 Частотность форм lecgan:")
    for form, count in form_dist.most_common(15):
        print(f"  {form:15s}: {count}")

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

    print("\n🎯 HDBSCAN кластеризация...")

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=8,
        min_samples=2,
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

    cluster_dist = Counter(clusters)
    for cluster_id in sorted(unique_clusters):
        print(f"   Кластер {cluster_id}: {cluster_dist[cluster_id]} контекстов")

    cluster_info = analyze_clusters(contexts, clusters, feature_names, vectorizer)
    visualize_umap(features, clusters, cluster_info, contexts)
    save_results(contexts, cluster_info, clusters)

    print("\n" + "="*70)
    print("✅ АНАЛИЗ ЗАВЕРШЁН!")
    print("="*70)
    print("\nСозданные файлы:")
    print("  • lecgan_tfidf_umap.png - визуализация UMAP")
    print("  • lecgan_tfidf_results.txt - результаты с топ-словами")

if __name__ == '__main__':
    main()
