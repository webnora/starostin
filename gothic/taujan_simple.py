#!/usr/bin/env python3
"""
Упрощённый анализ taujan для отладки
"""

import re
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans']

def load_corpus(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        text = f.read()

    lines = [line.strip() for line in text.split('\n') if line.strip() and not line.startswith('#')]
    return lines

def extract_contexts(lines, window=8):
    patterns = [r'\btaujan\b', r'\btaujis\b', r'\btauji[þþ]\b', r'\btauja[iï]d\b',
                r'\btaujai[þþ]\b', r'\btaujand\b', r'\btaujands\b',
                r'\bgataujan\b', r'\bgatauji[þþ]\b']

    combined = '|'.join(patterns)
    contexts = []

    for line_num, line in enumerate(lines, 1):
        for m in re.finditer(combined, line, re.IGNORECASE):
            form = m.group()

            before_tokens = re.findall(r'\b[\w-]+\b', line[max(0, m.start()-150):m.start()])[-window:]
            after_tokens = re.findall(r'\b[\w-]+\b', line[m.end():min(len(line), m.end()+150)])[:window]

            contexts.append({
                'form': form,
                'tokens': before_tokens + after_tokens,
                'line': line_num
            })

    return contexts

def main():
    print("🔬 Анализ taujan")
    lines = load_corpus('/Users/nora/src/Starostin/gotica.txt')
    print(f"Загружено {len(lines)} строк")

    contexts = extract_contexts(lines)
    print(f"Найдено {len(contexts)} контекстов")

    # Частота форм
    form_dist = Counter([c['form'] for c in contexts])
    print("\nФормы:")
    for f, cnt in form_dist.most_common():
        print(f"  {f:15s}: {cnt}")

    # TF-IDF
    corpus = [' '.join(c['tokens']) for c in contexts]
    vectorizer = TfidfVectorizer(max_features=100, min_df=2)
    features = vectorizer.fit_transform(corpus).toarray()
    print(f"\nTF-IDF: {features.shape}")

    # Кластеризация
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(features)

    cluster_dist = Counter(clusters)
    print("\nКластеры:")
    for i in range(5):
        print(f"  Кластер {i}: {cluster_dist[i]}")

    # Ключевые слова
    feature_names = vectorizer.get_feature_names_out()
    centers = kmeans.cluster_centers_

    print("\nКлючевые слова кластеров:")
    for i in range(5):
        top_idx = centers[i].argsort()[-8:][::-1]
        top_words = [feature_names[j] for j in top_idx]
        print(f"  {i}: {', '.join(top_words)}")

    # t-SNE
    print("\nВычисляю t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    emb = tsne.fit_transform(features)

    # Визуализация
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 1. t-SNE по кластерам
    colors = plt.cm.tab10(np.linspace(0, 1, 5))
    for i in range(len(contexts)):
        axes[0].scatter(emb[i, 0], emb[i, 1], c=[colors[clusters[i]]], s=50, alpha=0.7)
    axes[0].set_title('t-SNE: Кластеры')
    axes[0].grid(True, alpha=0.3)

    # 2. t-SNE по формам
    form_to_color = {f: i for i, f in enumerate(sorted(set(c['form'] for c in contexts)))}
    form_colors = plt.cm.Set3(np.linspace(0, 1, len(form_to_color)))

    for i, ctx in enumerate(contexts):
        c = form_colors[form_to_color[ctx['form']]]
        axes[1].scatter(emb[i, 0], emb[i, 1], c=[c], s=50, alpha=0.7)
    axes[1].set_title('t-SNE: Формы')
    axes[1].grid(True, alpha=0.3)

    # 3. Распределение кластеров
    axes[2].bar(cluster_dist.keys(), cluster_dist.values(), color=[colors[i] for i in cluster_dist.keys()])
    axes[2].set_title('Распределение кластеров')
    axes[2].set_xlabel('Кластер')
    axes[2].set_ylabel('Количество')
    axes[2].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('/Users/nora/src/Starostin/taujan_simple.png', dpi=300)
    print("✓ Сохранено: taujan_simple.png")

    # Детали кластеров
    print("\n" + "="*60)
    print("ДЕТАЛИ КЛАСТЕРОВ")
    print("="*60)

    for cluster_id in range(5):
        cluster_ctxs = [ctx for i, ctx in enumerate(contexts) if clusters[i] == cluster_id]
        print(f"\nКластер {cluster_id} ({len(cluster_ctxs)}):")

        words = []
        for ctx in cluster_ctxs:
            words.extend([w.lower() for w in ctx['tokens']])

        word_dist = Counter(words)
        print("  Топ слова:", ', '.join([w for w, _ in word_dist.most_common(10)]))

        forms = Counter([ctx['form'] for ctx in cluster_ctxs])
        print("  Формы:", ', '.join([f"{f}({c})" for f, c in forms.most_common()]))

if __name__ == '__main__':
    main()
