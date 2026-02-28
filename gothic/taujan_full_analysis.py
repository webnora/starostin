#!/usr/bin/env python3
"""
Полный анализ глагола taujan:
1. Семантическая кластеризация по смыслам
2. t-SNE визуализация
3. Анализ контекстов (существительные, прилагательные, дополнения)
"""

import re
from collections import Counter, defaultdict
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_corpus(filename):
    """Загружает корпус"""
    print("📖 Загрузка корпуса...")
    with open(filename, 'r', encoding='utf-8') as f:
        text = f.read()
    lines = [l.strip() for l in text.split('\n') if l.strip() and not l.startswith('#')]
    print(f"✓ Загружено {len(lines)} строк")
    return lines

def extract_contexts(lines, window=10):
    """Извлекает контексты taujan"""

    patterns = [
        r'\btaujan\b', r'\btaujis\b', r'\btauji[þþ]\b', r'\btauja[iï]d\b',
        r'\btaujai[þþ]\b', r'\btaujand\b', r'\btaujands\b',
        r'\bgataujan\b', r'\bgatauj\b', r'\bgatauji[þþ]\b',
        r'\bgataujands\b', r'\bgataujand\b', r'\btaujaid\b'
    ]

    combined = '|'.join(patterns)
    contexts = []

    for line_num, line in enumerate(lines, 1):
        for match in re.finditer(combined, line, re.IGNORECASE):
            form = match.group()

            # Контекст
            before = line[max(0, match.start()-200):match.start()]
            after = line[match.end():min(len(line), match.end()+200)]

            before_tokens = re.findall(r'\b[\w-]+\b', before)[-window:]
            after_tokens = re.findall(r'\b[\w-]+\b', after)[:window]
            full_tokens = re.findall(r'\b[\w-]+\b', line)

            contexts.append({
                'form': form,
                'before': before_tokens,
                'after': after_tokens,
                'tokens': full_tokens,
                'line': line_num,
                'text': line
            })

    print(f"✓ Найдено {len(contexts)} вхождений taujan")
    return contexts

def categorize_by_markers(contexts):
    """Категоризация по контекстуальным маркерам"""

    markers = {
        'divine': ['gud', 'attin', 'frauja', 'Iesus', 'Xristus', 'ahma', 'sunus', 'faþer'],
        'teaching': ['waurd', 'lais', 'witoþ', 'qenþ', 'rodida', 'anabiud', 'qaþ'],
        'good_deeds': ['waila', 'godo', 'armaion', 'gudabaubo', 'libains', 'sokei'],
        'evil_deeds': ['ubilo', 'frawaurht', 'horin', 'wiþra', 'iddeig', 'sunjs'],
        'miracles': ['taikne', 'macleig', 'galaubein', 'usleiþ'],
        'obedience': ['wiljan', 'skalks', 'fraletan', 'gahausjan', 'qimi', 'gaggi'],
        'creation': ['waurstw', 'akran', 'bagms', 'waurkjan', 'gaskapjan', 'taujiþs'],
        'tree_metaphor': ['bagms', 'akran', 'þiuþeigs', 'ubils', 'watars']
    }

    categories = defaultdict(list)

    for ctx in contexts:
        text = ' '.join(ctx['tokens']).lower()
        categorized = False

        for cat, cat_markers in markers.items():
            if any(m.lower() in text for m in cat_markers):
                categories[cat].append(ctx)
                categorized = True
                break

        if not categorized:
            categories['other'].append(ctx)

    print("\n📊 Ручная категоризация:")
    for cat, ctxs in sorted(categories.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"  {cat:20s}: {len(ctxs):3d}")

    return categories

def analyze_pos_patterns(contexts):
    """Анализ частей речи в контекстах"""

    # Считаем слова рядом с taujan
    word_by_form = defaultdict(Counter)
    all_context_words = Counter()

    prep_list = ['in', 'ana', 'fra', 'du', 'us', 'faura', 'bi', 'und', 'miþ', 'uh', 'þanuh']

    for ctx in contexts:
        form = ctx['form']
        context_words = ctx['before'][-5:] + ctx['after'][:5]

        for word in context_words:
            w = word.lower()
            word_by_form[form][w] += 1
            all_context_words[w] += 1

    print("\n🔤 Частотные слова по формам taujan:")
    for form in sorted(word_by_form.keys()):
        print(f"\n  {form}:")
        for word, count in word_by_form[form].most_common(8):
            print(f"    {word:20s}: {count}")

    # Препозиции
    preps = Counter()
    for ctx in contexts:
        for word in ctx['before'][-3:] + ctx['after'][:3]:
            if word.lower() in prep_list:
                preps[word.lower()] += 1

    print(f"\n📍 Препозиции:")
    for prep, count in preps.most_common():
        print(f"  {prep:10s}: {count}")

    # Существительные-объекты (упрощённо)
    nouns_near_taujan = []
    for ctx in contexts:
        # Ближайшее слово после taujan
        if ctx['after']:
            nouns_near_taujan.append(ctx['after'][0].lower())

    noun_dist = Counter(nouns_near_taujan)
    print(f"\n📦 Слова после taujan (объекты действия):")
    for word, count in noun_dist.most_common(15):
        print(f"  {word:20s}: {count}")

    return word_by_form, preps, noun_dist

def cluster_analysis(contexts, n_clusters=6):
    """Кластеризация контекстов"""

    # TF-IDF
    corpus = [' '.join(ctx['before'][-6:] + ctx['after'][:6]) for ctx in contexts]

    vectorizer = TfidfVectorizer(max_features=80, min_df=2, token_pattern=r'\b[\w-]+\b')
    features = vectorizer.fit_transform(corpus).toarray()

    print(f"\n🔢 TF-IDF признаки: {features.shape}")

    # K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20, max_iter=300)
    clusters = kmeans.fit_predict(features)

    # Распределение
    cluster_dist = Counter(clusters)
    print(f"\n🎯 K-Means кластеры (n={n_clusters}):")
    for i in range(n_clusters):
        print(f"  Кластер {i}: {cluster_dist[i]} контекстов")

    # Ключевые слова
    feature_names = vectorizer.get_feature_names_out()
    centers = kmeans.cluster_centers_

    print(f"\n🔑 Ключевые слова кластеров:")
    for i in range(n_clusters):
        top_idx = centers[i].argsort()[-10:][::-1]
        top_words = [feature_names[j] for j in top_idx]
        print(f"  Кластер {i}: {', '.join(top_words)}")

    return clusters, features, vectorizer

def visualize(contexts, features, clusters):
    """Визуализация результатов"""

    # t-SNE
    print("\n🎨 Вычисляю t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=25, max_iter=1000, init='pca')
    tsne_emb = tsne.fit_transform(features)

    # PCA
    pca = PCA(n_components=2)
    pca_emb = pca.fit_transform(features)

    fig = plt.figure(figsize=(22, 14))

    # 1. t-SNE кластеры
    ax1 = plt.subplot(2, 4, 1)
    colors = plt.cm.tab10(np.linspace(0, 1, max(clusters) + 1))
    for i in range(len(contexts)):
        ax1.scatter(tsne_emb[i, 0], tsne_emb[i, 1],
                   c=[colors[clusters[i]]], s=70, alpha=0.7, edgecolors='black', linewidth=0.5)
    ax1.set_title('t-SNE: Кластеры контекстов', fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # 2. t-SNE по формам
    ax2 = plt.subplot(2, 4, 2)
    unique_forms = sorted(set(ctx['form'] for ctx in contexts))
    form_to_idx = {f: i for i, f in enumerate(unique_forms)}
    form_colors = plt.cm.Set3(np.linspace(0, 1, len(unique_forms)))

    for i, ctx in enumerate(contexts):
        ax2.scatter(tsne_emb[i, 0], tsne_emb[i, 1],
                   c=[form_colors[form_to_idx[ctx['form']]]], s=70, alpha=0.7,
                   edgecolors='black', linewidth=0.5)
    ax2.set_title('t-SNE: По формам глагола', fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Легенда
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=form_colors[form_to_idx[f]], label=f)
                      for f in unique_forms]
    ax2.legend(handles=legend_elements, loc='best', fontsize=7)

    # 3. PCA кластеры
    ax3 = plt.subplot(2, 4, 3)
    for i in range(len(contexts)):
        ax3.scatter(pca_emb[i, 0], pca_emb[i, 1],
                   c=[colors[clusters[i]]], s=70, alpha=0.7, edgecolors='black', linewidth=0.5)
    ax3.set_title(f'PCA (variance={pca.explained_variance_ratio_.sum():.2f})', fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # 4. Распределение кластеров
    ax4 = plt.subplot(2, 4, 4)
    cluster_dist = Counter(clusters)
    ax4.bar(cluster_dist.keys(), cluster_dist.values(),
           color=[colors[i] for i in cluster_dist.keys()], edgecolor='black')
    ax4.set_title('Распределение по кластерам', fontweight='bold')
    ax4.set_xlabel('Кластер')
    ax4.set_ylabel('Количество')
    ax4.grid(True, alpha=0.3, axis='y')

    # 5. Частотность форм
    ax5 = plt.subplot(2, 4, 5)
    form_dist = Counter([ctx['form'] for ctx in contexts])
    forms = list(form_dist.keys())
    counts = list(form_dist.values())
    ax5.barh(range(len(forms)), counts, color='steelblue', edgecolor='black')
    ax5.set_yticks(range(len(forms)))
    ax5.set_yticklabels(forms)
    ax5.set_title('Частотность форм taujan', fontweight='bold')
    ax5.set_xlabel('Количество')
    ax5.grid(True, alpha=0.3, axis='x')

    # 6. t-SNE: Бог vs Люди
    ax6 = plt.subplot(2, 4, 6)

    divine_markers = ['gud', 'attin', 'frauja', 'Iesus', 'Xristus', 'ahma']
    human_markers = ['mann', 'jah', 'jus', 'izwis', 'weis', 'saei']

    divine_pts = []
    human_pts = []
    neutral_pts = []

    for i, ctx in enumerate(contexts):
        text = ' '.join(ctx['tokens']).lower()
        if any(m.lower() in text for m in divine_markers):
            divine_pts.append(i)
        elif any(m.lower() in text for m in human_markers):
            human_pts.append(i)
        else:
            neutral_pts.append(i)

    for i in divine_pts:
        ax6.scatter(tsne_emb[i, 0], tsne_emb[i, 1], c='gold', s=80, alpha=0.8,
                   edgecolors='black', linewidth=1, label='Divine' if i == divine_pts[0] else '')
    for i in human_pts:
        ax6.scatter(tsne_emb[i, 0], tsne_emb[i, 1], c='lightblue', s=80, alpha=0.8,
                   edgecolors='black', linewidth=1, label='Human' if i == human_pts[0] else '')
    for i in neutral_pts:
        ax6.scatter(tsne_emb[i, 0], tsne_emb[i, 1], c='lightgray', s=60, alpha=0.6,
                   edgecolors='black', linewidth=0.5, label='Other' if i == neutral_pts[0] else '')

    ax6.set_title('t-SNE: Divine vs Human', fontweight='bold')
    ax6.legend(loc='best')
    ax6.grid(True, alpha=0.3)

    # 7. Примеры контекстов
    ax7 = plt.subplot(2, 4, 7)
    ax7.axis('off')

    example_text = "ПРИМЕРЫ КОНТЕКСТОВ:\n\n"

    for cluster_id in sorted(set(clusters)):
        cluster_ctxs = [ctx for i, ctx in enumerate(contexts) if clusters[i] == cluster_id]
        example_text += f"Кластер {cluster_id} ({len(cluster_ctxs)}):\n"

        for ctx in cluster_ctxs[:2]:
            before = ' '.join(ctx['before'][-3:])
            after = ' '.join(ctx['after'][:3])
            example_text += f"  ...{before} [{ctx['form']}] {after}...\n"
        example_text += "\n"

    ax7.text(0, 1, example_text, fontsize=7, family='monospace',
            verticalalignment='top', transform=ax7.transAxes)

    # 8. Статистика по категориям
    ax8 = plt.subplot(2, 4, 8)
    ax8.axis('off')

    # Простая статистика
    stats_text = "СТАТИСТИКА:\n\n"

    form_dist = Counter([ctx['form'] for ctx in contexts])
    stats_text += "Формы:\n"
    for form, count in form_dist.most_common():
        stats_text += f"  {form:15s}: {count}\n"

    stats_text += f"\nВсего контекстов: {len(contexts)}\n"
    stats_text += f"Кластеров: {max(clusters) + 1}\n"

    ax8.text(0, 1, stats_text, fontsize=9, family='monospace',
            verticalalignment='top', transform=ax8.transAxes)

    plt.tight_layout()
    plt.savefig('/Users/nora/src/Starostin/taujan_full_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Визуализация сохранена: taujan_full_analysis.png")

    plt.show()

def detailed_cluster_analysis(contexts, clusters):
    """Детальный анализ каждого кластера"""

    print("\n" + "="*70)
    print("🔍 ДЕТАЛЬНЫЙ АНАЛИЗ КЛАСТЕРОВ")
    print("="*70)

    results = {}

    for cluster_id in sorted(set(clusters)):
        cluster_ctxs = [ctx for i, ctx in enumerate(contexts) if clusters[i] == cluster_id]

        print(f"\n{'='*70}")
        print(f"КЛАСТЕР {cluster_id} ({len(cluster_ctxs)} контекстов)")
        print(f"{'='*70}")

        # Формы в кластере
        forms = Counter([ctx['form'] for ctx in cluster_ctxs])
        print(f"\nФормы:")
        for form, count in forms.most_common():
            pct = count / len(cluster_ctxs) * 100
            print(f"  {form:15s}: {count:2d} ({pct:5.1f}%)")

        # Все слова
        all_words = []
        for ctx in cluster_ctxs:
            all_words.extend([w.lower() for w in ctx['before'][-5:] + ctx['after'][:5]])

        word_dist = Counter(all_words)
        print(f"\nКлючевые слова:")
        for word, count in word_dist.most_common(15):
            print(f"  {word:20s}: {count}")

        # Контексты
        print(f"\nПримеры контекстов:")
        for ctx in cluster_ctxs[:4]:
            before = ' '.join(ctx['before'][-4:])
            after = ' '.join(ctx['after'][:4])
            print(f"  [{ctx['line']:4d}] ...{before} [{ctx['form']}] {after}...")

        results[cluster_id] = {
            'count': len(cluster_ctxs),
            'forms': dict(forms),
            'words': dict(word_dist.most_common(15)),
            'examples': [ctx for ctx in cluster_ctxs[:4]]
        }

    return results

def category_analysis(categories):
    """Анализ категорий"""

    print("\n" + "="*70)
    print("📂 АНАЛИЗ КАТЕГОРИЙ")
    print("="*70)

    for cat_name, cat_ctxs in sorted(categories.items(), key=lambda x: len(x[1]), reverse=True):
        if len(cat_ctxs) == 0:
            continue

        print(f"\n{'='*70}")
        print(f"{cat_name.upper()} ({len(cat_ctxs)} контекстов)")
        print(f"{'='*70}")

        forms = Counter([ctx['form'] for ctx in cat_ctxs])
        print(f"\nФормы:")
        for form, count in forms.most_common():
            print(f"  {form:15s}: {count}")

        all_words = []
        for ctx in cat_ctxs:
            all_words.extend([w.lower() for w in ctx['before'][-5:] + ctx['after'][:5]])

        word_dist = Counter(all_words)
        print(f"\nКлючевые слова:")
        for word, count in word_dist.most_common(8):
            print(f"  {word:20s}: {count}")

        print(f"\nПримеры:")
        for ctx in cat_ctxs[:2]:
            before = ' '.join(ctx['before'][-4:])
            after = ' '.join(ctx['after'][:4])
            print(f"  [{ctx['line']}] ...{before} [{ctx['form']}] {after}...")

def save_results(contexts, clusters, categories):
    """Сохранение результатов"""

    print("\n💾 Сохранение результатов...")

    with open('/Users/nora/src/Starostin/taujan_full_results.txt', 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("ПОЛНЫЙ АНАЛИЗ τAUJAN\n")
        f.write("="*70 + "\n\n")

        # Формы
        f.write("ЧАСТОТНОСТЬ ФОРМ:\n")
        forms = Counter([ctx['form'] for ctx in contexts])
        for form, count in forms.most_common():
            f.write(f"  {form:15s}: {count}\n")

        # Кластеры
        f.write("\n\nКЛАСТЕРЫ:\n")
        for cluster_id in sorted(set(clusters)):
            cluster_ctxs = [ctx for i, ctx in enumerate(contexts) if clusters[i] == cluster_id]
            f.write(f"\nКластер {cluster_id} ({len(cluster_ctxs)}):\n")

            all_words = []
            for ctx in cluster_ctxs:
                all_words.extend([w.lower() for w in ctx['before'][-5:] + ctx['after'][:5]])

            word_dist = Counter(all_words)
            for word, count in word_dist.most_common(15):
                f.write(f"  {word:20s}: {count}\n")

        # Все контексты
        f.write("\n\n" + "="*70 + "\n")
        f.write("ВСЕ КОНТЕКСТЫ:\n")
        f.write("="*70 + "\n\n")

        for ctx in contexts:
            before = ' '.join(ctx['before'][-5:])
            after = ' '.join(ctx['after'][:5])
            f.write(f"[{ctx['line']:4d}] {ctx['form']:15s} ...{before} [{ctx['form']}] {after}...\n")

    print("✓ Результаты сохранены: taujan_full_results.txt")

def main():
    print("="*70)
    print("🔬 ПОЛНЫЙ АНАЛИЗ τAUJAN")
    print("="*70)

    # Загрузка
    lines = load_corpus('/Users/nora/src/Starostin/gotica.txt')
    contexts = extract_contexts(lines, window=10)

    # Частота форм
    form_dist = Counter([ctx['form'] for ctx in contexts])
    print(f"\n📊 Частотность форм:")
    for form, count in form_dist.most_common():
        print(f"  {form:15s}: {count}")

    # Категоризация
    categories = categorize_by_markers(contexts)

    # POS анализ
    word_by_form, preps, nouns = analyze_pos_patterns(contexts)

    # Кластеризация
    clusters, features, vectorizer = cluster_analysis(contexts, n_clusters=6)

    # Визуализация
    visualize(contexts, features, clusters)

    # Детальный анализ
    cluster_results = detailed_cluster_analysis(contexts, clusters)

    # Категории
    category_analysis(categories)

    # Сохранение
    save_results(contexts, clusters, categories)

    print("\n" + "="*70)
    print("✅ АНАЛИЗ ЗАВЕРШЁН!")
    print("="*70)
    print("\nСозданные файлы:")
    print("  • taujan_full_analysis.png - визуализация")
    print("  • taujan_full_results.txt - результаты")

if __name__ == '__main__':
    main()
