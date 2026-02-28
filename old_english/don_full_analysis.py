#!/usr/bin/env python3
"""
Полный анализ глагола don (делать) в древнеанглийском (Old English)
Calgary Corpus
"""

import re
import os
from collections import Counter, defaultdict
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_corpus(directory):
    """Загружает все .txt файлы из папки"""
    print("📖 Загрузка корпуса...")

    all_lines = []
    filenames = []

    for filename in sorted(os.listdir(directory)):
        if filename.endswith('.txt'):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()

                # Убираем комментарии в квадратных скобках
                text = re.sub(r'\[[^\]]*\]', '', text)

                # Разбиваем по # и получаем отдельные строки/фразы
                segments = text.split('#')

                for segment in segments:
                    segment = segment.strip()
                    if segment and len(segment) > 10:  # Пропускаем очень короткие
                        all_lines.append(segment)
                filenames.append(filename)

    print(f"✓ Загружено {len(filenames)} файлов, {len(all_lines)} строк")
    return all_lines, filenames

def extract_don_contexts(lines, window=10):
    """Извлекает контексты для всех форм don"""

    # Формы глагола don в древнеанглийском
    don_patterns = [
        r'\bdon\b',           # инфинитив
        r'\bd[eo][ðð]\b',      # deþ, doð (3 л. наст.)
        r'\bdyst\b',          # dost (2 л. наст.)
        r'\bdyde\b',          # dyde (прош. 1/3 л.)
        r'\bdydest\b',        # dydest (прош. 2 л.)
        r'\bdydon\b',         # dydon (прош. пл.)
        r'\bged[eo][ðð]n\b',   # gedon (перфект)
        r'\bged[yy]des\b',    # gedydes (прош. перф.)
        r'\bged[yy]den\b',    # gedýden (парт. перф.)
        r'\b(do|dye|doge)\b'  # subjunctive variants
    ]

    combined = '|'.join(don_patterns)
    contexts = []

    line_num = 0
    for line in lines:
        line_num += 1

        # Очистка строки от # и лишних пробелов
        clean_line = re.sub(r'\s+', ' ', line.replace('#', ' '))

        for match in re.finditer(combined, clean_line, re.IGNORECASE):
            form = match.group()

            # Контекст
            before = clean_line[:match.start()]
            after = clean_line[match.end():]

            before_tokens = re.findall(r'\b[a-zþæðœáéíóúý]+[-]?[a-zþæðœáéíóúý]*\b', before)[-window:]
            after_tokens = re.findall(r'\b[a-zþæðœáéíóúý]+[-]?[a-zþæðœáéíóúý]*\b', after)[:window]

            contexts.append({
                'form': form,
                'before': before_tokens,
                'after': after_tokens,
                'line': line_num,
                'text': clean_line
            })

    print(f"✓ Найдено {len(contexts)} вхождений форм don")
    return contexts

def categorize_by_markers(contexts):
    """Категоризация по контекстуальным маркерам"""

    markers = {
        'divine': ['god', 'heofon', 'drihten', 'scyppend', 'ælmihtig', 'fæder',
                  'helend', 'crist', 'dryhten', 'lord', 'heaven'],
        'teaching': ['lareow', 'læran', 'witega', 'boc', 'godspell', 'spell'],
        'good_deeds': ['göd', 'soð', 'riht', 'æþelu', 'dæd', 'halig'],
        'evil_deeds': ['yfel', 'synn', 'sawul', 'deoful', 'wyrsa', 'firen'],
        'battle': ['beadn', 'wig', 'guþ', 'feohtan', 'sweord', 'lund', 'wiga'],
        'wisdom': ['wis', 'snotor', 'gemynd', 'mod', 'þanc', 'wisdom'],
        'creation': ['gesceap', 'niman', 'wyrcan', 'scieppan', 'timbran']
    }

    categories = defaultdict(list)

    for ctx in contexts:
        text = ' '.join(ctx['tokens'] if 'tokens' in ctx else ctx['before'] + ctx['after']).lower()
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

    word_by_form = defaultdict(Counter)
    all_context_words = Counter()

    for ctx in contexts:
        form = ctx['form']
        context_words = ctx['before'][-5:] + ctx['after'][:5]

        for word in context_words:
            w = word.lower()
            word_by_form[form][w] += 1
            all_context_words[w] += 1

    print("\n🔤 Частотные слова по формам don (топ-5):")
    for form in sorted(word_by_form.keys()):
        print(f"\n  {form}:")
        for word, count in word_by_form[form].most_common(5):
            print(f"    {word:20s}: {count}")

    # Слова после don
    words_after_don = []
    for ctx in contexts:
        if ctx['after']:
            words_after_don.append(ctx['after'][0].lower())

    noun_dist = Counter(words_after_don)
    print(f"\n📦 Слова после don (топ-15):")
    for word, count in noun_dist.most_common(15):
        print(f"  {word:20s}: {count}")

    return word_by_form, noun_dist

def cluster_analysis(contexts, n_clusters=6):
    """Кластеризация контекстов"""

    # TF-IDF
    corpus = [' '.join(ctx['before'][-6:] + ctx['after'][:6]) for ctx in contexts]

    vectorizer = TfidfVectorizer(max_features=80, min_df=2, token_pattern=r'\b[a-zþæðœáéíóúý]+[-]?[a-zþæðœáéíóúý]*\b')
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
    ax1.set_title('t-SNE: Clusters of don contexts', fontweight='bold')
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
    ax2.set_title('t-SNE: By verb form', fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Легенда
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=form_colors[form_to_idx[f]], label=f)
                      for f in unique_forms[:10]]  # Только первые 10
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
    ax4.set_title('Cluster distribution', fontweight='bold')
    ax4.set_xlabel('Cluster')
    ax4.set_ylabel('Count')
    ax4.grid(True, alpha=0.3, axis='y')

    # 5. Частотность форм
    ax5 = plt.subplot(2, 4, 5)
    form_dist = Counter([ctx['form'] for ctx in contexts])
    forms = list(form_dist.keys())
    counts = list(form_dist.values())
    ax5.barh(range(len(forms)), counts, color='steelblue', edgecolor='black')
    ax5.set_yticks(range(len(forms)))
    ax5.set_yticklabels(forms)
    ax5.set_title('Frequency of don forms', fontweight='bold')
    ax5.set_xlabel('Count')
    ax5.grid(True, alpha=0.3, axis='x')

    # 6. Divine vs Human
    ax6 = plt.subplot(2, 4, 6)

    divine_markers = ['god', 'heofon', 'drihten', 'scyppend', 'ælmihtig', 'fæder']
    human_markers = ['mon', 'we', 'git', 'hi', 'þu', 'ic', 'eow']

    divine_pts = []
    human_pts = []
    neutral_pts = []

    for i, ctx in enumerate(contexts):
        text = ' '.join(ctx['before'] + ctx['after']).lower()
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

    example_text = "CONTEXT EXAMPLES:\n\n"

    for cluster_id in sorted(set(clusters)):
        cluster_ctxs = [ctx for i, ctx in enumerate(contexts) if clusters[i] == cluster_id]
        example_text += f"Cluster {cluster_id} ({len(cluster_ctxs)}):\n"

        for ctx in cluster_ctxs[:2]:
            before = ' '.join(ctx['before'][-3:])
            after = ' '.join(ctx['after'][:3])
            example_text += f"  ...{before} [{ctx['form']}] {after}...\n"
        example_text += "\n"

    ax7.text(0, 1, example_text, fontsize=7, family='monospace',
            verticalalignment='top', transform=ax7.transAxes)

    # 8. Статистика
    ax8 = plt.subplot(2, 4, 8)
    ax8.axis('off')

    stats_text = "STATISTICS:\n\n"

    form_dist = Counter([ctx['form'] for ctx in contexts])
    stats_text += "Forms:\n"
    for form, count in form_dist.most_common():
        stats_text += f"  {form:15s}: {count}\n"

    stats_text += f"\nTotal contexts: {len(contexts)}\n"
    stats_text += f"Clusters: {max(clusters) + 1}\n"

    ax8.text(0, 1, stats_text, fontsize=9, family='monospace',
            verticalalignment='top', transform=ax8.transAxes)

    plt.tight_layout()
    plt.savefig('/Users/nora/src/Starostin/don_full_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Визуализация сохранена: don_full_analysis.png")

    plt.show()

def detailed_cluster_analysis(contexts, clusters):
    """Детальный анализ каждого кластера"""

    print("\n" + "="*70)
    print("🔍 DETAILED CLUSTER ANALYSIS")
    print("="*70)

    for cluster_id in sorted(set(clusters)):
        cluster_ctxs = [ctx for i, ctx in enumerate(contexts) if clusters[i] == cluster_id]

        print(f"\n{'='*70}")
        print(f"CLUSTER {cluster_id} ({len(cluster_ctxs)} contexts)")
        print(f"{'='*70}")

        # Формы в кластере
        forms = Counter([ctx['form'] for ctx in cluster_ctxs])
        print(f"\nForms:")
        for form, count in forms.most_common():
            pct = count / len(cluster_ctxs) * 100
            print(f"  {form:15s}: {count:2d} ({pct:5.1f}%)")

        # Все слова
        all_words = []
        for ctx in cluster_ctxs:
            all_words.extend([w.lower() for w in ctx['before'][-5:] + ctx['after'][:5]])

        word_dist = Counter(all_words)
        print(f"\nKey words:")
        for word, count in word_dist.most_common(15):
            print(f"  {word:20s}: {count}")

        # Контексты
        print(f"\nExamples:")
        for ctx in cluster_ctxs[:4]:
            before = ' '.join(ctx['before'][-4:])
            after = ' '.join(ctx['after'][:4])
            print(f"  [{ctx['line']:4d}] ...{before} [{ctx['form']}] {after}...")

def save_results(contexts, clusters):
    """Сохранение результатов"""

    print("\n💾 Сохранение результатов...")

    with open('/Users/nora/src/Starostin/don_full_results.txt', 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("FULL ANALYSIS OF DON (Old English)\n")
        f.write("Calgary Corpus\n")
        f.write("="*70 + "\n\n")

        # Формы
        f.write("FORM FREQUENCY:\n")
        forms = Counter([ctx['form'] for ctx in contexts])
        for form, count in forms.most_common():
            f.write(f"  {form:15s}: {count}\n")

        # Кластеры
        f.write("\n\nCLUSTERS:\n")
        for cluster_id in sorted(set(clusters)):
            cluster_ctxs = [ctx for i, ctx in enumerate(contexts) if clusters[i] == cluster_id]
            f.write(f"\nCluster {cluster_id} ({len(cluster_ctxs)}):\n")

            all_words = []
            for ctx in cluster_ctxs:
                all_words.extend([w.lower() for w in ctx['before'][-5:] + ctx['after'][:5]])

            word_dist = Counter(all_words)
            for word, count in word_dist.most_common(15):
                f.write(f"  {word:20s}: {count}\n")

        # Все контексты
        f.write("\n\n" + "="*70 + "\n")
        f.write("ALL CONTEXTS:\n")
        f.write("="*70 + "\n\n")

        for ctx in contexts:
            before = ' '.join(ctx['before'][-5:])
            after = ' '.join(ctx['after'][:5])
            f.write(f"[{ctx['line']:4d}] {ctx['form']:15s} ...{before} [{ctx['form']}] {after}...\n")

    print("✓ Результаты сохранены: don_full_results.txt")

def main():
    print("="*70)
    print("🔬 FULL ANALYSIS OF DON (OLD ENGLISH)")
    print("="*70)

    # Загрузка
    lines, files = load_corpus('/Users/nora/src/Starostin/calgary')
    contexts = extract_don_contexts(lines, window=10)

    if len(contexts) == 0:
        print("❌ Не найдено контекстов don")
        return

    # Частота форм
    form_dist = Counter([ctx['form'] for ctx in contexts])
    print(f"\n📊 Частотность форм:")
    for form, count in form_dist.most_common():
        print(f"  {form:15s}: {count}")

    # Категоризация
    # categories = categorize_by_markers(contexts)

    # POS анализ
    word_by_form, nouns = analyze_pos_patterns(contexts)

    # Кластеризация
    clusters, features, vectorizer = cluster_analysis(contexts, n_clusters=6)

    # Визуализация
    visualize(contexts, features, clusters)

    # Детальный анализ
    detailed_cluster_analysis(contexts, clusters)

    # Сохранение
    save_results(contexts, clusters)

    print("\n" + "="*70)
    print("✅ ANALYSIS COMPLETE!")
    print("="*70)
    print("\nCreated files:")
    print("  • don_full_analysis.png - visualization")
    print("  • don_full_results.txt - results")

if __name__ == '__main__':
    main()
