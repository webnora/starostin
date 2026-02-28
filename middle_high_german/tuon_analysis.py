#!/usr/bin/env python3
"""
Унификация и анализ MHG корпуса
1. Извлечение чистого текста из размеченных файлов
2. Поиск глагола tuon (делать)
3. Кластеризация и визуализация
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

def parse_mhg_filename(filename):
    """Извлекает метаданные из имени файла"""
    # Пример: 1050_rheinauer_alemannic.ver0_8.txt
    basename = os.path.basename(filename).replace('.ver0_8.txt', '').replace('.ver0_7a.txt', '').replace('.ver0_7b.txt', '')

    parts = basename.split('_')

    metadata = {
        'filename': filename,
        'basename': basename,
        'date': parts[0] if len(parts) > 0 else 'unknown',
        'source': parts[1] if len(parts) > 1 else 'unknown',
        'dialect': parts[2] if len(parts) > 2 else 'unknown'
    }

    return metadata

def extract_plain_text_from_mhg(filepath):
    """Извлекает чистый текст из MHG файла с разметкой"""

    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    # Извлекаем словоформы из конструкции (TAG lemma=wordform) или (TAG wordform=lemma)
    # Паттерн захватывает: (TAG^features wordform=lemma) или (TAG lemma=wordform)
    tokens = re.findall(r'\([A-Z][A-Z^]*\s+([a-zA-ZäöüßÄÖÜəëįŋᵉͮ]+)=([a-zA-ZäöüßÄÖÜəëįŋᵉͮ<>\-]+)\)', content)
    # Или обратный порядок
    tokens_reverse = re.findall(r'\([A-Z][A-Z^]*\s+([a-zA-ZäöüßÄÖÜəëįŋᵉͮ<>\-]+)=([a-zA-ZäöüßÄÖÜəëįŋᵉͮ]+)\)', content)

    wordforms = []

    # Из первого паттерна: (TAG lemma=wordform) -> wordform
    for match in tokens:
        lemma, wordform = match
        # Очищаем от диакритики и спецсимволов
        wordform_clean = re.sub(r'[<>\-†*]', '', wordform)
        if wordform_clean and len(wordform_clean) > 0:
            wordforms.append(wordform_clean)

    # Из второго паттерна: (TAG wordform=lemma) -> wordform
    for match in tokens_reverse:
        wordform, lemma = match
        wordform_clean = re.sub(r'[<>\-†*]', '', wordform)
        if wordform_clean and len(wordform_clean) > 0:
            wordforms.append(wordform_clean)

    # Также извлекаем знаки препинания
    punctuation = re.findall(r'\([,.?!:;]\s+([,.:;?!])=\1\)', content)
    wordforms.extend([p[0] for p in punctuation if p])

    # Разбиваем на предложения по ID или по точкам
    sentences = []
    current_sentence = []

    # Извлекаем ID предложений для разделения
    sentence_ids = re.findall(r'\(ID\s+([^)]+)\)', content)

    for wordform in wordforms:
        current_sentence.append(wordform)
        # Разделитель предложений - точка или знак
        if wordform in ['.', '!', '?']:
            if current_sentence:
                sentences.append(' '.join(current_sentence))
                current_sentence = []

    # Добавляем последнее предложение если есть
    if current_sentence:
        sentences.append(' '.join(current_sentence))

    return sentences

def build_unified_corpus(directory):
    """Строит унифицированный корпус из всех MHG файлов"""
    print("📖 Построение унифицированного MHG корпуса...")

    all_sentences = []
    metadata_list = []

    files = [f for f in os.listdir(directory) if f.endswith('.txt')]
    print(f"  Найдено {len(files)} файлов")

    for filename in sorted(files):
        filepath = os.path.join(directory, filename)

        try:
            metadata = parse_mhg_filename(filepath)
            sentences = extract_plain_text_from_mhg(filepath)

            # Добавляем метаданные к каждому предложению
            for sent in sentences:
                all_sentences.append(sent)
                metadata_list.append(metadata)

        except Exception as e:
            print(f"  Ошибка в {filename}: {e}")
            continue

    print(f"✓ Извлечено {len(all_sentences)} предложений")
    return all_sentences, metadata_list

def extract_tuon_contexts(sentences, window=10):
    """Извлекает контексты для всех форм tuon"""

    # Формы глагола tuon в древневерхнемецком
    # tuon (инфинитив), tuo (настоящее), tuost, tuot, tat (прошедшее), tāten, gitān
    tuon_patterns = [
        r'\btuon\b',
        r'\btuo\b',
        r'\btuost\b',
        r'\btuot\b',
        r'\btat\b',
        r'\btāten\b',
        r'\btaten\b',
        r'\bgitān\b',
        r'\bgedan\b',
        r'\bgetan\b'
    ]

    combined = '|'.join(tuon_patterns)
    contexts = []

    for sent_num, sentence in enumerate(sentences, 1):
        # Извлекаем слова
        words = re.findall(r'[a-zA-ZäöüßÄÖÜəëįŋᵉͮáéíóúý]+[,.:;?!]*', sentence)

        # Ищем формы tuon
        for word_idx, word in enumerate(words):
            clean_word = re.sub(r'[,.:;?!]', '', word)

            if re.match(combined, clean_word, re.IGNORECASE):
                form = clean_word

                # Контекст
                before_words = [re.sub(r'[,.:;?!]', '', w) for w in words[:word_idx]]
                after_words = [re.sub(r'[,.:;?!]', '', w) for w in words[word_idx+1:]]

                contexts.append({
                    'form': form,
                    'before': before_words[-window:],
                    'after': after_words[:window],
                    'sentence': sentence,
                    'sent_num': sent_num
                })

    print(f"✓ Найдено {len(contexts)} вхождений форм tuon")
    return contexts

def categorize_by_markers(contexts):
    """Категоризация по контекстуальным маркерам"""

    markers = {
        'divine': ['got', 'herre', 'trut', 'himel', 'engel', 'bischof', 'priester', 'chaiser'],
        'teaching': ['lere', 'lêr', 'predige', 'wizzen', 'kunt', 'verstân'],
        'good_deeds': ['guot', 'güet', 'reht', 'barmherze', 'milte', 'werke'],
        'evil_deeds': ['bôse', 'untat', 'sunde', 'missetat', 'quâlat', 'nôt'],
        'battle': ['strît', 'kampf', 'schwert', 'ritter', 'slahen', 'gewafen'],
        'command': ['sol', 'muoz', 'bot', 'hîzen', 'befelhen', 'geheißen'],
        'wisdom': ['wîsheit', 'list', 'kunstre', 'meister', ' kunst']
    }

    categories = defaultdict(list)

    for ctx in contexts:
        text = ' '.join(ctx['before'] + ctx['after']).lower()
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

    print("\n🔤 Частотные слова по формам tuon (топ-5):")
    for form in sorted(word_by_form.keys()):
        print(f"\n  {form}:")
        for word, count in word_by_form[form].most_common(5):
            print(f"    {word:20s}: {count}")

    # Слова после tuon
    words_after = []
    for ctx in contexts:
        if ctx['after']:
            words_after.append(ctx['after'][0].lower())

    noun_dist = Counter(words_after)
    print(f"\n📦 Слова после tuon (топ-15):")
    for word, count in noun_dist.most_common(15):
        print(f"  {word:20s}: {count}")

    return word_by_form, noun_dist

def cluster_analysis(contexts, n_clusters=6):
    """Кластеризация контекстов"""

    # TF-IDF
    corpus = [' '.join(ctx['before'][-6:] + ctx['after'][:6]) for ctx in contexts]

    vectorizer = TfidfVectorizer(max_features=80, min_df=2, token_pattern=r'[a-zA-ZäöüßÄÖÜəëįŋᵉͮ]+')
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
    ax1.set_title('t-SNE: Кластеры контекстов tuon', fontweight='bold')
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

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=form_colors[form_to_idx[f]], label=f)
                      for f in unique_forms[:10]]
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
    ax5.set_title('Частотность форм tuon', fontweight='bold')
    ax5.set_xlabel('Количество')
    ax5.grid(True, alpha=0.3, axis='x')

    # 6. Divine vs Human
    ax6 = plt.subplot(2, 4, 6)

    divine_markers = ['got', 'herre', 'trut', 'himel', 'engel', 'chaiser']
    human_markers = ['man', 'wir', 'ir', 'ich', 'mensche', 'ritter']

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

    # 8. Статистика
    ax8 = plt.subplot(2, 4, 8)
    ax8.axis('off')

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
    plt.savefig('/Users/nora/src/Starostin/tuon_full_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Визуализация сохранена: tuon_full_analysis.png")

    plt.show()

def detailed_cluster_analysis(contexts, clusters):
    """Детальный анализ каждого кластера"""

    print("\n" + "="*70)
    print("🔍 ДЕТАЛЬНЫЙ АНАЛИЗ КЛАСТЕРОВ")
    print("="*70)

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
        print(f"\nПримеры:")
        for ctx in cluster_ctxs[:4]:
            before = ' '.join(ctx['before'][-4:])
            after = ' '.join(ctx['after'][:4])
            print(f"  [{ctx['sent_num']:4d}] ...{before} [{ctx['form']}] {after}...")

def save_results(contexts, clusters):
    """Сохранение результатов"""

    print("\n💾 Сохранение результатов...")

    with open('/Users/nora/src/Starostin/tuon_full_results.txt', 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("ПОЛНЫЙ АНАЛИЗ TUON (Древневерхненемецкий)\n")
        f.write("Middle High German\n")
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
            f.write(f"[{ctx['sent_num']:4d}] {ctx['form']:15s} ...{before} [{ctx['form']}] {after}...\n")

    print("✓ Результаты сохранены: tuon_full_results.txt")

def main():
    print("="*70)
    print("🔬 ПОЛНЫЙ АНАЛИЗ TUON (ДРЕДНЕВЕРХНЕНЕМЕЦКИЙ)")
    print("="*70)

    # 1. Строим унифицированный корпус
    sentences, metadata = build_unified_corpus('/Users/nora/src/Starostin/MHG')

    if len(sentences) == 0:
        print("❌ Не удалось извлечь предложения")
        return

    # Сохраняем унифицированный корпус
    print("\n💾 Сохранение унифицированного корпуса...")
    with open('/Users/nora/src/Starostin/mhg_unified.txt', 'w', encoding='utf-8') as f:
        for sent in sentences:
            f.write(sent + '\n')
    print(f"✓ Унифицированный корпус: mhg_unified.txt ({len(sentences)} предложений)")

    # 2. Извлекаем контексты tuon
    contexts = extract_tuon_contexts(sentences, window=10)

    if len(contexts) == 0:
        print("❌ Не найдено контекстов tuon")
        return

    # Частота форм
    form_dist = Counter([ctx['form'] for ctx in contexts])
    print(f"\n📊 Частотность форм:")
    for form, count in form_dist.most_common():
        print(f"  {form:15s}: {count}")

    # Категоризация
    categories = categorize_by_markers(contexts)

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
    print("✅ АНАЛИЗ ЗАВЕРШЁН!")
    print("="*70)
    print("\nСозданные файлы:")
    print("  • mhg_unified.txt - унифицированный корпус")
    print("  • tuon_full_analysis.png - визуализация")
    print("  • tuon_full_results.txt - результаты")

if __name__ == '__main__':
    main()
