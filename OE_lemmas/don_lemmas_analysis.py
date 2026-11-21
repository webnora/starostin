#!/usr/bin/env python3
"""
Анализ глагола DON в lemmas.txt (Древнеанглийский)
Извлечение и интерпретация контекстов
"""

import re
import random
from collections import Counter, defaultdict

def extract_don_contexts_from_lemmas(filepath, window=10):
    """Извлечение контекстов don из lemmas.txt"""

    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    contexts = []
    sent_num = 0

    # Регулярное выражение для don и родственных слов
    don_pattern = r'\bdon\b'

    for line_num, line in enumerate(lines, 1):
        line = line.strip()
        if not line or line.startswith('&'):
            continue

        sent_num += 1

        # Ищем don в строке
        if re.search(don_pattern, line, re.IGNORECASE):
            # Разбиваем на слова
            words = re.findall(r'[a-zA-Zàáæðéèíóòúýþÿ-]+[,.?!:;–]*', line)

            for word_idx, word in enumerate(words):
                clean_word = re.sub(r'[,.?!:;–-]', '', word)

                if re.match(don_pattern, clean_word, re.IGNORECASE):
                    form = clean_word

                    before_words = [re.sub(r'[,.?!:;–-]', '', w).lower() for w in words[:word_idx]]
                    after_words = [re.sub(r'[,.?!:;–-]', '', w).lower() for w in words[word_idx+1:]]

                    contexts.append({
                        'form': form,
                        'before': before_words[-window:],
                        'after': after_words[:window],
                        'sentence': line,
                        'line_num': line_num,
                        'sent_num': sent_num
                    })
                    break  # Только первое вхождение в строке

    return contexts

def classify_contexts(contexts):
    """Классификация контекстов по категориям"""

    categories = {
        'divine': ['dryhten', 'god', 'halig', 'heofon', ' crist', 'engel', 'heaven'],
        'teaching': ['lar', 'secgan', 'spelling', 'bodian', 'cyðan', ' læran'],
        'command': ['biddan', 'bebeodan', 'hatan', 'scyran', 'swarian'],
        'good_deeds': ['riht', 'soð', 'fremian', 'gife', 'mildheortnes'],
        'evil_deeds': ['yfel', 'facen', 'swica', 'leas', 'unriht'],
        'battle': ['siger', 'winn', 'guð', 'wæpen', 'feoht', 'here'],
        'comparison': ['swa', 'swilce', 'swa þe']
    }

    categorized = defaultdict(list)
    uncategorized = []

    for ctx in contexts:
        text = ' '.join(ctx['before'] + ctx['after']).lower()
        cat_found = False

        for cat, markers in categories.items():
            if any(marker in text for marker in markers):
                categorized[cat].append(ctx)
                cat_found = True
                break

        if not cat_found:
            uncategorized.append(ctx)

    return categorized, uncategorized

def analyze_forms(contexts):
    """Анализ форм глагола don"""

    # Разбираем слова по типам
    infinitives = []
    past_tense = []
    imperatives = []
    participles = []
    other = []

    form_patterns = {
        'infinitive': r'\bdon$',
        'past': r'\bged(on|æd|id|en|en|on|un|on)\b',
        'imperative': r'\b(don|do|d[\ou][\waÿ]+)\b',
        'participle': r'\bgedon\b'
    }

    for ctx in contexts:
        form = ctx['form'].lower()
        sentence = ctx['sentence']

        # Проверяем различные формы
        if re.match(r'\bgedon\b', sentence):
            participles.append(ctx)
        elif re.search(r'\bd[oe]d[eo]\b|\bd[iy]d[eo]\b', sentence):
            past_tense.append(ctx)
        elif form == 'don':
            # Определяем по позиции или контексту
            if any(word in ctx['before'] for word in ['ic', 'þu', 'we', 'ge', 'skal']):
                if any(word in ctx['before'] for word in ['þu', 'skul', 'let']):
                    imperatives.append(ctx)
                else:
                    infinitives.append(ctx)
            else:
                other.append(ctx)
        else:
            other.append(ctx)

    return {
        'infinitive': infinitives,
        'past_tense': past_tense,
        'imperative': imperatives,
        'participle': participles,
        'other': other
    }

def main():
    print("="*70)
    print("🔬 АНАЛИЗ DON В LEMMAS.TXT")
    print("="*70)

    # 1. Извлечение контекстов
    contexts = extract_don_contexts_from_lemmas('/Users/nora/src/Starostin/lemmas.txt')

    print(f"\n✓ Найдено {len(contexts)} контекстов с don")

    # 2. Частотность форм
    form_dist = Counter([ctx['form'] for ctx in contexts])
    print(f"\n📊 Частотность форм:")
    for form, count in form_dist.most_common():
        pct = count / len(contexts) * 100
        print(f"  {form:15s}: {count:3d} ({pct:5.1f}%)")

    # 3. Анализ грамматических форм
    print(f"\n🔤 Грамматический анализ:")
    forms = analyze_forms(contexts)

    for form_type, ctxs in forms.items():
        if ctxs:
            print(f"\n  {form_type.upper()}: {len(ctxs)} ({len(ctxs)/len(contexts)*100:.1f}%)")

    # 4. Категоризация
    categorized, uncategorized = classify_contexts(contexts)

    print(f"\n📊 Ручная категоризация:")
    total_cat = sum(len(v) for v in categorized.values())
    print(f"  Категоризировано: {total_cat} ({total_cat/len(contexts)*100:.1f}%)")
    print(f"  Не категоризировано: {len(uncategorized)} ({len(uncategorized)/len(contexts)*100:.1f}%)")

    for cat, ctxs in sorted(categorized.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"    {cat:20s}: {len(ctxs):3d}")

    # 5. Примеры контекстов
    print(f"\n📝 20 случайных примеров:")
    sample = random.sample(contexts, min(20, len(contexts)))
    for i, ctx in enumerate(sample, 1):
        before = ' '.join(ctx['before'][-5:])
        after = ' '.join(ctx['after'][:5:])
        print(f"  [{i:2d}] ...{before} [{ctx['form']}] {after}...")

    # 6. Детально по категориям
    print(f"\n" + "="*70)
    print("🔍 ДЕТАЛЬНЫЙ АНАЛИЗ ПО КАТЕГОРИЯМ")
    print("="*70)

    for cat, ctxs in sorted(categorized.items(), key=lambda x: len(x[1]), reverse=True):
        if len(ctxs) == 0:
            continue

        print(f"\n{cat.upper()} ({len(ctxs)} контекстов):")
        print("-"*70)

        # Топ слова
        all_words = []
        for ctx in ctxs:
            all_words.extend([w.lower() for w in ctx['before'][-8:] + ctx['after'][:8]])

        word_dist = Counter(all_words)
        print("Топ-10 слов:")
        for word, count in word_dist.most_common(10):
            print(f"  {word:20s}: {count}")

        print(f"\nПримеры (первые 3):")
        for ctx in ctxs[:3]:
            before = ' '.join(ctx['before'][-5:])
            after = ' '.join(ctx['after'][:5:])
            print(f"  [{ctx['line_num']:4d}] ...{before} [{ctx['form']}] {after}...")

    # Сохранение результатов
    print(f"\n💾 Сохранение результатов...")
    with open('/Users/nora/src/Starostin/old_english/don_lemmas_results.txt', 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("АНАЛИЗ DON В LEMMAS.TXT\n")
        f.write("="*70 + "\n\n")

        f.write(f"Всего контекстов: {len(contexts)}\n\n")

        f.write("ЧАСТОТНОСТЬ ФОРМ:\n")
        for form, count in form_dist.most_common():
            pct = count / len(contexts) * 100
            f.write(f"  {form:15s}: {count:3d} ({pct:5.1f}%)\n")

        f.write(f"\n\nКАТЕГОРИЗАЦИЯ:\n")
        for cat, ctxs in sorted(categorized.items(), key=lambda x: len(x[1]), reverse=True):
            if len(ctxs) == 0:
                continue
            f.write(f"\n{cat.upper()} ({len(ctxs)}):\n")
            for ctx in ctxs:
                before = ' '.join(ctx['before'][-5:])
                after = ' '.join(ctx['after'][:5:])
                f.write(f"  [{ctx['line_num']:4d}] ...{before} [{ctx['form']}] {after}...\n")

    print("✓ Результаты сохранены: don_lemmas_results.txt")

if __name__ == '__main__':
    main()
