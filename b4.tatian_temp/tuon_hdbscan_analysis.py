#!/usr/bin/env python3
"""
Анализ лексемы tuon (делать) в корпусе Татиана
TF-IDF векторизация + HDBSCAN кластеризация + UMAP визуализация
"""

import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
import hdbscan
import umap
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

class TatianTuonAnalyzer:
    def __init__(self, corpus_path: str):
        self.corpus_path = Path(corpus_path)
        self.all_contexts = []
        self.tuon_contexts = []

    def extract_ohg_text(self, xml_file: Path) -> Tuple[List[str], List[str]]:
        """Извлекает древневерхненемецкий текст из XML файла ExMARaLDA"""
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()

            # Находим tier с OHG текстом
            words = []
            for tier in root.findall(".//tier"):
                category = tier.get("category", "")
                display_name = tier.get("display-name", "")

                if category == "words" and display_name == "ahd":
                    for event in tier.findall("event"):
                        text = event.text
                        if text and text.strip():
                            words.append(text.strip())
                    break

            return words, []
        except Exception as e:
            print(f"Error parsing {xml_file}: {e}")
            return [], []

    def extract_context_window(self, words: List[str], target_idx: int,
                               window_size: int = 5) -> str:
        """Извлекает контекстное окно вокруг целевого слова"""
        start = max(0, target_idx - window_size)
        end = min(len(words), target_idx + window_size + 1)
        context = words[start:end]
        return " ".join(context)

    def is_tuon_form(self, word: str) -> bool:
        """Проверяет, является ли слово формой глагола tuon"""
        tuon_forms = {
            'tuon', 'tuo', 'tuot', 'tuont', 'tuos', 'tuome', 'tuom',
            'tat', 'tati', 'tatin', 'tatun', 'tān', 'tan', 'tig',
            'tuonteru', 'tuonte', 'tuontan', 'tuomu', 'tuomon', 'tuomit', 'tuoment'
        }
        return word.lower() in tuon_forms

    def load_corpus(self):
        """Загружает корпус и извлекает контексты с tuon"""
        xml_files = list(self.corpus_path.glob("**/*.xml"))

        print(f"Found {len(xml_files)} XML files")

        tuon_count = 0
        for xml_file in xml_files:
            words, _ = self.extract_ohg_text(xml_file)

            if not words:
                continue

            for idx, word in enumerate(words):
                if self.is_tuon_form(word):
                    context = self.extract_context_window(words, idx)
                    self.tuon_contexts.append({
                        'word': word,
                        'context': context,
                        'file': xml_file.name
                    })
                    tuon_count += 1

        print(f"Found {tuon_count} occurrences of tuon")
        return self.tuon_contexts

    def preprocess_text(self, text: str) -> str:
        """Предварительная обработка текста"""
        # Приводим к нижнему регистру
        text = text.lower()
        # Удаляем лишние пробелы
        text = ' '.join(text.split())
        return text

    def create_tfidf_matrix(self, contexts: List[str]) -> Tuple[np.ndarray, List[str]]:
        """Создает TF-IDF матрицу"""
        # Предобработка
        processed_contexts = [self.preprocess_text(c) for c in contexts]

        # Параметры векторизации для древнегерманских языков
        vectorizer = TfidfVectorizer(
            min_df=1,  # Минимальная документная частота
            max_df=0.9,  # Максимальная документная частота
            ngram_range=(1, 2),  # Учитываем униграммы и биграммы
            token_pattern=r'(?u)\b\w+\b',  # Паттерн для слов
            lowercase=True
        )

        tfidf_matrix = vectorizer.fit_transform(processed_contexts)
        feature_names = vectorizer.get_feature_names_out()

        print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
        print(f"Number of unique words: {len(feature_names)}")

        return tfidf_matrix.toarray(), feature_names

    def cluster_hdbscan(self, embeddings: np.ndarray,
                       min_cluster_size: int = 3) -> Tuple[np.ndarray, hdbscan.HDBSCAN]:
        """Кластеризация HDBSCAN"""
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=1,
            metric='euclidean',
            cluster_selection_method='eom',
            prediction_data=True
        )

        labels = clusterer.fit_predict(embeddings)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)

        print(f"\nHDBSCAN Results:")
        print(f"  Number of clusters: {n_clusters}")
        print(f"  Number of noise points: {n_noise}")

        return labels, clusterer

    def visualize_umap(self, embeddings: np.ndarray, labels: np.ndarray,
                       contexts: List[Dict], output_path: str):
        """Визуализация UMAP"""
        reducer = umap.UMAP(
            n_neighbors=min(15, len(embeddings) - 1),
            min_dist=0.1,
            metric='euclidean',
            random_state=42
        )

        embedding_2d = reducer.fit_transform(embeddings)

        plt.figure(figsize=(14, 10))

        # Цветовая схема
        unique_labels = set(labels)
        colors = plt.cm.tab20(np.linspace(0, 1, 20))

        for idx, label in enumerate(unique_labels):
            if label == -1:
                color = 'lightgray'
                marker = 'o'
                label_name = 'Noise'
                alpha = 0.4
            else:
                color = colors[label % 20]
                marker = 'o'
                label_name = f'Cluster {label}'
                alpha = 0.7

            mask = labels == label
            plt.scatter(
                embedding_2d[mask, 0],
                embedding_2d[mask, 1],
                c=[color],
                label=label_name,
                marker=marker,
                s=100,
                alpha=alpha,
                edgecolors='black',
                linewidths=0.5
            )

        # Добавляем номера точек
        for i, (x, y) in enumerate(embedding_2d):
            plt.annotate(
                str(i + 1),
                (x, y),
                fontsize=7,
                alpha=0.6
            )

        plt.title('UMAP Visualization of tuon (делать) Contexts\nTatian Corpus',
                 fontsize=14, fontweight='bold')
        plt.xlabel('UMAP 1', fontsize=12)
        plt.ylabel('UMAP 2', fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nVisualization saved to: {output_path}")
        plt.close()

    def analyze_clusters(self, contexts: List[Dict], labels: np.ndarray,
                        feature_names: List[str], tfidf_matrix: np.ndarray):
        """Анализ кластеров"""
        print("\n" + "="*70)
        print("CLUSTER ANALYSIS")
        print("="*70)

        unique_labels = sorted(set(labels))

        for label in unique_labels:
            if label == -1:
                continue

            cluster_indices = np.where(labels == label)[0]
            cluster_contexts = [contexts[i] for i in cluster_indices]

            print(f"\n{'='*70}")
            print(f"CLUSTER {label} ({len(cluster_indices)} contexts)")
            print(f"{'='*70}")

            # Показываем контексты в кластере
            for i, idx in enumerate(cluster_indices[:10], 1):
                ctx = contexts[idx]
                print(f"{i}. [{ctx['word']}] {ctx['context']}")

            if len(cluster_indices) > 10:
                print(f"... and {len(cluster_indices) - 10} more")

    def save_results(self, contexts: List[Dict], labels: np.ndarray,
                    output_path: str):
        """Сохраняет результаты в файл"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("tuon (делать) - Tatian Corpus Analysis Results\n")
            f.write("="*70 + "\n\n")

            unique_labels = sorted(set(labels))

            for label in unique_labels:
                cluster_name = f"Cluster {label}" if label != -1 else "Noise"
                f.write(f"\n{cluster_name}\n")
                f.write("-"*70 + "\n")

                cluster_indices = np.where(labels == label)[0]

                for i, idx in enumerate(cluster_indices, 1):
                    ctx = contexts[idx]
                    f.write(f"{i}. [{ctx['word']}] {ctx['context']}\n")
                    f.write(f"   File: {ctx['file']}\n")

                f.write(f"\nTotal: {len(cluster_indices)} contexts\n\n")

        print(f"Results saved to: {output_path}")

    def run_analysis(self):
        """Запускает полный анализ"""
        print("="*70)
        print("TUON ANALYSIS - TATIAN CORPUS")
        print("="*70)

        # Загрузка корпуса
        print("\n1. Loading corpus...")
        contexts = self.load_corpus()

        if len(contexts) < 5:
            print("ERROR: Not enough tuon occurrences found!")
            return

        context_texts = [ctx['context'] for ctx in contexts]

        # TF-IDF векторизация
        print("\n2. Creating TF-IDF matrix...")
        tfidf_matrix, feature_names = self.create_tfidf_matrix(context_texts)

        # HDBSCAN кластеризация
        print("\n3. HDBSCAN clustering...")
        min_cluster_size = max(2, len(contexts) // 10)
        labels, clusterer = self.cluster_hdbscan(tfidf_matrix, min_cluster_size)

        # Визуализация UMAP
        print("\n4. UMAP visualization...")
        output_dir = Path("/Users/nora/src/Starostin/b4.tatian")
        self.visualize_umap(tfidf_matrix, labels, contexts,
                           output_dir / "tuon_hdbscan_umap.png")

        # Анализ кластеров
        print("\n5. Analyzing clusters...")
        self.analyze_clusters(contexts, labels, feature_names, tfidf_matrix)

        # Сохранение результатов
        print("\n6. Saving results...")
        self.save_results(contexts, labels,
                         output_dir / "tuon_hdbscan_results.txt")

        print("\n" + "="*70)
        print("ANALYSIS COMPLETE!")
        print("="*70)


def main():
    corpus_path = "/Users/nora/src/Starostin/b4.tatian_temp/exmaralda/b4.tatian"

    analyzer = TatianTuonAnalyzer(corpus_path)
    analyzer.run_analysis()


if __name__ == "__main__":
    main()
