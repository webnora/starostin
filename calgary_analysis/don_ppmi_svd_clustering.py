#!/usr/bin/env python3
"""
PPMI + SVD + HDBSCAN Analysis for Old English Verb 'dōn'
==========================================================

This script performs semantic analysis focusing on CONTEXTUAL PROFILES
of the verb 'dōn' across all its forms, treating different grammatical
forms as instances of the same lexical item.

Author: Computational Linguistics Analysis
Date: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score
import umap
import hdbscan
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)


class ContextualPPMIAnalyzer:
    """
    Analyzer focusing on CONTEXTUAL PROFILES of 'dōn' across all forms.
    """

    def __init__(self, min_word_freq=2, n_components=200, min_cluster_size=15):
        """
        Initialize analyzer.

        Args:
            min_word_freq: Minimum frequency for vocabulary inclusion
            n_components: Number of SVD components
            min_cluster_size: Minimum cluster size for HDBSCAN
        """
        self.min_word_freq = min_word_freq
        self.n_components = n_components
        self.min_cluster_size = min_cluster_size
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.contexts = []
        self.context_vectors = None

    def load_and_preprocess(self, filepath):
        """Load contexts and extract the target word position."""
        print("=" * 70)
        print("STEP 1: Loading and Preprocessing Contexts")
        print("=" * 70)

        contexts = []
        context_labels = []  # Store which form of 'dōn' appears

        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    tokens = line.lower().split()
                    contexts.append(tokens)
                    # Try to identify which form of 'dōn' is in this context
                    don_forms = ['don', 'do', 'dyde', 'dydon', 'gedon', 'doð', 'deð', 'dest', 'de']
                    found_form = None
                    for form in don_forms:
                        if form in tokens:
                            found_form = form
                            break
                    context_labels.append(found_form if found_form else 'unknown')

        self.contexts = contexts
        self.context_labels = context_labels

        print(f"✓ Loaded {len(contexts)} contexts")

        # Count forms
        form_counts = Counter(context_labels)
        print(f"✓ Form distribution:")
        for form, count in sorted(form_counts.items(), key=lambda x: -x[1]):
            print(f"    {form:10s}: {count:3d} contexts")

        self._build_vocabulary()
        return contexts

    def _build_vocabulary(self):
        """Build vocabulary from contexts."""
        word_counts = Counter()
        for context in self.contexts:
            word_counts.update(context)

        vocab_words = [word for word, count in word_counts.items()
                      if count >= self.min_word_freq]

        self.word_to_idx = {word: idx for idx, word in enumerate(sorted(vocab_words))}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}

        print(f"✓ Vocabulary size: {len(self.word_to_idx)} words (min_freq={self.min_word_freq})")

    def build_context_vectors(self):
        """
        Build vectors for each CONTEXT (not each word).
        Each context is represented as a bag-of-words vector.
        """
        print("\n" + "=" * 70)
        print("STEP 2: Building Context Vectors")
        print("=" * 70)

        vocab_size = len(self.word_to_idx)
        num_contexts = len(self.contexts)

        # Each context = bag of words
        context_matrix = np.zeros((num_contexts, vocab_size), dtype=np.float32)

        for i, context in enumerate(self.contexts):
            for word in context:
                if word in self.word_to_idx:
                    context_matrix[i, self.word_to_idx[word]] += 1

        # Apply TF-IDF weighting
        # TF: term frequency (already in matrix)
        # IDF: inverse document frequency
        doc_freq = np.sum(context_matrix > 0, axis=0)
        idf = np.log(num_contexts / (doc_freq + 1))

        tfidf_matrix = context_matrix * idf

        print(f"✓ Context-TFIDF matrix shape: {tfidf_matrix.shape}")
        print(f"✓ Non-zero entries: {np.count_nonzero(tfidf_matrix)}")

        # Apply SVD to reduce dimensionality
        svd = TruncatedSVD(n_components=self.n_components, random_state=42)
        context_vectors = svd.fit_transform(tfidf_matrix)

        # Normalize
        context_vectors = normalize(context_vectors, norm='l2')

        self.context_vectors = context_vectors
        self.tfidf_matrix = tfidf_matrix

        explained_variance = sum(svd.explained_variance_ratio_)
        print(f"✓ SVD output shape: {context_vectors.shape}")
        print(f"✓ Explained variance: {explained_variance:.2%}")

        return context_vectors

    def cluster_contexts(self):
        """Apply HDBSCAN clustering to context vectors."""
        print("\n" + "=" * 70)
        print("STEP 3: HDBSCAN Clustering of Contexts")
        print("=" * 70)

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            metric='euclidean',  # Vectors are already L2-normalized
            cluster_selection_method='eom',
            prediction_data=True
        )

        clusters = clusterer.fit_predict(self.context_vectors)

        # Count clusters
        unique_clusters = set(clusters)
        n_clusters = len(unique_clusters) - (1 if -1 in unique_clusters else 0)
        n_noise = list(clusters).count(-1)

        print(f"✓ Number of clusters: {n_clusters}")
        print(f"✓ Noise points: {n_noise} ({n_noise/len(clusters)*100:.1f}%)")

        # Silhouette score (only if we have clusters)
        if n_clusters > 0 and len(set(clusters)) > 1:
            mask = clusters != -1
            if np.sum(mask) > 1:
                score = silhouette_score(self.context_vectors[mask], clusters[mask], metric='euclidean')
                print(f"✓ Silhouette score: {score:.3f}")

        self.clusters = clusters
        self.clusterer = clusterer

        return clusters, n_clusters, n_noise

    def analyze_clusters(self):
        """Analyze the semantic content of each cluster."""
        print("\n" + "=" * 70)
        print("STEP 4: Cluster Analysis")
        print("=" * 70)

        unique_clusters = sorted(set(self.clusters))

        for cluster_id in unique_clusters:
            print(f"\n{'Cluster ' + str(cluster_id) if cluster_id >= 0 else 'Noise':-^70}")

            if cluster_id == -1:
                # Noise
                indices = np.where(self.clusters == -1)[0]
                print(f"Total noise contexts: {len(indices)} ({len(indices)/len(self.contexts)*100:.1f}%)")
            else:
                # Named cluster
                indices = np.where(self.clusters == cluster_id)[0]
                print(f"Total contexts: {len(indices)} ({len(indices)/len(self.contexts)*100:.1f}%)")

                # Get words that appear in these contexts
                word_freq = Counter()
                for idx in indices:
                    for word in self.contexts[idx]:
                        if word in self.word_to_idx:
                            word_freq[word] += 1

                # Top words in this cluster
                print(f"\nTop distinctive words:")
                for word, freq in word_freq.most_common(20):
                    # Skip the don forms themselves
                    if word not in ['don', 'do', 'dyde', 'dydon', 'gedon', 'doð', 'deð', 'dest', 'de']:
                        print(f"  {word:15s}: {freq:3d}")

                # Form distribution in this cluster
                form_dist = Counter([self.context_labels[i] for i in indices])
                print(f"\nForm distribution:")
                for form, count in sorted(form_dist.items(), key=lambda x: -x[1]):
                    print(f"  {form:10s}: {count:2d} ({count/len(indices)*100:.1f}%)")

                # Sample contexts
                print(f"\nSample contexts:")
                sample_indices = np.random.choice(indices, min(3, len(indices)), replace=False)
                for i, idx in enumerate(sample_indices, 1):
                    context_str = ' '.join(self.contexts[idx])
                    print(f"  {i}. {context_str[:100]}...")

    def visualize_clusters(self, output_path='don_ppmi_svd_clusters.png'):
        """Visualize clusters using UMAP."""
        print("\n" + "=" * 70)
        print("STEP 5: UMAP Visualization")
        print("=" * 70)

        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
        embeddings_2d = reducer.fit_transform(self.context_vectors)

        fig, ax = plt.subplots(figsize=(16, 12))

        # Color palette
        n_clusters = len(set(self.clusters)) - (1 if -1 in self.clusters else 0)
        colors = sns.color_palette("husl", n_clusters)

        # Plot noise in light gray
        noise_mask = self.clusters == -1
        if np.any(noise_mask):
            ax.scatter(embeddings_2d[noise_mask, 0], embeddings_2d[noise_mask, 1],
                      c='lightgray', s=30, alpha=0.5, label='Noise')

        # Plot clusters
        for cluster_id in sorted(set(self.clusters)):
            if cluster_id == -1:
                continue
            mask = self.clusters == cluster_id
            ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                      c=[colors[cluster_id]], s=80, alpha=0.7,
                      label=f'Cluster {cluster_id}', edgecolors='black', linewidths=0.5)

        ax.set_xlabel('UMAP 1', fontsize=12)
        ax.set_ylabel('UMAP 2', fontsize=12)
        ax.set_title(f'Semantic Clusters of Old English "dōn" (Calgary Corpus)\nPPMI + SVD + HDBSCAN | {len(self.contexts)} contexts',
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Visualization saved to: {output_path}")

        return embeddings_2d

    def interpret_semantic_profiles(self):
        """Provide interpretation of semantic profiles."""
        print("\n" + "=" * 70)
        print("STEP 6: Semantic Profile Interpretation")
        print("=" * 70)

        unique_clusters = sorted([c for c in set(self.clusters) if c >= 0])

        if not unique_clusters:
            print("No significant clusters found. The contexts appear to form")
            print("a continuous semantic space without clear boundaries.")
            return

        print("\n📊 IDENTIFIED SEMANTIC PROFILES:\n")

        profiles = []

        for cluster_id in unique_clusters:
            indices = np.where(self.clusters == cluster_id)[0]

            # Get top words
            word_freq = Counter()
            for idx in indices:
                for word in self.contexts[idx]:
                    if word not in ['don', 'do', 'dyde', 'dydon', 'gedon', 'doð', 'deð', 'dest', 'de']:
                        word_freq[word] += 1

            top_words = [w for w, _ in word_freq.most_common(15)]

            # Get form distribution
            form_dist = Counter([self.context_labels[i] for i in indices])

            profiles.append({
                'id': cluster_id,
                'size': len(indices),
                'percentage': len(indices) / len(self.contexts) * 100,
                'top_words': top_words,
                'forms': dict(form_dist)
            })

        # Print profiles
        for p in profiles:
            print(f"{'='*70}")
            print(f"PROFILE {p['id']}: {p['size']} contexts ({p['percentage']:.1f}%)")
            print(f"{'='*70}")
            print(f"Key words: {', '.join(p['top_words'][:10])}")
            print(f"Forms: ", end='')
            for form, count in sorted(p['forms'].items(), key=lambda x: -x[1]):
                print(f"{form}({count}) ", end='')
            print('\n')


def main():
    """Main analysis pipeline."""
    print("\n" + "=" * 70)
    print("CONTEXTUAL PPMI + SVD + HDBSCAN ANALYSIS")
    print("Old English 'dōn' - Calgary Corpus")
    print("=" * 70)

    # Initialize
    analyzer = ContextualPPMIAnalyzer(
        min_word_freq=2,
        n_components=200,
        min_cluster_size=15
    )

    # Load data
    analyzer.load_and_preprocess('don_clean.txt')

    # Build context vectors
    analyzer.build_context_vectors()

    # Cluster contexts
    analyzer.cluster_contexts()

    # Analyze clusters
    analyzer.analyze_clusters()

    # Visualize
    analyzer.visualize_clusters()

    # Interpret
    analyzer.interpret_semantic_profiles()

    print("\n" + "=" * 70)
    print("✅ Analysis complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
