#!/usr/bin/env python3
"""
True PPMI + SVD Analysis for Old English Verb 'dōn'
====================================================

This script builds word-word co-occurrence matrices,
computes PPMI, applies SVD, and then projects CONTEXTS
into the semantic space for clustering.

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
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)


class TruePPMIAnalyzer:
    """
    True PPMI analyzer: builds word-word co-occurrence,
    computes PPMI, applies SVD, projects contexts.
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

    def load_and_preprocess(self, filepath):
        """Load contexts."""
        print("=" * 70)
        print("STEP 1: Loading and Preprocessing Contexts")
        print("=" * 70)

        contexts = []
        context_labels = []

        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    tokens = line.lower().split()
                    contexts.append(tokens)
                    # Identify which form of 'dōn'
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

    def build_ppmi_matrix(self):
        """
        Build word-word co-occurrence matrix and compute PPMI.
        """
        print("\n" + "=" * 70)
        print("STEP 2: Building PPMI Matrix")
        print("=" * 70)

        vocab_size = len(self.word_to_idx)
        cooccurrence = np.zeros((vocab_size, vocab_size), dtype=np.float32)

        # Build co-occurrence matrix (words co-occurring within same context)
        for context in self.contexts:
            words_in_context = [self.word_to_idx[w] for w in context
                               if w in self.word_to_idx]
            words_in_context = list(set(words_in_context))  # Unique words

            for i in words_in_context:
                for j in words_in_context:
                    cooccurrence[i, j] += 1

        total_pairs = np.sum(cooccurrence)

        print(f"✓ Co-occurrence matrix shape: {cooccurrence.shape}")
        print(f"✓ Total co-occurrence pairs: {int(total_pairs)}")
        print(f"✓ Non-zero entries: {np.count_nonzero(cooccurrence)}")

        # Compute PPMI
        print("\nComputing PPMI...")

        vocab_size = len(self.word_to_idx)

        # Calculate word probabilities
        word_counts = np.sum(cooccurrence, axis=1)
        total_words = np.sum(word_counts)
        p_word = word_counts / total_words

        # PPMI matrix
        ppmi = np.zeros_like(cooccurrence)

        for i in range(vocab_size):
            for j in range(vocab_size):
                if cooccurrence[i, j] > 0:
                    p_joint = cooccurrence[i, j] / total_pairs
                    p_independent = p_word[i] * p_word[j]
                    pmi = np.log2(p_joint / p_independent)
                    ppmi[i, j] = max(pmi, 0)

        self.ppmi_matrix = ppmi

        print(f"✓ PPMI matrix shape: {ppmi.shape}")
        print(f"✓ Non-zero PPMI entries: {np.count_nonzero(ppmi)}")
        print(f"✓ PPMI sparsity: {1 - np.count_nonzero(ppmi) / ppmi.size:.2%}")
        print(f"✓ PPMI max value: {np.max(ppmi):.4f}")

        return ppmi

    def apply_svd(self):
        """
        Apply SVD to PPMI matrix to get word embeddings.
        """
        print("\n" + "=" * 70)
        print("STEP 3: Applying SVD to PPMI Matrix")
        print("=" * 70)

        svd = TruncatedSVD(n_components=self.n_components, random_state=42)
        word_embeddings = svd.fit_transform(self.ppmi_matrix)

        # Normalize
        word_embeddings = normalize(word_embeddings, norm='l2')

        self.word_embeddings = word_embeddings

        explained_variance = sum(svd.explained_variance_ratio_)
        print(f"✓ Word embeddings shape: {word_embeddings.shape}")
        print(f"✓ Explained variance: {explained_variance:.2%}")

        return word_embeddings

    def project_contexts_to_embedding_space(self):
        """
        Project each context into the word embedding space.
        Each context is represented as the average of its word vectors.
        """
        print("\n" + "=" * 70)
        print("STEP 4: Projecting Contexts to Embedding Space")
        print("=" * 70)

        context_vectors = []

        for context in self.contexts:
            # Get word vectors for all words in context
            word_vectors = []
            for word in context:
                if word in self.word_to_idx:
                    word_idx = self.word_to_idx[word]
                    word_vectors.append(self.word_embeddings[word_idx])

            if word_vectors:
                # Average of word vectors (simple bag-of-vectors)
                context_vec = np.mean(word_vectors, axis=0)
            else:
                # Zero vector if no words in vocab
                context_vec = np.zeros(self.n_components)

            context_vectors.append(context_vec)

        self.context_vectors = np.array(context_vectors)

        print(f"✓ Context vectors shape: {self.context_vectors.shape}")

        return self.context_vectors

    def cluster_contexts(self):
        """Apply HDBSCAN clustering to context vectors."""
        print("\n" + "=" * 70)
        print("STEP 5: HDBSCAN Clustering of Contexts")
        print("=" * 70)

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            metric='euclidean',
            cluster_selection_method='eom',
            prediction_data=True
        )

        clusters = clusterer.fit_predict(self.context_vectors)

        unique_clusters = set(clusters)
        n_clusters = len(unique_clusters) - (1 if -1 in unique_clusters else 0)
        n_noise = list(clusters).count(-1)

        print(f"✓ Number of clusters: {n_clusters}")
        print(f"✓ Noise points: {n_noise} ({n_noise/len(clusters)*100:.1f}%)")

        # Silhouette score
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
        print("STEP 6: Cluster Analysis")
        print("=" * 70)

        unique_clusters = sorted(set(self.clusters))

        for cluster_id in unique_clusters:
            print(f"\n{'Cluster ' + str(cluster_id) if cluster_id >= 0 else 'Noise':-^70}")

            if cluster_id == -1:
                indices = np.where(self.clusters == -1)[0]
                print(f"Total noise contexts: {len(indices)} ({len(indices)/len(self.contexts)*100:.1f}%)")
            else:
                indices = np.where(self.clusters == cluster_id)[0]
                print(f"Total contexts: {len(indices)} ({len(indices)/len(self.contexts)*100:.1f}%)")

                # Get words in these contexts
                word_freq = Counter()
                for idx in indices:
                    for word in self.contexts[idx]:
                        if word in self.word_to_idx:
                            word_freq[word] += 1

                # Top words
                print(f"\nTop distinctive words:")
                don_forms = ['don', 'do', 'dyde', 'dydon', 'gedon', 'doð', 'deð', 'dest', 'de']
                for word, freq in word_freq.most_common(20):
                    if word not in don_forms:
                        print(f"  {word:15s}: {freq:3d}")

                # Form distribution
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

    def visualize_clusters(self, output_path='don_ppmi_true_clusters.png'):
        """Visualize clusters using UMAP."""
        print("\n" + "=" * 70)
        print("STEP 7: UMAP Visualization")
        print("=" * 70)

        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
        embeddings_2d = reducer.fit_transform(self.context_vectors)

        fig, ax = plt.subplots(figsize=(16, 12))

        # Color palette
        n_clusters = len(set(self.clusters)) - (1 if -1 in self.clusters else 0)
        colors = sns.color_palette("husl", n_clusters)

        # Plot noise
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
        ax.set_title(f'PPMI + SVD Analysis: Old English "dōn" (Calgary Corpus)\n{len(self.contexts)} contexts | True PPMI word embeddings',
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Visualization saved to: {output_path}")

        return embeddings_2d


def main():
    """Main analysis pipeline."""
    print("\n" + "=" * 70)
    print("TRUE PPMI + SVD + HDBSCAN ANALYSIS")
    print("Old English 'dōn' - Calgary Corpus")
    print("=" * 70)

    analyzer = TruePPMIAnalyzer(
        min_word_freq=2,
        n_components=200,
        min_cluster_size=15
    )

    # Load
    analyzer.load_and_preprocess('don_clean.txt')

    # PPMI
    analyzer.build_ppmi_matrix()

    # SVD
    analyzer.apply_svd()

    # Project contexts
    analyzer.project_contexts_to_embedding_space()

    # Cluster
    analyzer.cluster_contexts()

    # Analyze
    analyzer.analyze_clusters()

    # Visualize
    analyzer.visualize_clusters()

    print("\n" + "=" * 70)
    print("✅ Analysis complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
