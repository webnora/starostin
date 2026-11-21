#!/usr/bin/env python3
"""
PPMI + SVD Analysis for Old English Verb 'dōn'
================================================

This script performs semantic analysis of the Old English verb 'dōn' using:
1. Positive Pointwise Mutual Information (PPMI) for word co-occurrence
2. Singular Value Decomposition (SVD) for dimensionality reduction
3. UMAP for visualization

Author: Computational Linguistics Analysis
Date: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
import umap
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


class PPMI_SVD_Analyzer:
    """
    A class for performing PPMI + SVD analysis on word contexts.
    """

    def __init__(self, min_word_freq=2, n_components=200):
        """
        Initialize the analyzer.

        Args:
            min_word_freq: Minimum frequency for a word to be included in vocabulary
            n_components: Number of SVD components to keep
        """
        self.min_word_freq = min_word_freq
        self.n_components = n_components
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.ppmi_matrix = None
        self.svd_matrix = None
        self.contexts = []

    def load_and_preprocess(self, filepath):
        """
        Step 1: Load and preprocess contexts.

        Args:
            filepath: Path to the text file with contexts (one per line)

        Returns:
            List of tokenized contexts
        """
        print("=" * 70)
        print("STEP 1: Loading and Preprocessing Data")
        print("=" * 70)

        contexts = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    # Tokenize: split by whitespace
                    tokens = line.lower().split()
                    contexts.append(tokens)

        self.contexts = contexts

        print(f"✓ Loaded {len(contexts)} contexts")
        print(f"✓ Example context: {' '.join(contexts[0])}")

        # Build vocabulary
        self._build_vocabulary()

        return contexts

    def _build_vocabulary(self):
        """Build vocabulary from contexts, filtering by minimum frequency."""
        word_counts = Counter()

        # Count word occurrences
        for context in self.contexts:
            word_counts.update(context)

        # Filter by minimum frequency
        vocab_words = [word for word, count in word_counts.items()
                      if count >= self.min_word_freq]

        # Create word-index mappings
        self.word_to_idx = {word: idx for idx, word in enumerate(sorted(vocab_words))}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}

        print(f"✓ Vocabulary size: {len(self.word_to_idx)} words (min_freq={self.min_word_freq})")
        print(f"✓ Total tokens in corpus: {sum(word_counts.values())}")

    def build_cooccurrence_matrix(self):
        """
        Step 2: Build word co-occurrence matrix.

        The matrix size is V x V where V is vocabulary size.
        Entry [i][j] = how many times word i appears in the same context as word j.
        """
        print("\n" + "=" * 70)
        print("STEP 2: Building Co-occurrence Matrix")
        print("=" * 70)

        vocab_size = len(self.word_to_idx)
        cooccurrence = np.zeros((vocab_size, vocab_size), dtype=np.float32)

        # For each context, count word co-occurrences
        for context in self.contexts:
            # Get unique words in this context
            words_in_context = [self.word_to_idx[w] for w in context
                               if w in self.word_to_idx]
            words_in_context = list(set(words_in_context))  # Unique words only

            # Increment co-occurrence for all pairs
            for i in words_in_context:
                for j in words_in_context:
                    cooccurrence[i, j] += 1

        total_pairs = np.sum(cooccurrence)

        print(f"✓ Co-occurrence matrix shape: {cooccurrence.shape}")
        print(f"✓ Total co-occurrence pairs: {int(total_pairs)}")
        print(f"✓ Non-zero entries: {np.count_nonzero(cooccurrence)}")

        return cooccurrence, total_pairs

    def compute_ppmi(self, cooccurrence, total_pairs):
        """
        Step 2 (continued): Compute PPMI matrix.

        PPMI(w, c) = max(log2(P(w,c) / (P(w) * P(c))), 0)

        where:
        - P(w,c) = count(w,c) / total_pairs
        - P(w) = count(w) / total_words
        - P(c) = count(c) / total_words
        """
        print("\n" + "=" * 70)
        print("STEP 2 (continued): Computing PPMI Matrix")
        print("=" * 70)

        vocab_size = len(self.word_to_idx)

        # Calculate word probabilities
        word_counts = np.sum(cooccurrence, axis=1)  # Sum over columns
        total_words = np.sum(word_counts)
        p_word = word_counts / total_words

        # Calculate PPMI
        ppmi = np.zeros_like(cooccurrence)

        for i in range(vocab_size):
            for j in range(vocab_size):
                if cooccurrence[i, j] > 0:
                    # P(w, c) = count(w, c) / total_pairs
                    p_joint = cooccurrence[i, j] / total_pairs

                    # P(w) * P(c)
                    p_independent = p_word[i] * p_word[j]

                    # PMI = log2(P(w,c) / (P(w) * P(c)))
                    pmi = np.log2(p_joint / p_independent)

                    # PPMI = max(PMI, 0)
                    ppmi[i, j] = max(pmi, 0)

        self.ppmi_matrix = ppmi

        print(f"✓ PPMI matrix shape: {ppmi.shape}")
        print(f"✓ Non-zero PPMI entries: {np.count_nonzero(ppmi)}")
        print(f"✓ PPMI sparsity: {1 - np.count_nonzero(ppmi) / ppmi.size:.2%}")
        print(f"✓ PPMI max value: {np.max(ppmi):.4f}")
        print(f"✓ PPMI mean (non-zero): {np.mean(ppmi[ppmi > 0]):.4f}")

        return ppmi

    def apply_svd(self):
        """
        Step 3: Apply SVD to reduce dimensionality.

        Uses TruncatedSVD for efficient computation on sparse matrices.
        """
        print("\n" + "=" * 70)
        print("STEP 3: Applying SVD for Dimensionality Reduction")
        print("=" * 70)

        # Use TruncatedSVD for efficiency
        svd = TruncatedSVD(n_components=self.n_components, random_state=42)
        svd_matrix = svd.fit_transform(self.ppmi_matrix)

        # Normalize vectors
        svd_matrix = normalize(svd_matrix, norm='l2')

        self.svd_matrix = svd_matrix

        explained_variance = sum(svd.explained_variance_ratio_)
        print(f"✓ SVD output shape: {svd_matrix.shape}")
        print(f"✓ Explained variance ratio: {explained_variance:.2%}")
        print(f"✓ Vector dimensions: {self.n_components}")

        return svd_matrix

    def visualize_embeddings(self, output_path='don_ppmi_svd_umap.png'):
        """
        Step 4: Visualize word embeddings using UMAP.

        Creates a 2D visualization of the word vectors.
        """
        print("\n" + "=" * 70)
        print("STEP 4: UMAP Visualization")
        print("=" * 70)

        # Apply UMAP for 2D visualization
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
        embeddings_2d = reducer.fit_transform(self.svd_matrix)

        # Find 'don' forms in vocabulary
        don_forms = ['don', 'do', 'dyde', 'dydon', 'gedon', 'doð', 'deð', 'dest']
        don_indices = []
        for form in don_forms:
            if form in self.word_to_idx:
                don_indices.append(self.word_to_idx[form])

        # Create plot
        fig, ax = plt.subplots(figsize=(14, 10))

        # Plot all words in light gray
        ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                  c='lightgray', alpha=0.3, s=20, label='Other words')

        # Plot 'don' forms in red
        if don_indices:
            don_embeddings = embeddings_2d[don_indices]
            ax.scatter(don_embeddings[:, 0], don_embeddings[:, 1],
                      c='red', s=150, alpha=0.8, edgecolors='black',
                      linewidths=2, label='dōn forms')

            # Annotate 'don' forms
            for idx in don_indices:
                word = self.idx_to_word[idx]
                ax.annotate(word, (embeddings_2d[idx, 0], embeddings_2d[idx, 1]),
                           fontsize=10, fontweight='bold',
                           xytext=(5, 5), textcoords='offset points')

        ax.set_xlabel('UMAP 1', fontsize=12)
        ax.set_ylabel('UMAP 2', fontsize=12)
        ax.set_title('PPMI + SVD Word Embeddings: Old English "dōn"\nCalgary Corpus (226 contexts)',
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Visualization saved to: {output_path}")

        return embeddings_2d, don_indices

    def find_similar_words(self, target_word, top_n=10):
        """
        Find words most similar to the target word using cosine similarity.

        Args:
            target_word: Word to find similarities for
            top_n: Number of similar words to return

        Returns:
            List of (word, similarity_score) tuples
        """
        if target_word not in self.word_to_idx:
            print(f"Word '{target_word}' not in vocabulary")
            return []

        target_idx = self.word_to_idx[target_word]
        target_vector = self.svd_matrix[target_idx]

        # Compute cosine similarity with all words
        similarities = np.dot(self.svd_matrix, target_vector)

        # Get top similar words (excluding the target itself)
        similar_indices = np.argsort(similarities)[::-1][1:top_n+1]

        results = [(self.idx_to_word[idx], similarities[idx])
                  for idx in similar_indices]

        return results

    def print_interpretation(self):
        """
        Step 5: Print interpretation of results.
        """
        print("\n" + "=" * 70)
        print("STEP 5: Interpretation of Results")
        print("=" * 70)

        print("\n📊 COMPARISON WITH TF-IDF + HDBSCAN:")
        print("-" * 70)
        print("Method           | Noise | Clusters | Silhouette")
        print("-" * 70)
        print("TF-IDF + HDBSCAN | 76.6% |    2     |   0.28")
        print("PPMI + SVD       |  N/A  |  N/A    |  N/A  ")
        print("\nNote: PPMI + SVD is a continuous embedding method, not a clustering")
        print("method. It captures semantic relationships in a vector space without")
        print("explicitly assigning items to discrete clusters.")

        print("\n🔍 KEY INSIGHTS:")
        print("-" * 70)
        print("1. PPMI captures word co-occurrence patterns better than TF-IDF for")
        print("   semantic analysis, as it measures the strength of association")
        print("   between words relative to chance.")
        print()
        print("2. SVD reduces dimensionality while preserving the most important")
        print("   semantic relationships, similar to how Word2Vec works.")
        print()
        print("3. The high noise rate (76.6%) in TF-IDF + HDBSCAN suggests that:")
        print("   - Many usages of 'dōn' are semantically distinct")
        print("   - TF-IDF may not capture the subtle semantic differences well")
        print("   - The verb is highly polysemous with many nuanced meanings")
        print()
        print("4. PPMI + SVD is better suited for capturing these nuances because")
        print("   it represents each usage as a point in a continuous semantic space")

        print("\n💡 NEXT STEPS:")
        print("-" * 70)
        print("- Apply HDBSCAN clustering on the SVD vectors for comparison")
        print("- Compare most similar words for different forms of 'dōn'")
        print("- Analyze the semantic space to identify meaning clusters")


def main():
    """
    Main function to run the PPMI + SVD analysis.
    """
    print("\n" + "=" * 70)
    print("PPMI + SVD ANALYSIS: Old English 'dōn' (Calgary Corpus)")
    print("=" * 70)

    # Initialize analyzer
    analyzer = PPMI_SVD_Analyzer(min_word_freq=2, n_components=200)

    # Step 1: Load and preprocess data
    analyzer.load_and_preprocess('don_clean.txt')

    # Step 2: Build co-occurrence matrix and compute PPMI
    cooccurrence, total_pairs = analyzer.build_cooccurrence_matrix()
    analyzer.compute_ppmi(cooccurrence, total_pairs)

    # Step 3: Apply SVD
    analyzer.apply_svd()

    # Step 4: Visualize with UMAP
    embeddings_2d, don_indices = analyzer.visualize_embeddings()

    # Step 5: Find similar words for different forms
    print("\n" + "=" * 70)
    print("SEMANTIC SIMILARITY: Forms of 'dōn'")
    print("=" * 70)

    don_forms = ['don', 'do', 'dyde', 'gedon', 'doð', 'deð']
    for form in don_forms:
        if form in analyzer.word_to_idx:
            print(f"\n📌 Words similar to '{form}':")
            similar = analyzer.find_similar_words(form, top_n=5)
            for word, sim in similar:
                print(f"   {word:15s} (similarity: {sim:.4f})")

    # Print interpretation
    analyzer.print_interpretation()

    print("\n" + "=" * 70)
    print("✅ Analysis complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
