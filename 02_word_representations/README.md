# 02 — Word Representations

## Goal
Build and compare multiple word-level representations, from raw co-occurrence counts up to
pretrained embeddings, and inspect how well each captures semantic similarity.

## What was done
- Built sparse **word–word co-occurrence matrices** (rows = full vocabulary, columns = top-15k words)
  across three corpus variants (clean/partial/full) and three context window sizes (2, 5, 10).
- Converted co-occurrence counts into a **PPMI (Positive Pointwise Mutual Information)** matrix to
  downweight generically frequent words and highlight meaningful associations.
- Applied **Truncated SVD** to the PPMI matrix to obtain dense 300-dimensional word embeddings, and
  visualized a curated word set with PCA.
- Trained a **Word2Vec (skip-gram)** model directly on the cleaned corpus (100–128 dims).
- Trained a custom **GloVe** model (via `mittens`) on the same co-occurrence matrix.
- Loaded **pretrained Stanford GloVe (6B, 100d)** vectors as an external benchmark for comparison.
- For each representation, inspected nearest neighbors of key sentiment/product words (`good`, `bad`,
  `product`, `quality`, `service`, `excellent`, `terrible`).

## Key findings
- **Raw co-occurrence + PPMI without SVD** produced noisy, low-quality neighbors — dominated by rare
  words with high mutual information by chance, not real semantic similarity.
- **SVD over PPMI** cleaned this up substantially, surfacing much more sensible neighbors (e.g. `good` →
  `great`, `decent`; `product` → `recommend`, `experience`).
- **Word2Vec (skip-gram)** trained on this corpus produced the most domain-relevant neighbors of all
  self-trained methods (e.g. `terrible` → `awful`, `horrible`, `dreadful`; `excellent` → `exceptional`,
  `outstanding`).
- **Custom GloVe** trained on this dataset's co-occurrence matrix also captured sensible domain-specific
  structure (e.g. `service` → `customer`, `support`).
- **Pretrained Stanford GloVe** gave broader, more general-English neighbors, but was noticeably less
  tuned to this dataset's product-review vocabulary and slang compared to the in-domain models.
- Overall: representations trained directly on the target corpus outperformed a generic pretrained
  embedding for this specific domain, reinforcing the choice to build TF-IDF/Word2Vec features
  in-domain for the classification stage.

## Files
- `02_word_representations.ipynb` — co-occurrence, PPMI, SVD, Word2Vec, GloVe (custom + pretrained), and
  nearest-neighbor comparisons.
