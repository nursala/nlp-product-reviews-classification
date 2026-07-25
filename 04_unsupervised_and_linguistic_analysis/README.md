# 04 — Unsupervised & Linguistic Analysis

## Goal
Look past the star-rating labels entirely and ask: what natural structure exists in the review text
itself, and what syntactic patterns characterize different types of reviews?

## What was done

### Clustering
- Ran **K-Means** for k=2..10 over TF-IDF vectors, using the Elbow method (inertia) and Silhouette score
  to pick k.
- Ran **Agglomerative (Hierarchical) Clustering** on a dimensionality-reduced (TruncatedSVD, 100 dims)
  sample, using cosine distance, for comparison.
- Visualized clusters in 2D via PCA.
- Evaluated cluster quality against the true rating labels using **Adjusted Rand Index (ARI)** and
  **Purity**.

### Topic Modeling
- Ran **LDA** independently within each K-Means cluster to extract representative topics/keywords.
- Ran a **global LDA** (6 topics) across the entire corpus for a macro view.

### Linguistic / Relation Extraction
- Used **spaCy (en_core_web_trf)** + **Textacy** to extract syntactic relations from representative
  sentences in each cluster: Subject–Verb–Object (SVO), Entity–Location, Entity–Attribute, and a custom
  Product–is–Adjective pattern for direct opinion statements.

## Key findings
- **k=4 was chosen over k=5** (despite 5 star ratings existing) — K-Means is unsupervised and clusters on
  textual/topical similarity, not sentiment labels. The data naturally organized into ~4 topical groups
  regardless of star count.
- **Hierarchical clustering underperformed**: nearly all documents collapsed into one giant cluster
  (9,749 of ~10,000), with three tiny outlier clusters. This is a known failure mode of distance-based
  clustering on high-dimensional sparse TF-IDF data — distances become nearly indistinguishable
  ("curse of dimensionality"), so K-Means (which optimizes variance directly) handled this data much better.
- The four K-Means clusters mapped to clearly interpretable themes via LDA: **negative reviews about
  quality/features**, **positive reviews with strong satisfaction**, **neutral/mixed reviews**, and a
  distinct **video game reviews** cluster.
- **ARI ≈ 0.56** (moderate alignment with true ratings) but **Purity was very low** — expected, since
  clusters captured *topic* (what the review is about) rather than *sentiment intensity* (the star
  rating). A 5-star and 1-star review about the same product can land in the same topic cluster.
- Running K-Means with **k=10** produced even richer, more specific product-category clusters (toys,
  travel gear, skincare, smart home, etc.), confirming the dataset spans diverse product categories, not
  just sentiment polarity — k=4 was kept for the main report because it best matched the Elbow/Silhouette
  metrics and stayed interpretable.
- **Relation extraction** confirmed the cluster themes at the syntax level: the negative cluster produced
  relations like `manufacturer – need – address issue`; the positive cluster produced `ride-on toy –
  brought – joy`; the gaming cluster produced `shift mechanic – enables – player ability`.
- SVO relations dominated the extracted patterns; Entity–Location and Entity–Attribute relations were
  rare, since user reviews are written as casual free-form text rather than formal grammatical sentences
  — this motivated adding the custom Product–is–Adjective pattern to better capture direct opinions
  (e.g. "product is great").

## Files
- `04_unsupervised_and_linguistic_analysis.ipynb` — clustering, topic modeling, and relation extraction.
