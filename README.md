<h1 align="center">🛒 AI Product Reviews — NLP Rating Classification Pipeline</h1>

<p align="center">
  An end-to-end NLP pipeline that classifies product reviews into 1–5 star ratings, moving from raw text
  cleaning all the way through classical ML, unsupervised analysis, and fine-tuned Transformers.<br>
  Built with <strong>NLTK, spaCy, Gensim, scikit-learn, PyTorch, TensorFlow &amp; Hugging Face Transformers</strong>.
</p>

---

## 📖 About the Project

This project takes ~10,000 labeled product reviews and pushes them through a full NLP pipeline across
six progressive stages — cleaning and tokenizing raw text, building and comparing multiple word
representations, training classical ML baselines, uncovering hidden structure with clustering and
linguistic parsing, and finally training sequence models and fine-tuning Transformers to see how far
each approach pushes performance.

Every stage was evaluated independently and compared against the others, with the goal of understanding
**not just which model wins, but why** — and whether the added complexity of each step was actually worth it.

**Dataset:** [agentlans/ai-product-reviews](https://huggingface.co/datasets/agentlans/ai-product-reviews) (Hugging Face) — ~10,000 English-language product reviews labeled with a 1–5 star rating, fairly balanced across classes (~20% per class).

---

## 🗂️ Pipeline Stages

- 🧹 **[01 — Preprocessing](./01_preprocessing)** – Deduplication, language filtering, contraction expansion, URL/email/number normalization, stopword removal, tokenization, and POS-aware lemmatization.
- 🔢 **[02 — Word Representations](./02_word_representations)** – Co-occurrence + PPMI + SVD embeddings, self-trained Word2Vec and GloVe, benchmarked against pretrained Stanford GloVe.
- 🎯 **[03 — Classical Classification](./03_classical_classification)** – TF-IDF vs. Word2Vec document vectors, Logistic Regression vs. Naive Bayes, full hyperparameter sweeps and error analysis.
- 🔍 **[04 — Unsupervised & Linguistic Analysis](./04_unsupervised_and_linguistic_analysis)** – K-Means vs. Hierarchical Clustering, PCA visualization, per-cluster LDA topic modeling, and spaCy/Textacy relation extraction.
- 🧠 **[05 — Deep Learning (RNN/LSTM)](./05_deep_learning_rnn_lstm)** – SimpleRNN baseline, then packed-sequence GRU/LSTM/BiGRU/BiLSTM models in PyTorch with Word2Vec-initialized embeddings.
- 🤖 **[06 — Transformers](./06_transformers)** – Fine-tuned DistilBERT and BERT-base for the same 5-class task, with training curves and hard-example error analysis.

Each folder contains its own notebook and a `README.md` with detailed methodology, results, and findings.

---

## 📊 Results Summary

| Stage | Model | Accuracy | F1-macro |
|---|---|---|---|
| 3 | TF-IDF + Logistic Regression | 0.951–0.966 | ~0.95–0.97 |
| 3 | TF-IDF + Naive Bayes | 0.884 | 0.883 |
| 3 | Word2Vec + Logistic Regression | 0.915 | 0.914 |
| 3 | Word2Vec + Naive Bayes | 0.770 | 0.770 |
| 5 | GRU / BiGRU (packed, Word2Vec-init) | **0.969** | 0.968 |
| 5 | LSTM / BiLSTM (packed, Word2Vec-init) | 0.809 | 0.807 |
| 6 | DistilBERT (fine-tuned) | 0.977 | 0.976 |
| 6 | BERT-base (fine-tuned) | **0.979** | **0.979** |

**Key takeaway:** TF-IDF consistently beat Word2Vec for document-level classification (rating prediction
leans on specific word-level cues that averaged embeddings blur), GRU/BiGRU clearly outperformed
LSTM/BiLSTM in this training run, and Transformers won overall — but only by a slim margin over a
well-tuned classical baseline. On a clean, balanced dataset like this one, simple methods go a long way,
and the extra cost of a Transformer buys a real but modest improvement.

---

## 🔍 Selected Highlights

- **Stage 2:** representations trained directly on this corpus (Word2Vec, custom GloVe) captured
  domain-specific meaning noticeably better than generic pretrained Stanford GloVe vectors — e.g. `terrible`
  → `awful, horrible, dreadful` vs. more generic neighbors from the pretrained model.
- **Stage 4:** K-Means (k=4) found four clean, interpretable topic clusters (negative/quality complaints,
  positive experiences, neutral/mixed, and a distinct gaming-review cluster) — chosen over k=5 because
  unsupervised clustering finds *topic* structure, not sentiment labels. Hierarchical clustering failed on
  this data, collapsing almost everything into one giant cluster — a classic high-dimensional sparse-data
  failure mode.
- **Stage 5:** properly packing variable-length sequences (`pack_padded_sequence`) was essential for
  correct RNN training; GRU/BiGRU reached ~96.9% accuracy, on par with the classical TF-IDF baseline.
- **Stage 6:** BERT and DistilBERT both converged cleanly with no overfitting, and the majority of
  remaining errors were on adjacent ratings (1↔2, 4↔5) and mixed-sentiment reviews — the same failure
  pattern observed at every earlier stage, pointing to genuine ambiguity in the data rather than a model
  weakness.

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| `pandas`, `numpy` | Data handling |
| `NLTK` | Tokenization, stopwords, lemmatization, POS tagging |
| `scikit-learn` | TF-IDF, Logistic Regression, Naive Bayes, K-Means, PCA, SVD, metrics |
| `Gensim` | Word2Vec, LDA topic modeling |
| `mittens` | Custom GloVe training |
| `spaCy` + `Textacy` | Dependency parsing, NER, relation extraction |
| `TensorFlow / Keras` | SimpleRNN baseline |
| `PyTorch` | Packed-sequence GRU/LSTM/BiGRU/BiLSTM models |
| `Hugging Face Transformers` | DistilBERT / BERT fine-tuning |
| `matplotlib`, `seaborn` | Visualizations |

---

## ▶️ How to Run

1. Clone the repo and install dependencies: `pip install -r requirements.txt`
2. Download the dataset from the [Hugging Face link](https://huggingface.co/datasets/agentlans/ai-product-reviews) and place it in each stage folder as needed (see individual notebooks for expected filenames).
3. Run the notebooks in order (`01_preprocessing` → `06_transformers`) — each stage builds on artifacts
   produced by the previous one (cleaned CSV, tokenized columns, saved embeddings, etc.).

---

## 🚀 Possible Future Work

- Revisit the LSTM/BiLSTM training setup in Stage 5 (learning-rate schedule, more epochs) to see if the
  gap to GRU/BiGRU closes.
- Add t-SNE as an alternative to PCA for embedding visualization.
- Formalize the zero-shot/one-shot LLM comparison from Stage 6 into a proper evaluation.
- Package the best-performing model (BERT, or TF-IDF + Logistic Regression for a low-latency option)
  behind a simple inference API.

---

## 🏁 Conclusion

This project walks a single dataset through the full modern NLP toolkit — from regex-based cleaning to
fine-tuned Transformers — and treats each stage as a genuine experiment rather than a checkbox. The
biggest insight isn't that BERT won (it did, by a small margin), it's *why* the gap between BERT and a
well-tuned TF-IDF + Logistic Regression baseline stayed so small: a clean, balanced dataset lets simple
methods punch far above their weight, and knowing when the added cost of a heavier model is actually
worth it is as important as knowing how to build one.

## 👥 Team
Built as a group project across a semester-long NLP course.
