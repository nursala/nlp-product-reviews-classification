# 05 — Deep Learning (RNN / LSTM / GRU)

## Goal
Move from bag-of-words style features to sequence models that can learn context and word order
directly, and see how far that pushes performance versus the classical baselines.

## What was done
- Split data 70/15/15 (train/val/test), stratified by rating; analyzed review length distribution to
  pick a max sequence length (230 tokens).
- Built a Keras tokenizer (30k vocab, OOV token) fit on the training set only.
- Trained a dataset-specific **Word2Vec** model (128 dims) and used it to initialize an embedding matrix
  aligned with the tokenizer vocabulary.
- **Baseline (Keras/TensorFlow):** SimpleRNN with the Word2Vec embedding layer, tested both **frozen**
  and **fine-tuned**.
- **Main experiments (PyTorch):** a configurable `PackedRNNClassifier` supporting GRU or LSTM, uni- or
  bidirectional, using `pack_padded_sequence` to correctly ignore padding tokens during training —
  covering four architectures: **GRU**, **LSTM**, **BiGRU**, **BiLSTM**, all with fine-tuned Word2Vec
  embeddings.
- Evaluated all models on validation and test sets with accuracy, macro/weighted F1, confusion matrices,
  and classification reports.

## Results (test set)

| Model | Test Accuracy | F1-macro |
|---|---|---|
| SimpleRNN + frozen Word2Vec (Keras) | ~0.66 (val) | — |
| SimpleRNN + fine-tuned Word2Vec (Keras) | ~0.66 (val) | — |
| **GRU (unidirectional, packed)** | **0.969** | **0.968** |
| **BiGRU (packed)** | **0.969** | **0.968** |
| LSTM (unidirectional, packed) | 0.809 | 0.807 |
| BiLSTM (packed) | 0.809 | 0.807 |

## Key findings
- The simple Keras `SimpleRNN` baseline (frozen or fine-tuned embeddings) plateaued around 60–66%
  validation accuracy — a useful floor showing the value of a better architecture and training setup.
- Moving to **packed-sequence GRU/BiGRU in PyTorch** was a major jump, reaching **~96.9% test accuracy**
  and **F1-macro ~0.968** — on par with the classical TF-IDF + Logistic Regression baseline from Stage 3,
  and a strong result for a from-scratch sequence model.
- **LSTM and BiLSTM underperformed GRU/BiGRU** in this run (~81% accuracy), converging less cleanly over
  12 epochs. This is likely training-configuration sensitivity (LSTM has more gates/parameters and can
  need different learning-rate/epoch tuning than GRU) rather than a fundamental limitation of LSTMs —
  worth revisiting with a learning-rate schedule or more epochs if extending this stage.
- Properly handling variable-length sequences (`pack_padded_sequence`, explicit lengths in the
  DataLoader) was essential — an earlier version of the pipeline without this fix was silently letting
  padding tokens influence predictions.
- Bidirectionality (Bi- prefix) did not provide a clear additional gain over the unidirectional versions
  here, suggesting the unidirectional context alone was already sufficient for this classification task.

## Files
- `05_deep_learning_rnn_lstm.ipynb` — tokenization, Word2Vec embedding, SimpleRNN baseline, and packed
  GRU/LSTM/BiGRU/BiLSTM experiments with full evaluation.
