# 06 — Transformers (Fine-Tuning BERT & DistilBERT)

## Goal
Fine-tune modern Transformer models on the same 5-class rating task and see whether attention-based
architectures justify their added computational cost over the classical and RNN baselines.

## What was done
- Split data 70/15/15 (train/val/test), stratified by rating (same protocol as Stage 5).
- Fine-tuned two pretrained Hugging Face models end-to-end: **DistilBERT-base-uncased** and
  **BERT-base-uncased**, both for 5-class sequence classification.
- Used the Hugging Face `Trainer` with `AdamW`, learning rate `2e-5`, batch size 16, 3 epochs, evaluating
  every epoch and keeping the best checkpoint by macro F1.
- Ran preliminary experiments with higher/lower learning rates to understand training stability before
  settling on the final configuration.
- Plotted training/validation loss and accuracy/F1 curves per epoch.
- Performed detailed error analysis: confusion matrices, high-confidence wrong predictions, low-confidence
  (small-margin) predictions, and short misclassified reviews.
- Ran a brief **zero-shot / one-shot LLM** experiment as a qualitative comparison point (see note below).

## Results (test set)

| Model | Accuracy | F1-macro |
|---|---|---|
| DistilBERT (fine-tuned) | 0.977 | 0.976 |
| **BERT-base (fine-tuned)** | **0.979** | **0.979** |

## Key findings
- **BERT slightly outperformed DistilBERT** (~0.3% accuracy gap) — a small but consistent edge, at roughly
  double the parameter count and training cost.
- Both Transformers **outperformed every prior stage**, but only marginally over the Stage 3 classical
  baseline (TF-IDF + Logistic Regression, ~95–97%) and the Stage 5 GRU/BiGRU models (~96.9%) — this
  dataset was clean and well-balanced enough that simpler, cheaper models already captured most of the
  signal.
- Learning rate sensitivity was significant: rates that were too high caused unstable, oscillating loss
  across epochs; rates that were too low converged too slowly within 3 epochs. `2e-5` gave the best
  balance of stability and final performance.
- No meaningful overfitting or underfitting was observed for either model — training and validation loss
  decreased together without diverging.
- Remaining errors concentrated on **adjacent ratings (1↔2, 4↔5)** and reviews with **mixed sentiment**
  (positive and negative language in the same review) — the same failure pattern seen in every earlier
  stage, suggesting this is a genuine property of the data (subjective, overlapping language) rather than
  a model limitation.
- A short zero-shot/one-shot LLM experiment suggested large language models can reason about ambiguous,
  mixed-sentiment reviews reasonably well without fine-tuning, but results were less consistent than the
  fine-tuned models and depended heavily on prompt phrasing — treated as a complementary qualitative
  check rather than a competing model (results are in a separate spreadsheet, not included in this repo).

## Conclusion across all modeling stages
| Stage | Best Model | Accuracy |
|---|---|---|
| 3 | TF-IDF + Logistic Regression | ~0.95–0.97 |
| 5 | GRU / BiGRU (packed, Word2Vec-initialized) | ~0.969 |
| 6 | BERT-base (fine-tuned) | **0.979** |

Transformers won, but by a small margin — the practical takeaway is that when a dataset is this clean
and balanced, the choice between a cheap classical baseline and an expensive fine-tuned Transformer
should be driven by how much that last 1–3% of accuracy is actually worth for the use case.

## Files
- `06_transformers.ipynb` — DistilBERT/BERT fine-tuning, training curves, evaluation, and error analysis.
