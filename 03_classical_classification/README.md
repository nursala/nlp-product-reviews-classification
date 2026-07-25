# 03 — Basic Text Classification (Classical Models)

## Goal
Build document-level feature representations and train classical ML baselines for 5-class rating
prediction, comparing representations, models, and hyperparameters rigorously.

## What was done
- Split the cleaned data 80/10/10 into train/validation/test, stratified by rating.
- Built two document-level representations: **TF-IDF** (unigrams, max 20k features) and **averaged
  Word2Vec** document vectors (from a Word2Vec model trained on the training split only).
- Trained **Logistic Regression** and **Multinomial Naive Bayes** on both representations (4 combinations
  total).
- Ran hyperparameter sweeps: regularization strength `C` for Logistic Regression (0.1, 1, 10) and
  smoothing `alpha` for Naive Bayes (0.1, 0.5, 1.0).
- Performed qualitative error analysis on misclassified examples for both models.
- Tested both models live on a small hand-written set of English reviews spanning very negative to very
  positive tone.

## Results (test set)

| Representation | Model | Accuracy | F1-macro |
|---|---|---|---|
| TF-IDF | Logistic Regression | **0.951–0.966** | **~0.95–0.97** |
| TF-IDF | Naive Bayes | 0.877–0.884 | ~0.88 |
| Word2Vec | Logistic Regression | 0.915 | 0.914 |
| Word2Vec | Naive Bayes | 0.770 | 0.770 |

(Ranges reflect validation vs. test and different `C`/`alpha` settings tried; see notebook for exact
per-run numbers.)

## Key findings
- **TF-IDF consistently beat Word2Vec** for this task across both models — rating prediction depends on
  specific word-level cues, which an averaged embedding tends to blur, especially for mixed-tone reviews.
- **Logistic Regression consistently beat Naive Bayes** on both representations — LR handles the
  fine-grained, overlapping TF-IDF feature space better than Naive Bayes' independence assumption allows.
- The best hyperparameters found were `C=1` for Logistic Regression and `alpha=1.0` for Naive Bayes —
  representation choice mattered far more than hyperparameter tuning.
- Most errors were between **adjacent ratings** (1↔2, 4↔5), which reflects genuine subjective overlap in
  how reviewers phrase similar sentiment at different star levels — sometimes hard to disambiguate even
  for a human reader.
- TF-IDF + Logistic Regression was established as the strongest classical baseline going into the later,
  more advanced stages.

## Files
- `03_classical_classification.ipynb` — TF-IDF/Word2Vec feature building, model training, hyperparameter
  sweeps, and error analysis.
