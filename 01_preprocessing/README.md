# 01 — Dataset Selection & Preprocessing

## Goal
Take raw, noisy product reviews and turn them into a clean, tokenized, lemmatized dataset ready for
every later stage of the pipeline.

## What was done
- Loaded ~10,000 labeled product reviews (`review` text + `rating` 1–5) from the
  [agentlans/ai-product-reviews](https://huggingface.co/datasets/agentlans/ai-product-reviews) dataset.
- Removed rows with missing/empty text or labels.
- Filtered to English-only text using `langdetect`.
- Removed exact duplicate reviews, then near-duplicates via a normalized-text heuristic (lowercased,
  punctuation stripped, whitespace collapsed).
- Cleaned each review: expanded contractions, replaced URLs/emails/numbers with fixed tokens
  (`_url_`, `_email_`, `_digit_`), stripped emoji/non-ASCII characters and punctuation, lowercased.
- Tokenized with NLTK, handled hyphenated/apostrophe'd words, removed an extended stopword list.
- Lemmatized with `WordNetLemmatizer` using POS tags (rather than stemming) to keep base words meaningful.
- Computed basic statistics: label distribution, text-length distribution, top frequent words, and top bigrams.

## Key findings
- The label distribution across 1–5 stars is close to balanced (~19–21% per class).
- After cleaning, the dataset is fully English, duplicate-free, and free of empty records.
- The language is natural (real reviews), with some slang and typos, but generally clean enough for
  direct downstream use without heavier normalization.

## Files
- `01_preprocessing.ipynb` — full cleaning, tokenization, lemmatization, and stats pipeline.
