# ContextShield

ContextShield is an NLP research prototype for evidence-based claim verification over a 108,559-sentence Wikipedia corpus.

## Architecture & Pipeline

1. **Retrieval**:
   - **Lexical**: Sparse word-level TF-IDF retrieval (`retrieval_tfidf_word.joblib`, `retrieval_tfidf_matrix.npz`).
   - **Semantic**: Dense cosine similarity using `sentence-transformers/all-MiniLM-L6-v2` with 384-dimensional normalized embeddings (`train_corpus_embeddings.npy`).
   - **Fusion**: Reciprocal Rank Fusion ($RRF_k = 60$) combining top lexical and semantic candidates, with additional unique semantic passages appended up to 15 total sentences.
2. **Verification**:
   - FeatureUnion combining word n-grams (1-2) and character n-grams (3-5) with sublinear term-frequency scaling.
   - LogisticRegression classifier with balanced class weights.
   - Pairwise input formatting: `f"{claim} [SEP] {evidence_text}"`.
   - Aggregation: Mean class score pooling across retrieved evidence passages.
   - Classes: `SUPPORTS`, `REFUTES`, `NOT ENOUGH INFO`.

## Running the Application

Ensure the Python virtual environment is activated:

```bash
source .venv/bin/activate
```

Launch the Streamlit app:

```bash
python -m streamlit run app.py --server.fileWatcherType none
```

To run the unit and integration tests:

```bash
python -m unittest discover tests -v
```

To run the paired development set evaluation and significance test:

```bash
python check_significance.py
```

## Evaluation Results (Development Set)

Evaluated on 19,998 development set claims:

- **Baseline Retrieval (TF-IDF Top 10)**:
  - Accuracy: `54.05%` (0.540504)
  - Macro-F1: `54.08%` (0.540761)
- **Hybrid Retrieval (RRF Top 10 + Semantic Append to 15)**:
  - Accuracy: `54.51%` (0.545055)
  - Macro-F1: `54.60%` (0.546019)
- **Statistical Significance**:
  - McNemar exact paired two-sided test: $p = 0.022787$ (net $+91$ correct predictions).

## Limitations & Calibration Warning

- Model outputs are **uncalibrated scores** from a linear logistic regression classifier, not real-world factual probabilities.
- Retrieval similarity measures textual overlap with the corpus, not truthfulness.
- Bag-of-words and character n-grams lack deep compositionality and may be swayed by phrasing artifacts. Outputs should be treated as provisional research indicators.
