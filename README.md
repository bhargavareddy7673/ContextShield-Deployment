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
   - **Input Format**: Model inputs are structured in the verified Colab training format:
     ```python
     "claim: " + claim + " evidence: " + evidence_text
     ```
   - **Evidence Grouping**: Sentences from the same source article are joined with spaces (`" ".join(...)`) to construct coherent article evidence sets, mirroring the original training evidence construction.
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

To run the comparative strategy evaluation across 19,998 development set claims:

```bash
python evaluate_dev.py
```

To run the paired development set significance test:

```bash
python check_significance.py
```

## Comparative Strategy Evaluation (19,998 Development Set Claims)

Evaluated on identical development set examples (`retrieved_dev_top10.jsonl`):

| Strategy | Input Format & Grouping | Accuracy | Macro-F1 | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Old Baseline** | `claim [SEP] sentence` (per-sentence mean) | 54.05% | 54.08% | Uses legacy delimiter; high NEI bias |
| **Strategy 2** | `claim: ... evidence: ...` (**Article-Grouped Mean**) | **51.53%** | **48.81%** | **Matches training construction; SUPPORTS recall 81.1%** |
| **Strategy 1** | `claim: ... evidence: ...` (**Per-Sentence Mean**) | 51.49% | 48.64% | Evaluates each sentence; SUPPORTS recall 81.6% |
| **Strategy 3** | `claim: ... evidence: ...` (Top-1 Sentence) | 50.86% | 48.69% | Rank 1 sentence single prediction |
| **Strategy 4** | `claim: ... evidence: ...` (Top-3 Sentences Joined) | 49.90% | 49.47% | Top 3 sentences concatenated |
| **Strategy 5** | `claim: ... evidence: ...` (All Retrieved Joined) | 46.32% | 45.58% | Concatenating all 10 passages causes noise dilution |

### Key Findings
1. **Evidence Grouping Matters**: Blindly concatenating all retrieved passages into one text causes a severe **5.2% accuracy drop** (down to 46.32%) due to lexical dilution from unrelated sentences.
2. **Article-Grouped Evidence**: Clustering sentences by source article directly matches how FEVER evidence sets were formed and achieves the highest accuracy among training-format strategies.
3. **Format Trade-offs**: Switching from `[SEP]` to `claim: ... evidence: ...` substantially boosts `SUPPORTS` recall (from 62.7% to 81.1%–81.6%) and `REFUTES` precision (from 67.3% to 69.0%), though reducing borderline `NOT ENOUGH INFO` bias.

## Limitations & Calibration Warning

- Model outputs are **uncalibrated scores** from a linear logistic regression classifier over TF-IDF features, not real-world factual probabilities.
- Retrieval similarity measures textual overlap with the corpus, not truthfulness.
- Bag-of-words and character n-grams lack deep compositionality and can be influenced by phrasing artifacts.
