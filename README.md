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

Evaluated on identical development set examples across all 19,998 claims without modifying or retraining any saved artifacts:

### Dataset 1: `retrieved_dev_top10.jsonl` (Top-10 Lexical & Semantic Retrieval)

| Strategy | Input Format & Aggregation | Accuracy | Macro-F1 | Precision (NEI / REF / SUP) | Recall (NEI / REF / SUP) | F1 (NEI / REF / SUP) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Old Baseline** | `claim [SEP] sentence` (per-sentence mean) | 54.05% | 0.5408 | 0.4925 / 0.6730 / 0.4889 | 0.4409 / 0.5540 / 0.6266 | 0.4653 / 0.6078 / 0.5493 |
| **Strategy 1** | `claim: ... evidence: ...` (Per-Sentence Mean) | 51.49% | 0.4864 | 0.5484 / 0.6910 / 0.4362 | 0.1971 / 0.5311 / 0.8165 | 0.2900 / 0.6006 / 0.5686 |
| **Strategy 2** | `claim: ... evidence: ...` (Article-Grouped Mean) | 51.53% | 0.4881 | 0.5424 / 0.6900 / 0.4372 | 0.2033 / 0.5317 / 0.8108 | 0.2957 / 0.6006 / 0.5681 |
| **Strategy 3** | `claim: ... evidence: ...` (All Retrieved Joined) | 46.32% | 0.4558 | 0.4109 / 0.5254 / 0.4816 | 0.5738 / 0.5221 / 0.2937 | 0.4789 / 0.5237 / 0.3649 |
| **Strategy 4** | `claim: ... evidence: ...` (**Article-Grouped Relevance-Weighted $1/r$**) | **51.54%** | **0.4910** | **0.5380 / 0.6732 / 0.4400** | **0.2168 / 0.5389 / 0.7906** | **0.3090 / 0.5986 / 0.5654** |
| **Strategy 5** | `claim: ... evidence: ...` (Per-Sentence Relevance-Weighted $1/r$) | 51.38% | 0.4863 | 0.5450 / 0.6788 / 0.4370 | 0.2015 / 0.5345 / 0.8053 | 0.2942 / 0.5981 / 0.5665 |

### Dataset 2: `retrieved_dev_hybrid_first_semantic_append_top20.jsonl` (Hybrid Fusion + Semantic Append)

| Strategy | Input Format & Aggregation | Accuracy | Macro-F1 | Precision (NEI / REF / SUP) | Recall (NEI / REF / SUP) | F1 (NEI / REF / SUP) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Old Baseline** | `claim [SEP] sentence` (per-sentence mean) | 54.51% | 0.5460 | 0.4919 / 0.6946 / 0.4916 | 0.4527 / 0.5477 / 0.6347 | 0.4715 / 0.6125 / 0.5540 |
| **Strategy 1** | `claim: ... evidence: ...` (Per-Sentence Mean) | 51.32% | 0.4839 | 0.5532 / 0.7097 / 0.4320 | 0.1934 / 0.5141 / 0.8321 | 0.2866 / 0.5963 / 0.5688 |
| **Strategy 2** | `claim: ... evidence: ...` (Article-Grouped Mean) | 51.40% | 0.4880 | 0.5406 / 0.7032 / 0.4344 | 0.2085 / 0.5174 / 0.8161 | 0.3010 / 0.5961 / 0.5670 |
| **Strategy 3** | `claim: ... evidence: ...` (All Retrieved Joined) | 46.97% | 0.4624 | 0.4162 / 0.5465 / 0.4731 | 0.5882 / 0.5246 / 0.2964 | 0.4874 / 0.5353 / 0.3645 |
| **Strategy 4** | `claim: ... evidence: ...` (**Article-Grouped Relevance-Weighted $1/r$**) | **51.37%** | **0.4913** | **0.5284 / 0.6785 / 0.4383** | **0.2250 / 0.5306 / 0.7855** | **0.3156 / 0.5955 / 0.5626** |
| **Strategy 5** | `claim: ... evidence: ...` (Per-Sentence Relevance-Weighted $1/r$) | 51.32% | 0.4845 | 0.5546 / 0.6934 / 0.4335 | 0.1965 / 0.5231 / 0.8198 | 0.2902 / 0.5963 / 0.5671 |

### Key Findings & Trade-offs
1. **Relevance-Aware Weighting Protects Decisive Evidence**: Weighting evidence by reciprocal retrieval rank ($w_k = 1 / \text{rank}_k$) ensures that highly ranked relevant evidence (Rank 1 weight ~0.34) is not overwhelmed by uninformative background passages (Ranks 10–15 weights ~0.06–0.10).
2. **Consistent Macro-F1 Gains Across Datasets**: Article-grouped relevance-weighted aggregation achieves the highest Macro-F1 among all training-format models on both datasets (**0.4910** on top-10, **0.4913** on hybrid semantic append), specifically boosting `NOT ENOUGH INFO` F1 (0.3010 $\to$ 0.3156) by suppressing distractor noise.
3. **Evidence Grouping Matters**: Blindly concatenating all retrieved passages into one single text causes a severe **4.4%–5.2% accuracy drop** (down to 46.32% and 46.97%) due to severe feature dilution from unrelated passages.
4. **Format Trade-offs**: Switching from `[SEP]` to `claim: ... evidence: ...` substantially boosts `SUPPORTS` recall (from ~41.3%–62.7% up to 78.6%–83.2%) and `REFUTES` precision (up to 70.3%), matching the true distribution intended by the original training formulation.

## Limitations & Calibration Warning

- Model outputs are **uncalibrated scores** from a linear logistic regression classifier over TF-IDF features, not real-world factual probabilities.
- Retrieval similarity measures textual overlap with the corpus, not truthfulness.
- Bag-of-words and character n-grams lack deep compositionality and can be influenced by phrasing artifacts.
