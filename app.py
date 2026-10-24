
import json
from pathlib import Path

import joblib
import numpy as np
import scipy.sparse as sp
import streamlit as st
from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# Configuration
# --------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent / "data"
VERIFIER_PATH = DATA_DIR / "evidence_verifier_3class.joblib"
VECTORIZER_PATH = DATA_DIR / "retrieval_tfidf_word.joblib"
MATRIX_PATH = DATA_DIR / "retrieval_tfidf_matrix.npz"
CORPUS_PATH = DATA_DIR / "train_retrieval_corpus.jsonl"


TOP_K = 10
EMBEDDINGS_PATH = DATA_DIR / "train_corpus_embeddings.npy"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

RRF_K = 60
FINAL_EVIDENCE_K = 15

# --------------------------------------------------
# Load saved artifacts once
# --------------------------------------------------
@st.cache_resource
def load_pipeline():
    required = [
        VERIFIER_PATH,
        VECTORIZER_PATH,
        MATRIX_PATH,
        CORPUS_PATH,
        EMBEDDINGS_PATH,
    ]

    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "Required ContextShield files are missing: "
            + ", ".join(missing)
        )

    verifier = joblib.load(VERIFIER_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    matrix = sp.load_npz(MATRIX_PATH)

    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        corpus = [json.loads(line) for line in f if line.strip()]

    embeddings = np.load(EMBEDDINGS_PATH, mmap_mode="r")
    semantic_model = SentenceTransformer(
        EMBEDDING_MODEL_NAME,
        device="cpu",
    )

    if matrix.shape[0] != len(corpus):
        raise ValueError("Retrieval matrix and corpus row counts differ.")

    if matrix.shape[1] != len(vectorizer.vocabulary_):
        raise ValueError("Retrieval matrix and vocabulary dimensions differ.")

    if embeddings.shape[0] != len(corpus):
        raise ValueError("Semantic embeddings and corpus row counts differ.")

    if embeddings.shape[1] != semantic_model.get_embedding_dimension():
        raise ValueError("Semantic embedding dimensions do not match the model.")

    return (
        verifier,
        vectorizer,
        matrix,
        corpus,
        embeddings,
        semantic_model,
    )


def retrieve_evidence(
    claim,
    vectorizer,
    matrix,
    corpus,
    embeddings,
    semantic_model,
    top_k=TOP_K,
):
    source_k = min(top_k, len(corpus))
    if source_k == 0:
        return []

    # 1. Retrieve candidates using TF-IDF.
    query_vector = vectorizer.transform([claim])
    tfidf_scores = (matrix @ query_vector.T).toarray().ravel()
    tfidf_valid = np.flatnonzero(tfidf_scores > 0)

    tfidf_indices = tfidf_valid[
        np.argsort(tfidf_scores[tfidf_valid])[::-1][:source_k]
    ]

    # 2. Retrieve candidates using semantic similarity.
    query_embedding = semantic_model.encode(
        [claim],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )[0]

    semantic_scores = embeddings @ query_embedding
    semantic_indices = np.argsort(semantic_scores)[::-1][:source_k]

    # 3. Fuse both ranked lists using reciprocal-rank fusion.
    candidates = {}

    for rank, index in enumerate(tfidf_indices, start=1):
        index = int(index)
        key = (
            corpus[index]["article_id"],
            corpus[index]["sentence_id"],
        )

        candidates.setdefault(
            key,
            {
                "index": index,
                "rrf_score": 0.0,
                "retrieved_by": [],
            },
        )
        candidates[key]["rrf_score"] += 1.0 / (RRF_K + rank)
        candidates[key]["retrieved_by"].append("tfidf")

    for rank, index in enumerate(semantic_indices, start=1):
        index = int(index)
        key = (
            corpus[index]["article_id"],
            corpus[index]["sentence_id"],
        )

        candidates.setdefault(
            key,
            {
                "index": index,
                "rrf_score": 0.0,
                "retrieved_by": [],
            },
        )
        candidates[key]["rrf_score"] += 1.0 / (RRF_K + rank)
        candidates[key]["retrieved_by"].append("semantic")

    fused = sorted(
        candidates.values(),
        key=lambda item: item["rrf_score"],
        reverse=True,
    )

    # 4. Keep the hybrid top 10.
    hybrid_top = fused[:top_k]

    selected_keys = {
        (
            corpus[item["index"]]["article_id"],
            corpus[item["index"]]["sentence_id"],
        )
        for item in hybrid_top
    }

    selected = []

    for item in hybrid_top:
        record = corpus[item["index"]]
        selected.append(
            {
                "rank": len(selected) + 1,
                "article_id": record["article_id"],
                "sentence_id": record["sentence_id"],
                "text": record["text"],
                "retrieval_score": float(item["rrf_score"]),
                "retrieval_score_type": "reciprocal_rank_fusion",
                "retrieved_by": item["retrieved_by"],
            }
        )

    # 5. Append unique semantic candidates until we reach 15.
    for index in semantic_indices:
        if len(selected) >= min(FINAL_EVIDENCE_K, len(corpus)):
            break

        index = int(index)
        record = corpus[index]
        key = (record["article_id"], record["sentence_id"])

        if key in selected_keys:
            continue

        selected_keys.add(key)
        selected.append(
            {
                "rank": len(selected) + 1,
                "article_id": record["article_id"],
                "sentence_id": record["sentence_id"],
                "text": record["text"],
                "retrieval_score": float(semantic_scores[index]),
                "retrieval_score_type": "cosine_similarity",
                "retrieved_by": ["semantic"],
            }
        )

    return selected
# Verify claim against retrieved evidence
# --------------------------------------------------
def verify_claim(claim, evidence, verifier):
    if not evidence:
        return None, []

    pairs = [
        f"{claim} [SEP] {item['text']}"
        for item in evidence
    ]

    probabilities = verifier.predict_proba(pairs)
    classes = np.asarray(verifier.classes_)

    # Experimental strategy: average class probabilities
    # across the retrieved claim-evidence pairs.
    mean_probabilities = probabilities.mean(axis=0)
    best_index = int(np.argmax(mean_probabilities))

    results = []

    for item, probs in zip(evidence, probabilities):
        best = int(np.argmax(probs))
        result = dict(item)
        result["predicted_label"] = str(classes[best])
        result["model_score"] = float(probs[best])
        results.append(result)

    summary = {
        "label": str(classes[best_index]),
        "scores": {
            str(label): float(prob)
            for label, prob in zip(classes, mean_probabilities)
        },
    }

    return summary, results


# --------------------------------------------------
# Streamlit interface
# --------------------------------------------------
st.set_page_config(
    page_title="ContextShield",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ ContextShield")
st.subheader("Evidence-based claim checking")

st.write(
    "Enter a factual claim. ContextShield retrieves related sentences "
    "from its local evidence corpus and evaluates each claim–evidence pair."
)
st.info(
    "This prototype combines word-level TF-IDF and semantic retrieval "
    "with reciprocal-rank fusion, followed by a saved three-class "
    "verifier. Its predictions can be wrong. Scores are model outputs, "
    "not calibrated probabilities that a claim is true."
)


with st.form("claim_form"):
    claim = st.text_area(
        "Claim to check",
        placeholder="Example: The Carolina Panthers' inaugural NFL season was in 1995.",
        height=100,
    )
    submitted = st.form_submit_button(
        "Check claim",
        type="primary",
        use_container_width=True,
    )

if submitted:
    claim = claim.strip()

    if not claim:
        st.warning("Please enter a claim first.")
        st.stop()

    try:
       
        with st.spinner("Loading ContextShield and retrieving evidence..."):
            (
                verifier,
                vectorizer,
                matrix,
                corpus,
                embeddings,
                semantic_model,
            ) = load_pipeline()

        with st.spinner("Retrieving hybrid evidence..."):
            evidence = retrieve_evidence(
                claim,
                vectorizer,
                matrix,
                corpus,
                embeddings,
                semantic_model,
                TOP_K,
            )

        with st.spinner("Analyzing the claim against retrieved evidence..."):
            summary, evidence_results = verify_claim(
                claim, evidence, verifier
            )


        st.divider()
        st.header("Analysis result")

        if summary is None:
            st.warning(
                "No matching evidence was found in the local corpus. "
                "ContextShield cannot make a supported verdict for this claim."
            )
            st.stop()

        st.metric("Provisional model prediction", summary["label"])




        st.caption(
            "This is the class with the highest mean model score across "
            "retrieved claim–evidence pairs. It is not a verified fact."
        )

        st.subheader("Mean class scores")
        score_order = [
            "SUPPORTS",
            "REFUTES",
            "NOT ENOUGH INFO",
        ]

        for label in score_order:
            if label in summary["scores"]:
                score = summary["scores"][label]
                st.write(f"**{label}** — {score:.3f}")
                st.progress(max(0.0, min(1.0, score)))

        st.subheader("Retrieved evidence")
        st.caption(
            "Rank is based on textual similarity, not truth. "
            "Individual verifier predictions may disagree with the aggregate."
        )

        for item in evidence_results:
            heading = (
                f"Rank {item['rank']} · {item['article_id']} "
                f"· Sentence {item['sentence_id']}"
            )

            with st.expander(heading, expanded=(item["rank"] <= 3)):
                st.write(item["text"])
                st.write(
                    f"Retrieval similarity: "
                    f"{item['retrieval_score']:.4f}"
                )
                st.write(
                    f"Individual verifier prediction: "
                    f"{item['predicted_label']}"
                )
                st.write(
                    f"Individual model score: "
                    f"{item['model_score']:.3f}"
                )

        st.divider()
        st.caption(
            "Prototype limitation: hybrid retrieval combines TF-IDF and "
            "semantic similarity, but retrieved evidence may still be "
            "irrelevant or incomplete. The final prediction is a model "
           "estimate, not a verified fact."
        )

            

    except Exception as exc:
        st.error(
            "ContextShield could not complete the analysis. "
            "Check that the required files are available and readable."
        )
        st.exception(exc)

else:
    st.markdown("### How it works")
    st.markdown(
        """
        1. **Enter a claim** you want to investigate.
        2. **Retrieve evidence** from the saved local corpus.
        3. **Evaluate evidence** using the existing verifier.
        4. **Review the result** and inspect individual evidence sentences.
        """
    )

st.caption("ContextShield · Research prototype")
