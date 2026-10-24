
import json
from pathlib import Path

import joblib
import numpy as np
import scipy.sparse as sp
import streamlit as st


# --------------------------------------------------
# Configuration
# --------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent / "data"
VERIFIER_PATH = DATA_DIR / "evidence_verifier_3class.joblib"
VECTORIZER_PATH = DATA_DIR / "retrieval_tfidf_word.joblib"
MATRIX_PATH = DATA_DIR / "retrieval_tfidf_matrix.npz"
CORPUS_PATH = DATA_DIR / "train_retrieval_corpus.jsonl"

TOP_K = 10


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

    if matrix.shape[0] != len(corpus):
        raise ValueError("Retrieval matrix and corpus row counts differ.")

    if matrix.shape[1] != len(vectorizer.vocabulary_):
        raise ValueError("Retrieval matrix and vocabulary dimensions differ.")

    return verifier, vectorizer, matrix, corpus


# --------------------------------------------------
# Retrieve evidence for a new claim
# --------------------------------------------------
def retrieve_evidence(claim, vectorizer, matrix, corpus, top_k=TOP_K):
    query_vector = vectorizer.transform([claim])
    scores = (matrix @ query_vector.T).toarray().ravel()

    # Exclude zero-score candidates where possible.
    valid_indices = np.flatnonzero(scores > 0)

    if len(valid_indices) == 0:
        return []

    k = min(top_k, len(valid_indices))
    ranked = valid_indices[
        np.argsort(scores[valid_indices])[::-1][:k]
    ]

    return [
        {
            "rank": rank,
            "article_id": corpus[i]["article_id"],
            "sentence_id": corpus[i]["sentence_id"],
            "text": corpus[i]["text"],
            "retrieval_score": float(scores[i]),
        }
        for rank, i in enumerate(ranked, start=1)
    ]


# --------------------------------------------------
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
    "This prototype uses word-level TF-IDF retrieval and a saved "
    "three-class verifier. Its predictions can be wrong. Scores are "
    "model outputs, not calibrated probabilities that a claim is true."
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
            verifier, vectorizer, matrix, corpus = load_pipeline()

        with st.spinner("Analyzing the claim against retrieved evidence..."):
            evidence = retrieve_evidence(
                claim, vectorizer, matrix, corpus, TOP_K
            )
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
            "Prototype limitation: the current retrieval stage uses "
            "word TF-IDF, not the full hybrid retrieval strategy evaluated "
            "during development. Evidence availability and relevance limit "
            "the quality of the final prediction."
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
