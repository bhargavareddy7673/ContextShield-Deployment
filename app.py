"""ContextShield — Evidence-Based Claim Verification Streamlit Application.

Interactive prototype combining lexical (TF-IDF) and dense semantic retrieval
with reciprocal-rank fusion and a saved three-class evidence verifier.
"""

from pathlib import Path
import streamlit as st

from pipeline import (
    BASE_DIR,
    DATA_DIR,
    FINAL_EVIDENCE_K,
    RRF_K,
    TOP_K,
    format_claim_evidence,
    group_evidence_by_article,
    load_pipeline as _load_pipeline_raw,
    retrieve_evidence,
    verify_claim,
)

# --------------------------------------------------
# Cached artifact loader
# --------------------------------------------------
@st.cache_resource(show_spinner="Loading ContextShield verifier, corpus, and embeddings...")
def load_pipeline():
    """Cache loaded model artifacts in Streamlit resource cache."""
    return _load_pipeline_raw(DATA_DIR)


# --------------------------------------------------
# Main UI function
# --------------------------------------------------
def main():
    st.set_page_config(
        page_title="ContextShield — Claim Verification",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("🛡️ ContextShield")
    st.subheader("Evidence-Based Claim Verification Prototype")

    st.markdown(
        "ContextShield searches a 108,559-sentence local corpus using hybrid retrieval "
        "(word TF-IDF + dense semantic embeddings with Reciprocal Rank Fusion) and evaluates "
        "the claim against retrieved evidence passages using a saved three-class verifier."
    )

    # Scientific integrity notice
    st.info(
        "ℹ️ **Evaluation & Calibration Notice**: Model inputs are structured in the verified training format "
        "(`claim: {claim} evidence: {evidence_text}`). Scores are **uncalibrated outputs** "
        "produced by a linear Logistic Regression classifier over TF-IDF word and character features. "
        "They are **not calibrated probabilities** that a claim is true in the real world. "
        "Retrieval scores reflect textual and semantic similarity to the local corpus, not factual validity."
    )

    # Sidebar: System configuration and sample queries
    with st.sidebar:
        st.header("⚙️ System Architecture")
        st.markdown(
            """
            - **Input Format**: `claim: {claim} evidence: {evidence_text}`
            - **Retrieval**: Hybrid (Word TF-IDF + all-MiniLM-L6-v2)
            - **Fusion**: Reciprocal Rank Fusion ($k=60$)
            - **Evidence Pool**: Top 10 hybrid + 5 unique semantic
            - **Corpus**: 108,559 aligned Wikipedia sentences
            - **Verifier**: FeatureUnion (word + char TF-IDF) + Logistic Regression
            - **Classes**: `SUPPORTS`, `REFUTES`, `NOT ENOUGH INFO`
            """
        )

        st.divider()
        st.subheader("⚙️ Verification Strategy")
        strategy_option = st.radio(
            "Evidence Grouping Strategy:",
            [
                "Article-Grouped (Matches Training)",
                "Per-Sentence",
            ],
            index=0,
            help=(
                "Article-Grouped clusters sentences from the same article into a unified evidence set, "
                "mirroring the training construction. Per-Sentence evaluates every retrieved sentence independently."
            ),
        )
        grouping_strategy = "article_grouped" if "Article-Grouped" in strategy_option else "per_sentence"

        st.divider()
        st.subheader("📝 Try Sample Claims")
        sample_claims = [
            ("Fox 2000 Pictures released the film Soul Food.", "SUPPORTS example"),
            ("Savages was exclusively a German film.", "REFUTES example"),
            ("Tilda Swinton is a vegan.", "NOT ENOUGH INFO example"),
            (
                "The Carolina Panthers' inaugural NFL season was in 1995.",
                "Challenging sports claim",
            ),
        ]

        for text, desc in sample_claims:
            if st.button(f"📌 {text[:38]}...", help=f"{desc}: {text}", use_container_width=True):
                st.session_state["claim_input"] = text

    # Prepopulate input if selected from sidebar
    default_text = st.session_state.get(
        "claim_input",
        "The Carolina Panthers' inaugural NFL season was in 1995.",
    )

    with st.form("claim_form"):
        claim = st.text_area(
            "Claim to verify:",
            value=default_text,
            placeholder="Enter a factual claim to evaluate against the local corpus...",
            height=90,
        )
        col1, col2 = st.columns([1, 4])
        with col1:
            submitted = st.form_submit_button(
                "Verify Claim",
                type="primary",
                use_container_width=True,
            )
        with col2:
            st.caption(f"Strategy: **{strategy_option}** · Format: `claim: ... evidence: ...`")

    if submitted:
        claim_cleaned = claim.strip()
        if not claim_cleaned:
            st.warning("⚠️ Please enter a non-empty claim before clicking Verify.")
            st.stop()

        try:
            with st.spinner("1/3 Loading cached pipeline artifacts..."):
                (
                    verifier,
                    vectorizer,
                    matrix,
                    corpus,
                    embeddings,
                    semantic_model,
                ) = load_pipeline()

            with st.spinner("2/3 Performing hybrid lexical and semantic retrieval..."):
                evidence = retrieve_evidence(
                    claim_cleaned,
                    vectorizer,
                    matrix,
                    corpus,
                    embeddings,
                    semantic_model,
                    top_k=TOP_K,
                    final_k=FINAL_EVIDENCE_K,
                )

            with st.spinner("3/3 Verifying claim against retrieved evidence..."):
                summary, evidence_results = verify_claim(
                    claim_cleaned,
                    evidence,
                    verifier,
                    grouping_strategy=grouping_strategy,
                )

            st.divider()

            if summary is None or not evidence_results:
                st.warning(
                    "⚠️ No matching evidence was retrieved from the local corpus. "
                    "ContextShield cannot form a judgment for this claim."
                )
                st.stop()

            # Display Analysis Summary
            st.header("Analysis Result")

            col_label, col_margin, col_sets, col_count = st.columns([2, 1, 1, 1])
            with col_label:
                pred_label = summary["label"]
                if pred_label == "SUPPORTS":
                    st.success(f"### Provisional Verdict: **{pred_label}**")
                elif pred_label == "REFUTES":
                    st.error(f"### Provisional Verdict: **{pred_label}**")
                else:
                    st.warning(f"### Provisional Verdict: **{pred_label}**")

            with col_margin:
                st.metric("Top Class Score", f"{summary['predicted_class_score']:.3f}")

            with col_sets:
                st.metric("Evidence Sets", f"{summary['evidence_sets_count']}")

            with col_count:
                st.metric("Sentences Evaluated", f"{summary['evidence_count']}")

            st.caption(
                f"Prediction determined by the highest mean score across {summary['evidence_sets_count']} "
                f"evidence set(s) using **{summary['grouping_strategy']}** grouping. "
                f"Decision margin over second class: {summary['margin']:.3f}."
            )

            # Class Score Distribution
            st.subheader("Mean Verifier Scores Across Evidence Sets")
            col_s1, col_s2, col_s3 = st.columns(3)
            score_cols = {
                "SUPPORTS": col_s1,
                "REFUTES": col_s2,
                "NOT ENOUGH INFO": col_s3,
            }

            for label, col in score_cols.items():
                val = summary["scores"].get(label, 0.0)
                with col:
                    st.markdown(f"**{label}**")
                    st.markdown(f"### `{val:.3f}`")
                    st.progress(max(0.0, min(1.0, val)))

            # Article-Level Evidence Sets (if article_grouped)
            if summary.get("article_groups"):
                st.divider()
                st.subheader(f"Article Evidence Sets ({len(summary['article_groups'])} sets)")
                st.caption(
                    "Sentences sharing the same source article are joined with spaces to form coherent evidence sets, "
                    "mirroring the training data construction."
                )
                for g in summary["article_groups"]:
                    g_heading = (
                        f"Article Set {g['rank']} · {g['article_id']} ({g['sentence_count']} sentence(s)) · "
                        f"Verdict: {g['predicted_label']} ({g['model_score']:.3f})"
                    )
                    with st.expander(g_heading, expanded=(g["rank"] <= 2)):
                        st.markdown(f"> *\"{g['text']}\"*")
                        c1, c2 = st.columns(2)
                        with c1:
                            st.write(f"- Source Article: `{g['article_id']}`")
                            st.write(f"- Sentences: `{g['sentence_ids']}`")
                        with c2:
                            st.write(f"- Group Verdict: **{g['predicted_label']}**")
                            st.write(f"- Model Score: `{g['model_score']:.3f}`")

            # Retrieved Evidence Inspection
            st.divider()
            st.subheader(f"Retrieved Individual Evidence Passages ({len(evidence_results)} sentences)")
            st.caption(
                "Evidence is ranked by retrieval similarity (RRF fusion or dense semantic cosine). "
                "Individual sentence verdicts represent how the verifier evaluates each pair in isolation."
            )

            for item in evidence_results:
                retrieved_via = ", ".join(item.get("retrieved_by", []))
                score_type = item.get("retrieval_score_type", "unknown")
                score_val = item.get("retrieval_score", 0.0)

                if score_type == "reciprocal_rank_fusion":
                    score_badge = f"RRF Score: {score_val:.4f}"
                else:
                    score_badge = f"Cosine: {score_val:.4f}"

                heading = (
                    f"Rank {item['rank']:02d} · [{score_badge}] · {item['article_id']} "
                    f"(Sentence {item['sentence_id']}) · Retrieved via {retrieved_via.upper()}"
                )

                with st.expander(heading, expanded=(item["rank"] <= 3)):
                    st.markdown(f"> *\"{item['text']}\"*")

                    c_info, c_pred = st.columns([2, 2])
                    with c_info:
                        st.markdown("**Retrieval Details**")
                        st.write(f"- Channel: `{retrieved_via}`")
                        st.write(f"- Primary Score: `{score_val:.4f}` ({score_type})")
                        st.write(f"- TF-IDF lexical score: `{item.get('tfidf_score', 0.0):.4f}`")
                        st.write(f"- Dense semantic cosine: `{item.get('semantic_score', 0.0):.4f}`")

                    with c_pred:
                        sent_pred = item.get("predicted_label", "UNKNOWN")
                        st.markdown(f"**Sentence-level Verdict**: `{sent_pred}`")
                        sent_scores = item.get("class_scores", {})
                        for s_cls in ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]:
                            s_sc = sent_scores.get(s_cls, 0.0)
                            st.write(f"- {s_cls}: `{s_sc:.3f}`")

            st.divider()
            st.caption(
                "🛡️ **Methodological Context**: Evidence retrieval and verification are experimental. "
                "Linear model decisions are sensitive to vocabulary and phrasing artifacts from the training set. "
                "Always cross-reference primary sources for verification of factual claims."
            )

        except Exception as exc:
            st.error(
                "An unexpected error occurred while analyzing the claim. "
                "Please ensure all required data artifacts are accessible."
            )
            st.exception(exc)

    else:
        st.markdown("### How ContextShield Works")
        st.markdown(
            """
            1. **Query Processing**: The input claim is vectorized using word-level TF-IDF and encoded with `all-MiniLM-L6-v2`.
            2. **Hybrid Retrieval**: Top candidates are scored against the 108,559-sentence corpus with both sparse TF-IDF and dense embeddings.
            3. **Reciprocal Rank Fusion**: Both rankings are fused with $k=60$ to yield top hybrid evidence passages, followed by high-similarity semantic candidates up to 15 total sentences.
            4. **Evidence Grouping & Verification**: Retrieved evidence is formatted using the verified Colab training format (`claim: {claim} evidence: {evidence_text}`). Evidence can be evaluated as coherent article-level sets or individual sentences.
            5. **Aggregation & Display**: Class probabilities are pooled to produce a provisional verdict, article set breakdowns, and per-evidence sentence inspection.
            """
        )

    st.caption("ContextShield · NLP Claim Verification Prototype")


if __name__ == "__main__":
    main()
