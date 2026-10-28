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
    HF_REPO_ID,
    RRF_K,
    TOP_K,
    check_missing_artifacts,
    ensure_artifacts_exist,
    format_claim_evidence,
    generate_explanation,
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
    missing = check_missing_artifacts(DATA_DIR)
    if missing:
        hf_token = None
        try:
            if hasattr(st, "secrets") and "HF_TOKEN" in st.secrets:
                hf_token = st.secrets["HF_TOKEN"]
        except Exception:
            pass
        ensure_artifacts_exist(DATA_DIR, token=hf_token)
    return _load_pipeline_raw(DATA_DIR, auto_download=False)


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
    st.subheader("Evidence-Based Fact Verification Prototype")

    st.markdown(
        "ContextShield verifies factual claims against a 108,559-sentence Wikipedia corpus. "
        "It uses hybrid retrieval (word TF-IDF + dense embeddings with Reciprocal Rank Fusion) "
        "and a trained three-class linear verifier to evaluate whether evidence **SUPPORTS**, "
        "**REFUTES**, or provides **NOT ENOUGH INFO** for the claim."
    )

    # Scientific integrity notice
    st.info(
        "ℹ️ **Evaluation & Scientific Integrity Notice**: Model inputs are structured in the verified "
        "training format (`claim: {claim} evidence: {evidence_text}`). Scores reflect feature activations "
        "over retrieved Wikipedia passages. They are **uncalibrated classifier outputs**, not real-world "
        "probabilities of truth. The system abstains when evidence is insufficient or non-decisive."
    )

    # Ensure verification model and corpus artifacts exist locally on deployment boot
    missing_artifacts = check_missing_artifacts(DATA_DIR)
    if missing_artifacts:
        missing_count = len(missing_artifacts)
        with st.spinner(
            f"📥 Downloading {missing_count} missing verification artifact(s) from Hugging Face "
            f"({', '.join(missing_artifacts)})... This happens once on deployment startup."
        ):
            try:
                hf_token = None
                try:
                    if hasattr(st, "secrets") and "HF_TOKEN" in st.secrets:
                        hf_token = st.secrets["HF_TOKEN"]
                except Exception:
                    pass
                ensure_artifacts_exist(DATA_DIR, token=hf_token)
                st.success("✅ ContextShield artifacts downloaded and verified successfully!")
            except Exception as exc:
                st.error(
                    f"❌ Error downloading model artifacts from Hugging Face (`{HF_REPO_ID}`):\n\n"
                    f"```{exc}```\n\n"
                    "Please ensure the Hugging Face repository is accessible or provide a read token in Streamlit Secrets."
                )
                st.stop()

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

        aggregation_option = st.radio(
            "Aggregation Method:",
            [
                "Relevance-Weighted (Reciprocal Rank)",
                "Uniform Mean (Baseline)",
            ],
            index=0,
            help=(
                "Relevance-Weighted weights each evidence candidate by reciprocal retrieval rank (1 / rank), "
                "preventing irrelevant or uninformative low-ranked background passages from diluting decisive evidence. "
                "Uniform Mean computes an unweighted average across all evidence sets."
            ),
        )
        aggregation_method = "relevance_weighted" if "Relevance-Weighted" in aggregation_option else "mean"

        st.divider()
        st.subheader("📝 Demonstration Claims")
        sample_claims = [
            (
                "Fox 2000 Pictures released the film Soul Food.",
                "🟢 Conclusive SUPPORTS example",
                "SUPPORTS",
            ),
            (
                "Savages was exclusively a German film.",
                "🔴 Conclusive REFUTES example",
                "REFUTES",
            ),
            (
                "Tilda Swinton is a vegan.",
                "🟡 NOT ENOUGH INFO (Unmentioned entity fact)",
                "NOT ENOUGH INFO",
            ),
            (
                "The Carolina Panthers' inaugural NFL season was in 1995.",
                "⚖️ Borderline sports claim (Phrasing sensitivity)",
                "NOT ENOUGH INFO",
            ),
        ]

        def on_select_claim(selected_claim_text):
            st.session_state["claim_input_text"] = selected_claim_text
            st.session_state["auto_verify_trigger"] = True

        for text, desc, badge in sample_claims:
            st.button(
                f"📌 {text[:38]}...",
                help=f"{desc}\n\nFull Claim: \"{text}\"",
                use_container_width=True,
                on_click=on_select_claim,
                args=(text,),
            )

    # Initialize text area state
    if "claim_input_text" not in st.session_state:
        st.session_state["claim_input_text"] = "The Carolina Panthers' inaugural NFL season was in 1995."

    with st.form("claim_form"):
        claim = st.text_area(
            "Claim to verify:",
            key="claim_input_text",
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
            st.caption(
                f"Strategy: **{strategy_option}** · "
                f"Aggregation: **{aggregation_option.split(' ')[0]}** · "
                f"Format: `claim: ... evidence: ...`"
            )

    should_verify = submitted or st.session_state.pop("auto_verify_trigger", False)

    if should_verify:
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
                    aggregation_method=aggregation_method,
                )

            st.divider()

            if summary is None or not evidence_results:
                st.warning(
                    "⚠️ No matching evidence was retrieved from the local corpus. "
                    "ContextShield cannot form a judgment for this claim."
                )
                st.stop()

            # Generate grounded, factual explanation
            explanation = generate_explanation(claim_cleaned, summary, evidence_results)

            # Display Analysis Summary Header
            st.header("Analysis Result")

            # High-contrast Verdict Banner
            pred_label = summary["label"]
            if pred_label == "SUPPORTS":
                st.success(
                    f"### 🟢 Verdict: **SUPPORTS**\n\n"
                    f"**{explanation['verdict_headline']}** — "
                    f"The retrieved evidence from the local Wikipedia corpus confirms this claim."
                )
            elif pred_label == "REFUTES":
                st.error(
                    f"### 🔴 Verdict: **REFUTES**\n\n"
                    f"**{explanation['verdict_headline']}** — "
                    f"The retrieved evidence from the local Wikipedia corpus contradicts this claim."
                )
            else:
                banner_detail = (
                    "Relevant evidence was retrieved, but the classifier could not confirm the claim."
                    if explanation.get("has_relevance_warning")
                    else "The retrieved Wikipedia passages do not contain sufficient factual evidence to confirm or refute this claim."
                )
                st.warning(
                    f"### 🟡 Verdict: **NOT ENOUGH INFO**\n\n"
                    f"**{explanation['verdict_headline']}** — {banner_detail}"
                )

            # Key Evidence Callout Box (Prominently displayed)
            key_item = explanation.get("key_passage") or (evidence_results[0] if evidence_results else None)
            if key_item:
                art_name = key_item["article_id"].replace("_", " ")
                ret_via = ", ".join(key_item.get("retrieved_by", ["hybrid"])).upper()
                st.markdown("#### 🔍 Primary Evidence Passage")
                st.info(
                    f"> *\"{key_item['text'].strip()}\"*\n\n"
                    f"**Source Article**: Wikipedia *{art_name}* (Sentence ID: {key_item['sentence_id']}) · "
                    f"**Retrieval Rank**: #{key_item['rank']:02d} · "
                    f"**Channel**: {ret_via} · "
                    f"**Passage Stance**: `{key_item.get('predicted_label', 'UNKNOWN')}` ({key_item.get('model_score', 0.0):.3f})"
                )

            # Plain-English Grounded Explanation
            st.markdown("#### 💡 Explanation")
            st.markdown(explanation["explanation_text"])

            if explanation["has_borderline_margin"]:
                st.caption(
                    "⚖️ **Borderline Margin Notice**: The classifier margin between the leading classes is narrow (< 0.10). "
                    "The system transparently reports NOT ENOUGH INFO to avoid asserting false certainty on sensitive phrasing."
                )

            # Metric Summary Cards
            st.divider()
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            with col_m1:
                st.metric("Leading Score", f"{summary['predicted_class_score']:.3f}")
            with col_m2:
                st.metric("Decision Margin", f"{summary['margin']:.3f}")
            with col_m3:
                st.metric("Certainty Tier", explanation["certainty_tier"])
            with col_m4:
                st.metric("Evidence Evaluated", f"{summary['evidence_count']} sentences")

            st.caption(
                f"Aggregated across {summary['evidence_sets_count']} evidence set(s) using "
                f"**{summary['grouping_strategy']}** grouping and **{summary['aggregation_method']}** weighting. "
                f"{explanation['certainty_description']}"
            )

            # Class Score Distribution Bars
            st.subheader("Aggregated Verifier Scores")
            col_s1, col_s2, col_s3 = st.columns(3)
            score_cols = {
                "SUPPORTS": col_s1,
                "REFUTES": col_s2,
                "NOT ENOUGH INFO": col_s3,
            }
            for label_name, col in score_cols.items():
                val = summary["scores"].get(label_name, 0.0)
                with col:
                    st.markdown(f"**{label_name}**")
                    st.markdown(f"### `{val:.3f}`")
                    st.progress(max(0.0, min(1.0, val)))

            # Tabs for Detailed Evidence Inspection
            st.divider()
            tab_articles, tab_sentences, tab_diagnostics = st.tabs([
                f"📚 Article Evidence Sets ({summary['evidence_sets_count']})",
                f"📄 All Retrieved Sentences ({len(evidence_results)})",
                "🔬 System Diagnostics & Methodology",
            ])

            with tab_articles:
                st.info(
                    "💡 **How Article Evidence Sets Work**: Sentences sharing the same source article are concatenated into "
                    "multi-sentence passages (mirroring the original model training format) and weighted by reciprocal retrieval rank.\n\n"
                    "**Why an article set's verdict can differ from the overall verdict:** The overall verdict is formed by pooling continuous "
                    "probability distributions across all article sets—not by a simple majority vote of discrete group labels. "
                    "A concatenated passage can yield `NOT ENOUGH INFO` due to multi-sentence context dilution, yet its individual "
                    "sentences and corroborating article sets contribute decisive probability mass toward the overall verdict."
                )
                if summary.get("article_groups"):
                    for g in summary["article_groups"]:
                        g_weight_pct = g.get("weight", 0.0) * 100
                        g_heading = (
                            f"Article Set {g['rank']:02d} · {g['article_id'].replace('_', ' ')} "
                            f"({g['sentence_count']} sent) · Weight: {g_weight_pct:.1f}% · Verdict: {g['predicted_label']} ({g['model_score']:.3f})"
                        )
                        with st.expander(g_heading, expanded=(g["rank"] <= 2)):
                            st.markdown(f"> *\"{g['text']}\"*")
                            c1, c2 = st.columns(2)
                            with c1:
                                st.write(f"- Source Article: `{g['article_id']}`")
                                st.write(f"- Sentence IDs: `{g['sentence_ids']}`")
                                st.write(f"- Min Retrieval Rank: `#{g.get('min_rank', g['rank'])}`")
                            with c2:
                                st.write(f"- Group Plurality Verdict: **{g['predicted_label']}** (`{g['model_score']:.3f}`)")
                                st.write(f"- Voting Weight: `{g_weight_pct:.1f}%`")
                                c_sc = g.get("class_scores", {})
                                st.write(
                                    f"- Class Scores: SUP: `{c_sc.get('SUPPORTS', 0.0):.3f}` | "
                                    f"NEI: `{c_sc.get('NOT ENOUGH INFO', 0.0):.3f}` | "
                                    f"REF: `{c_sc.get('REFUTES', 0.0):.3f}`"
                                )

                            if g.get("predicted_label") != summary["label"]:
                                st.caption(
                                    f"ℹ️ Note: This article set's plurality verdict ({g['predicted_label']}) differs from "
                                    f"the overall verdict ({summary['label']}). The overall verdict reflects reciprocal-rank "
                                    f"probability pooling across all retrieved article sets."
                                )

                            c_sents = g.get("constituent_sentences", [])
                            if c_sents:
                                st.markdown("**Constituent Sentences in this Set:**")
                                for cs in c_sents:
                                    cs_pred = cs.get("predicted_label", "UNKNOWN")
                                    cs_sc = cs.get("model_score", 0.0)
                                    st.markdown(
                                        f"- **Rank #{cs.get('rank', 0):02d}** (Sent {cs.get('sentence_id', '?')}): "
                                        f"`{cs_pred}` ({cs_sc:.3f}) — *\"{cs.get('text', '').strip()}\"*"
                                    )
                else:
                    st.info("Per-sentence strategy active; individual sentences are evaluated directly.")

            with tab_sentences:
                st.caption(
                    "Evidence ranked by hybrid retrieval similarity (RRF fusion or dense semantic cosine). "
                    "Individual verdicts represent how the verifier evaluates each passage in isolation."
                )
                for item in evidence_results:
                    retrieved_via = ", ".join(item.get("retrieved_by", [])).upper()
                    score_type = item.get("retrieval_score_type", "unknown")
                    score_val = item.get("retrieval_score", 0.0)
                    item_weight_pct = item.get("weight", 0.0) * 100

                    if score_type == "reciprocal_rank_fusion":
                        score_badge = f"RRF: {score_val:.4f}"
                    else:
                        score_badge = f"Cosine: {score_val:.4f}"

                    heading = (
                        f"Rank #{item['rank']:02d} · [{score_badge}] · {item['article_id'].replace('_', ' ')} "
                        f"(Sent {item['sentence_id']}) · Weight: {item_weight_pct:.1f}% · Verdict: {item.get('predicted_label', 'UNKNOWN')}"
                    )

                    with st.expander(heading, expanded=(item["rank"] <= 2)):
                        st.markdown(f"> *\"{item['text']}\"*")
                        c_info, c_pred = st.columns([2, 2])
                        with c_info:
                            st.markdown("**Retrieval Details**")
                            st.write(f"- Channel: `{retrieved_via}`")
                            st.write(f"- Primary Score: `{score_val:.4f}` ({score_type})")
                            st.write(f"- Voting Weight: `{item_weight_pct:.1f}%`")
                            st.write(f"- TF-IDF lexical score: `{item.get('tfidf_score', 0.0):.4f}`")
                            st.write(f"- Dense semantic cosine: `{item.get('semantic_score', 0.0):.4f}`")
                        with c_pred:
                            sent_pred = item.get("predicted_label", "UNKNOWN")
                            st.markdown(f"**Sentence-level Verdict**: `{sent_pred}`")
                            sent_scores = item.get("class_scores", {})
                            for s_cls in ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]:
                                s_sc = sent_scores.get(s_cls, 0.0)
                                st.write(f"- {s_cls}: `{s_sc:.3f}`")

            with tab_diagnostics:
                st.markdown("#### 🔬 Methodological & System Diagnostics")
                st.markdown(
                    """
                    - **Verified Training Input Format**: `claim: {claim} evidence: {evidence_text}`
                    - **Verifier Architecture**: Scikit-learn Pipeline with `FeatureUnion` (word n-grams 1-2, character n-grams 3-5) and `LogisticRegression(class_weight='balanced')`.
                    - **Aggregation Formula**: Convex reciprocal-rank pooling:
                      $$\\bar{P} = \\frac{\\sum_k w_k P_k}{\\sum_k w_k}, \\quad w_k = \\frac{1}{\\text{rank}_k}$$
                    - **Calibration Disclaimer**: Scores are linear classifier feature activations over textual evidence, not real-world probabilities of truth.
                    - **Corpus Coverage**: Evaluates over 108,559 aligned Wikipedia sentences. Claims asserting facts not covered in this corpus will safely default to NOT ENOUGH INFO.
                    """
                )

            st.divider()
            st.caption(
                "🛡️ **ContextShield NLP Prototype** · Evidence retrieval and verification are experimental. "
                "Always cross-reference primary sources."
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
            5. **Aggregation & Display**: Class probabilities are pooled using reciprocal rank weighting to produce an understandable verdict, primary evidence quote, and honest explanation.
            """
        )

    st.caption("ContextShield · NLP Claim Verification Prototype")


if __name__ == "__main__":
    main()
