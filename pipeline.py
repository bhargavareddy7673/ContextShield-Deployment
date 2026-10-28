"""ContextShield pipeline: artifact loading, hybrid retrieval, and claim verification."""

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import scipy.sparse as sp
from sentence_transformers import SentenceTransformer

# --------------------------------------------------
# Default configuration
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

VERIFIER_PATH = DATA_DIR / "evidence_verifier_3class.joblib"
VECTORIZER_PATH = DATA_DIR / "retrieval_tfidf_word.joblib"
MATRIX_PATH = DATA_DIR / "retrieval_tfidf_matrix.npz"
CORPUS_PATH = DATA_DIR / "train_retrieval_corpus.jsonl"
EMBEDDINGS_PATH = DATA_DIR / "train_corpus_embeddings.npy"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

RRF_K = 60
TOP_K = 10
FINAL_EVIDENCE_K = 15


def format_claim_evidence(claim: str, evidence_text: str) -> str:
    """Format claim and evidence pair for the saved verifier model.

    The saved verifier pipeline (FeatureUnion of word and char TF-IDF + Logistic Regression)
    was trained using the format:
        'claim: ' + row['claim'] + ' evidence: ' + row['evidence_text']
    where evidence_text represents evidence sentences joined with spaces.
    """
    return f"claim: {claim.strip()} evidence: {evidence_text.strip()}"


def group_evidence_by_article(evidence: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Group retrieved evidence passages by article_id.

    In the training data construction, all sentences belonging to a single evidence
    set were joined with spaces: " ".join(s["text"] for s in sentences).
    This grouping clusters retrieved sentences by their source article to form
    coherent evidence candidates, preserving the best retrieval rank per article.
    """
    if not evidence:
        return []

    article_groups: Dict[str, Dict[str, Any]] = {}
    for idx, item in enumerate(evidence, start=1):
        art_id = item["article_id"]
        rnk = item.get("rank", idx)
        if art_id not in article_groups:
            article_groups[art_id] = {
                "article_id": art_id,
                "sentence_ids": [item["sentence_id"]],
                "texts": [item["text"].strip()],
                "items": [item],
                "min_rank": rnk,
            }
        else:
            article_groups[art_id]["sentence_ids"].append(item["sentence_id"])
            article_groups[art_id]["texts"].append(item["text"].strip())
            article_groups[art_id]["items"].append(item)
            if rnk < article_groups[art_id]["min_rank"]:
                article_groups[art_id]["min_rank"] = rnk

    sorted_groups = sorted(article_groups.values(), key=lambda g: g["min_rank"])
    return [
        {
            "rank": idx,
            "min_rank": g["min_rank"],
            "article_id": g["article_id"],
            "sentence_ids": g["sentence_ids"],
            "text": " ".join(g["texts"]),
            "sentence_count": len(g["texts"]),
            "constituent_items": g["items"],
        }
        for idx, g in enumerate(sorted_groups, start=1)
    ]


def load_pipeline(data_dir: Optional[Path] = None) -> Tuple[
    Any, Any, sp.csr_matrix, List[Dict[str, Any]], np.ndarray, SentenceTransformer
]:
    """Load all saved model artifacts and verify integrity and alignment.

    Returns:
        (verifier, vectorizer, matrix, corpus, embeddings, semantic_model)
    """
    data = Path(data_dir) if data_dir else DATA_DIR
    verifier_path = data / "evidence_verifier_3class.joblib"
    vectorizer_path = data / "retrieval_tfidf_word.joblib"
    matrix_path = data / "retrieval_tfidf_matrix.npz"
    corpus_path = data / "train_retrieval_corpus.jsonl"
    embeddings_path = data / "train_corpus_embeddings.npy"

    required = [
        verifier_path,
        vectorizer_path,
        matrix_path,
        corpus_path,
        embeddings_path,
    ]

    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "Required ContextShield artifact files are missing: "
            + ", ".join(missing)
        )

    verifier = joblib.load(verifier_path)
    vectorizer = joblib.load(vectorizer_path)
    matrix = sp.load_npz(matrix_path)

    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus = [json.loads(line) for line in f if line.strip()]

    embeddings = np.load(embeddings_path, mmap_mode="r")
    semantic_model = SentenceTransformer(
        EMBEDDING_MODEL_NAME,
        device="cpu",
    )

    # Dimensionality and alignment checks
    if matrix.shape[0] != len(corpus):
        raise ValueError(
            f"Retrieval matrix rows ({matrix.shape[0]}) and corpus count ({len(corpus)}) differ."
        )

    if matrix.shape[1] != len(vectorizer.vocabulary_):
        raise ValueError(
            f"Retrieval matrix columns ({matrix.shape[1]}) and vocabulary size ({len(vectorizer.vocabulary_)}) differ."
        )

    if embeddings.shape[0] != len(corpus):
        raise ValueError(
            f"Semantic embeddings count ({embeddings.shape[0]}) and corpus count ({len(corpus)}) differ."
        )

    if embeddings.shape[1] != semantic_model.get_embedding_dimension():
        raise ValueError(
            f"Embedding dim ({embeddings.shape[1]}) does not match model ({semantic_model.get_embedding_dimension()})."
        )

    return (
        verifier,
        vectorizer,
        matrix,
        corpus,
        embeddings,
        semantic_model,
    )


def retrieve_evidence(
    claim: str,
    vectorizer: Any,
    matrix: sp.csr_matrix,
    corpus: List[Dict[str, Any]],
    embeddings: np.ndarray,
    semantic_model: SentenceTransformer,
    top_k: int = TOP_K,
    final_k: int = FINAL_EVIDENCE_K,
) -> List[Dict[str, Any]]:
    """Retrieve evidence using hybrid TF-IDF + semantic search with RRF fusion.

    1. Retrieves top_k lexical candidates via word TF-IDF.
    2. Retrieves semantic candidates via dense embeddings.
    3. Fuses both candidate lists using Reciprocal Rank Fusion (RRF_K=60) to keep hybrid top_k.
    4. Appends unique high-scoring semantic candidates up to final_k (default 15).
    """
    if not claim or not claim.strip():
        return []

    clean_claim = claim.strip()
    source_k = min(top_k, len(corpus))
    if source_k == 0:
        return []

    # 1. Retrieve candidates using TF-IDF.
    query_vector = vectorizer.transform([clean_claim])
    tfidf_scores = (matrix @ query_vector.T).toarray().ravel()
    tfidf_valid = np.flatnonzero(tfidf_scores > 0)

    if len(tfidf_valid) > 0:
        tfidf_indices = tfidf_valid[
            np.argsort(tfidf_scores[tfidf_valid])[::-1][:source_k]
        ]
    else:
        tfidf_indices = np.array([], dtype=int)

    # 2. Retrieve candidates using semantic similarity.
    # Retrieve enough semantic candidates so step 5 can append up to final_k unique items.
    semantic_pool_k = min(max(top_k, final_k, 25), len(corpus))
    query_embedding = semantic_model.encode(
        [clean_claim],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )[0]

    semantic_scores = embeddings @ query_embedding
    semantic_indices = np.argsort(semantic_scores)[::-1][:semantic_pool_k]

    # 3. Fuse both ranked lists using reciprocal-rank fusion.
    candidates: Dict[Tuple[str, int], Dict[str, Any]] = {}

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

    # Use the top source_k semantic candidates in the RRF fusion pool
    for rank, index in enumerate(semantic_indices[:source_k], start=1):
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

    # 4. Keep the hybrid top_k.
    hybrid_top = fused[:source_k]

    selected_keys = {
        (
            corpus[item["index"]]["article_id"],
            corpus[item["index"]]["sentence_id"],
        )
        for item in hybrid_top
    }

    selected: List[Dict[str, Any]] = []

    for item in hybrid_top:
        idx = item["index"]
        record = corpus[idx]
        selected.append(
            {
                "rank": len(selected) + 1,
                "article_id": record["article_id"],
                "sentence_id": record["sentence_id"],
                "text": record["text"],
                "retrieval_score": float(item["rrf_score"]),
                "retrieval_score_type": "reciprocal_rank_fusion",
                "retrieved_by": item["retrieved_by"],
                "tfidf_score": float(tfidf_scores[idx]) if idx < len(tfidf_scores) else 0.0,
                "semantic_score": float(semantic_scores[idx]),
            }
        )

    # 5. Append unique semantic candidates until we reach final_k.
    target_k = min(final_k, len(corpus))
    for index in semantic_indices:
        if len(selected) >= target_k:
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
                "tfidf_score": float(tfidf_scores[index]) if index < len(tfidf_scores) else 0.0,
                "semantic_score": float(semantic_scores[index]),
            }
        )

    return selected


def verify_claim(
    claim: str,
    evidence: List[Dict[str, Any]],
    verifier: Any,
    grouping_strategy: str = "article_grouped",
    aggregation_method: str = "relevance_weighted",
) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    """Verify a claim against retrieved evidence passages using the saved verifier.

    Args:
        claim: The factual claim string.
        evidence: List of retrieved evidence sentence dicts.
        verifier: Trained scikit-learn verifier pipeline.
        grouping_strategy: Strategy to construct evidence sets for aggregate verdict:
            - 'article_grouped' (default): Group sentences by article_id, joining with
              spaces as in the training evidence sets.
            - 'per_sentence': Treat each retrieved sentence as an independent evidence candidate.
        aggregation_method: Probability pooling method across evidence sets:
            - 'relevance_weighted' (default): Weight each evidence candidate by its reciprocal
              retrieval rank (w_k = 1 / rank_k). This prevents irrelevant or uninformative
              low-ranked background passages from diluting decisive, top-ranked evidence.
            - 'mean': Uniform unweighted mean pooling across all evidence sets (baseline).

    Returns:
        (summary_dict, evidence_results_list)
        where summary_dict is None if claim or evidence is empty.
    """
    if not claim or not claim.strip():
        return None, []

    if not evidence:
        return None, []

    clean_claim = claim.strip()
    classes = np.asarray(verifier.classes_)

    # 1. Evaluate each individual retrieved sentence for detailed inspection
    sent_pairs = [
        format_claim_evidence(clean_claim, item["text"])
        for item in evidence
    ]
    sent_probabilities = verifier.predict_proba(sent_pairs)

    # 2. Compute sentence weights based on aggregation_method
    if aggregation_method == "relevance_weighted":
        sent_weights = np.array(
            [1.0 / max(1, item.get("rank", idx)) for idx, item in enumerate(evidence, start=1)],
            dtype=float,
        )
    else:  # 'mean'
        sent_weights = np.ones(len(evidence), dtype=float)

    sent_weights_normalized = sent_weights / (sent_weights.sum() if sent_weights.sum() > 0 else 1.0)

    results: List[Dict[str, Any]] = []
    for idx, (item, probs) in enumerate(zip(evidence, sent_probabilities)):
        best = int(np.argmax(probs))
        result = dict(item)
        result["predicted_label"] = str(classes[best])
        result["model_score"] = float(probs[best])
        result["weight"] = float(sent_weights_normalized[idx])
        result["class_scores"] = {
            str(cls): float(prob)
            for cls, prob in zip(classes, probs)
        }
        results.append(result)

    # 3. Form evidence sets according to grouping_strategy for aggregate verdict
    article_summaries = []
    if grouping_strategy == "article_grouped":
        article_groups = group_evidence_by_article(evidence)
        art_pairs = [
            format_claim_evidence(clean_claim, g["text"])
            for g in article_groups
        ]
        art_probabilities = verifier.predict_proba(art_pairs)

        if aggregation_method == "relevance_weighted":
            art_weights = np.array(
                [1.0 / max(1, g.get("min_rank", g.get("rank", idx))) for idx, g in enumerate(article_groups, start=1)],
                dtype=float,
            )
            w_sum = art_weights.sum()
            agg_probabilities = (art_probabilities * art_weights[:, None]).sum(axis=0) / (w_sum if w_sum > 0 else 1.0)
        else:  # 'mean'
            art_weights = np.ones(len(article_groups), dtype=float)
            agg_probabilities = art_probabilities.mean(axis=0)

        art_weights_normalized = art_weights / (art_weights.sum() if art_weights.sum() > 0 else 1.0)

        for idx, (g, probs) in enumerate(zip(article_groups, art_probabilities)):
            best = int(np.argmax(probs))
            group_sent_ids = set(str(s) for s in g["sentence_ids"])
            matching_sentences = [
                r for r in results
                if str(r["article_id"]) == str(g["article_id"]) and str(r["sentence_id"]) in group_sent_ids
            ]
            article_summaries.append({
                "rank": g["rank"],
                "min_rank": g.get("min_rank", g["rank"]),
                "article_id": g["article_id"],
                "sentence_ids": g["sentence_ids"],
                "sentence_count": g["sentence_count"],
                "text": g["text"],
                "constituent_sentences": matching_sentences,
                "weight": float(art_weights_normalized[idx]),
                "predicted_label": str(classes[best]),
                "model_score": float(probs[best]),
                "class_scores": {
                    str(cls): float(prob) for cls, prob in zip(classes, probs)
                },
            })
    else:  # per_sentence
        if aggregation_method == "relevance_weighted":
            agg_probabilities = (sent_probabilities * sent_weights[:, None]).sum(axis=0) / sent_weights.sum()
        else:  # 'mean'
            agg_probabilities = sent_probabilities.mean(axis=0)

    best_index = int(np.argmax(agg_probabilities))

    # Calculate margin between highest and second highest score
    sorted_probs = np.sort(agg_probabilities)[::-1]
    margin = float(sorted_probs[0] - sorted_probs[1]) if len(sorted_probs) > 1 else 1.0

    summary = {
        "label": str(classes[best_index]),
        "predicted_class_score": float(agg_probabilities[best_index]),
        "margin": margin,
        "evidence_count": len(results),
        "evidence_sets_count": len(article_summaries) if article_summaries else len(results),
        "grouping_strategy": grouping_strategy,
        "aggregation_method": aggregation_method,
        "scores": {
            str(label): float(prob)
            for label, prob in zip(classes, agg_probabilities)
        },
        "article_groups": article_summaries,
    }

    return summary, results


def generate_explanation(
    claim: str,
    summary: Optional[Dict[str, Any]],
    evidence_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Generate an honest, grounded plain-English explanation of the verification verdict.

    Adheres strictly to scientific integrity:
    1. Cites only actual retrieved passages from evidence_results; never invents facts.
    2. Transparently explains borderline decisions and classifier limitations.
    3. Clearly distinguishes between conclusive evidence and insufficient information.
    4. Refrains from presenting linear classifier scores as real-world probabilities of truth.
    """
    if not summary or not evidence_results:
        return {
            "verdict": "NOT ENOUGH INFO",
            "verdict_headline": "No Evidence Retrieved",
            "explanation_text": (
                "The system found no relevant evidence passages in the local Wikipedia corpus "
                "for this claim. ContextShield cannot make a factual determination without evidence."
            ),
            "key_passage": None,
            "certainty_tier": "Insufficient Data",
            "certainty_description": "No evidence was available to score.",
            "has_borderline_margin": False,
        }

    label = summary["label"]
    score = summary["predicted_class_score"]
    margin = summary["margin"]
    scores = summary["scores"]
    evidence_count = summary.get("evidence_count", len(evidence_results))

    # Identify primary passage: Rank 1 retrieved item
    top_passage = evidence_results[0] if evidence_results else None

    # Assess decision certainty / margin tier
    if margin >= 0.25 and score >= 0.50:
        certainty_tier = "Decisive Alignment"
        certainty_desc = f"Strong stance separation: {label} leads by a margin of {margin:.3f} over alternative classes."
        is_borderline = False
    elif margin >= 0.10:
        certainty_tier = "Moderate Alignment"
        certainty_desc = f"Moderate stance separation: {label} leads with a margin of {margin:.3f}."
        is_borderline = False
    else:
        certainty_tier = "Borderline / Narrow Margin"
        certainty_desc = (
            f"Narrow decision margin ({margin:.3f}). The classifier scores are close between the top two "
            "classes, indicating high model sensitivity or mixed evidence."
        )
        is_borderline = True

    # Construct grounded explanations based on label and evidence
    if label == "SUPPORTS":
        verdict_headline = "Evidence Confirms Claim"
        sup_passages = [p for p in evidence_results if p.get("predicted_label") == "SUPPORTS"]
        primary = sup_passages[0] if sup_passages else top_passage
        art_clean = primary["article_id"].replace("_", " ")

        explanation_text = (
            f"ContextShield retrieved {evidence_count} evidence sentence(s) from the corpus. "
            f"The primary supporting evidence from Wikipedia article '{art_clean}' states: "
            f"\"{primary['text'].strip()}\". "
            f"The verifier evaluated this passage in the verified training format ('claim: ... evidence: ...') "
            f"and assigned a consolidated support score of {score:.3f} (margin: {margin:.3f} over other classes)."
        )
        art_groups = summary.get("article_groups", [])
        if art_groups and art_groups[0].get("predicted_label") != label:
            top_g = art_groups[0]
            top_name = top_g["article_id"].replace("_", " ")
            explanation_text += (
                f" Note on evidence aggregation: Although the multi-sentence passage for '{top_name}' "
                f"yields a plurality of {top_g['predicted_label']} ({top_g['model_score']:.3f}) due to "
                f"passage context dilution, its constituent primary sentence directly confirms the claim "
                f"({primary.get('predicted_label', 'SUPPORTS')} {primary.get('model_score', 0.0):.3f}), and "
                f"reciprocal-rank probability pooling across all retrieved article sets yields a consolidated {label} verdict."
            )
        key_passage = primary

    elif label == "REFUTES":
        verdict_headline = "Evidence Contradicts Claim"
        ref_passages = [p for p in evidence_results if p.get("predicted_label") == "REFUTES"]
        primary = ref_passages[0] if ref_passages else top_passage
        art_clean = primary["article_id"].replace("_", " ")

        explanation_text = (
            f"ContextShield retrieved evidence conflicting directly with the claim. "
            f"The primary contradictory passage from Wikipedia article '{art_clean}' states: "
            f"\"{primary['text'].strip()}\". "
            f"The verifier evaluated this conflict and assigned a consolidated refute score of {score:.3f} "
            f"(margin: {margin:.3f} over other classes)."
        )
        art_groups = summary.get("article_groups", [])
        if art_groups and art_groups[0].get("predicted_label") != label:
            top_g = art_groups[0]
            top_name = top_g["article_id"].replace("_", " ")
            explanation_text += (
                f" Note on evidence aggregation: Although the multi-sentence passage for '{top_name}' "
                f"yields a plurality of {top_g['predicted_label']} ({top_g['model_score']:.3f}), "
                f"its constituent primary sentence directly contradicts the claim "
                f"({primary.get('predicted_label', 'REFUTES')} {primary.get('model_score', 0.0):.3f}), and "
                f"reciprocal-rank probability pooling across all retrieved article sets yields a consolidated {label} verdict."
            )
        key_passage = primary

    else:  # NOT ENOUGH INFO
        verdict_headline = "Insufficient Evidence in Corpus"
        key_passage = top_passage
        art_clean = top_passage["article_id"].replace("_", " ") if top_passage else "corpus"

        # Check if primary retrieved passage has high relevance despite NOT ENOUGH INFO verdict
        has_high_relevance = False
        if top_passage:
            sem_sc = float(top_passage.get("semantic_score", 0.0))
            tf_sc = float(top_passage.get("tfidf_score", 0.0))
            if sem_sc >= 0.70 or (sem_sc >= 0.60 and tf_sc >= 0.35):
                has_high_relevance = True

        if has_high_relevance:
            verdict_headline = "Evidence Retrieved but Unconfirmed"
            explanation_text = (
                f"ContextShield retrieved a highly relevant evidence passage from Wikipedia article '{art_clean}' "
                f"(\"{top_passage['text'].strip()}\"), but the classifier assigned an overall verdict of "
                f"NOT ENOUGH INFO (score: {scores.get('NOT ENOUGH INFO', 0.0):.3f}). "
                f"Notice: While the retrieved passage directly addresses the claim subject and entities, "
                f"high retrieval similarity alone does not establish factual verification. The linear TF-IDF "
                f"classifier may fail to confirm the claim due to phrasing sensitivity, vocabulary differences, "
                f"or model representation limits. ContextShield transparently flags this potential misclassification "
                f"rather than asserting false certainty."
            )
        elif is_borderline and top_passage and "SUPPORTS" in scores and abs(scores.get("SUPPORTS", 0.0) - scores.get("NOT ENOUGH INFO", 0.0)) < 0.15:
            explanation_text = (
                f"ContextShield retrieved relevant background from Wikipedia article '{art_clean}' "
                f"(\"{top_passage['text'].strip()}\"), but the linear TF-IDF classifier's margin between "
                f"NOT ENOUGH INFO ({scores.get('NOT ENOUGH INFO', 0.0):.3f}) and alternative classes "
                f"is narrow ({margin:.3f}). Due to phrasing sensitivity of bag-of-words and character n-grams, "
                f"the system conservatively abstains from asserting a definitive verdict, correctly reporting "
                "NOT ENOUGH INFO rather than projecting false certainty."
            )
        else:
            explanation_text = (
                f"The local corpus contains general information regarding '{art_clean}', but the retrieved "
                f"passages do not contain verified factual details confirming or disproving this specific assertion. "
                f"Because the evidence is non-decisive (NOT ENOUGH INFO score: {scores.get('NOT ENOUGH INFO', 0.0):.3f}), "
                "ContextShield transparently abstains from making an unsupported factual determination."
            )

    return {
        "verdict": label,
        "verdict_headline": verdict_headline,
        "explanation_text": explanation_text,
        "key_passage": key_passage,
        "certainty_tier": certainty_tier,
        "certainty_description": certainty_desc,
        "has_borderline_margin": is_borderline,
        "has_relevance_warning": has_high_relevance if label == "NOT ENOUGH INFO" else False,
    }
