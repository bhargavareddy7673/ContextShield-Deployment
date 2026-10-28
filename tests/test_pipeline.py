"""Unit and integration tests for ContextShield pipeline."""

import unittest
from pathlib import Path
import numpy as np

from pipeline import (
    DATA_DIR,
    FINAL_EVIDENCE_K,
    TOP_K,
    format_claim_evidence,
    generate_explanation,
    group_evidence_by_article,
    load_pipeline,
    retrieve_evidence,
    verify_claim,
)


class TestFormatting(unittest.TestCase):
    def test_format_claim_evidence_basic(self):
        pair = format_claim_evidence("Earth is round.", "Earth is an oblate spheroid.")
        self.assertEqual(pair, "claim: Earth is round. evidence: Earth is an oblate spheroid.")

    def test_format_claim_evidence_whitespace_stripping(self):
        pair = format_claim_evidence("  Claim text  ", "  Evidence text  \n")
        self.assertEqual(pair, "claim: Claim text evidence: Evidence text")


class TestGrouping(unittest.TestCase):
    def test_group_evidence_by_article(self):
        evidence = [
            {"rank": 1, "article_id": "Team_A", "sentence_id": 0, "text": "Sentence zero ."},
            {"rank": 2, "article_id": "Team_B", "sentence_id": 0, "text": "Team B info ."},
            {"rank": 3, "article_id": "Team_A", "sentence_id": 2, "text": "Sentence two ."},
        ]
        groups = group_evidence_by_article(evidence)
        self.assertEqual(len(groups), 2)
        # First group should be Team_A because min_rank is 1
        self.assertEqual(groups[0]["article_id"], "Team_A")
        self.assertEqual(groups[0]["sentence_count"], 2)
        self.assertEqual(groups[0]["text"], "Sentence zero . Sentence two .")
        self.assertEqual(groups[0]["sentence_ids"], [0, 2])
        self.assertEqual(groups[0]["min_rank"], 1)
        self.assertEqual(groups[0]["rank"], 1)
        # Second group is Team_B
        self.assertEqual(groups[1]["article_id"], "Team_B")
        self.assertEqual(groups[1]["sentence_count"], 1)
        self.assertEqual(groups[1]["text"], "Team B info .")
        self.assertEqual(groups[1]["min_rank"], 2)
        self.assertEqual(groups[1]["rank"], 2)


class TestPipelineArtifacts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (
            cls.verifier,
            cls.vectorizer,
            cls.matrix,
            cls.corpus,
            cls.embeddings,
            cls.semantic_model,
        ) = load_pipeline(DATA_DIR)

    def test_pipeline_alignment(self):
        n_corpus = len(self.corpus)
        self.assertEqual(n_corpus, 108559)
        self.assertEqual(self.matrix.shape[0], n_corpus)
        self.assertEqual(self.embeddings.shape[0], n_corpus)
        self.assertEqual(self.matrix.shape[1], len(self.vectorizer.vocabulary_))
        self.assertEqual(self.embeddings.shape[1], 384)

    def test_verifier_classes(self):
        expected_classes = ["NOT ENOUGH INFO", "REFUTES", "SUPPORTS"]
        self.assertEqual(list(self.verifier.classes_), expected_classes)


class TestRetrieval(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (
            cls.verifier,
            cls.vectorizer,
            cls.matrix,
            cls.corpus,
            cls.embeddings,
            cls.semantic_model,
        ) = load_pipeline(DATA_DIR)

    def test_empty_and_whitespace_claim(self):
        self.assertEqual(
            retrieve_evidence(
                "",
                self.vectorizer,
                self.matrix,
                self.corpus,
                self.embeddings,
                self.semantic_model,
            ),
            [],
        )
        self.assertEqual(
            retrieve_evidence(
                "    \n\t  ",
                self.vectorizer,
                self.matrix,
                self.corpus,
                self.embeddings,
                self.semantic_model,
            ),
            [],
        )

    def test_oov_claim_graceful_handling(self):
        results = retrieve_evidence(
            "xyzqvr12345 nonesuchwordexists",
            self.vectorizer,
            self.matrix,
            self.corpus,
            self.embeddings,
            self.semantic_model,
            top_k=5,
            final_k=10,
        )
        self.assertIsInstance(results, list)
        self.assertLessEqual(len(results), 10)

    def test_retrieval_structure_and_deduplication(self):
        claim = "Fox 2000 Pictures released the film Soul Food."
        results = retrieve_evidence(
            claim,
            self.vectorizer,
            self.matrix,
            self.corpus,
            self.embeddings,
            self.semantic_model,
            top_k=TOP_K,
            final_k=FINAL_EVIDENCE_K,
        )
        self.assertEqual(len(results), FINAL_EVIDENCE_K)

        keys_seen = set()
        for idx, item in enumerate(results, start=1):
            self.assertEqual(item["rank"], idx)
            self.assertIn("article_id", item)
            self.assertIn("sentence_id", item)
            self.assertIn("text", item)
            self.assertIn("retrieval_score", item)
            self.assertIn("retrieval_score_type", item)
            self.assertIn("retrieved_by", item)
            self.assertIn("tfidf_score", item)
            self.assertIn("semantic_score", item)
            self.assertIn(
                item["retrieval_score_type"],
                ["reciprocal_rank_fusion", "cosine_similarity"],
            )

            key = (item["article_id"], item["sentence_id"])
            self.assertNotIn(key, keys_seen, f"Duplicate evidence item found: {key}")
            keys_seen.add(key)


class TestVerification(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (
            cls.verifier,
            cls.vectorizer,
            cls.matrix,
            cls.corpus,
            cls.embeddings,
            cls.semantic_model,
        ) = load_pipeline(DATA_DIR)

    def test_empty_evidence_or_claim(self):
        summary, results = verify_claim("Any claim", [], self.verifier)
        self.assertIsNone(summary)
        self.assertEqual(results, [])

        summary, results = verify_claim("", [{"text": "Some evidence"}], self.verifier)
        self.assertIsNone(summary)
        self.assertEqual(results, [])

    def test_aggregation_and_strategies(self):
        claim = "Fox 2000 Pictures released the film Soul Food."
        mock_evidence = [
            {
                "rank": 1,
                "article_id": "Soul_Food",
                "sentence_id": 0,
                "text": "Fox 2000 Pictures released the film on July 23 , 2010 .",
            },
            {
                "rank": 2,
                "article_id": "Soul_Food",
                "sentence_id": 1,
                "text": "Universal Pictures released the film in cinemas in the United States.",
            },
            {
                "rank": 3,
                "article_id": "Other_Article",
                "sentence_id": 0,
                "text": "Fox 2000 Pictures was a sister division.",
            },
        ]

        # Test article-grouped relevance-weighted (default)
        summary_art_rw, results_art_rw = verify_claim(
            claim, mock_evidence, self.verifier, grouping_strategy="article_grouped", aggregation_method="relevance_weighted"
        )
        self.assertIsNotNone(summary_art_rw)
        self.assertEqual(summary_art_rw["grouping_strategy"], "article_grouped")
        self.assertEqual(summary_art_rw["aggregation_method"], "relevance_weighted")
        self.assertEqual(summary_art_rw["evidence_sets_count"], 2)
        self.assertAlmostEqual(sum(summary_art_rw["scores"].values()), 1.0, places=4)
        # Check weights in article_groups
        art_groups = summary_art_rw["article_groups"]
        self.assertEqual(len(art_groups), 2)
        self.assertGreater(art_groups[0]["weight"], art_groups[1]["weight"])
        self.assertAlmostEqual(art_groups[0]["weight"] + art_groups[1]["weight"], 1.0, places=4)

        # Test article-grouped mean (baseline)
        summary_art_m, _ = verify_claim(
            claim, mock_evidence, self.verifier, grouping_strategy="article_grouped", aggregation_method="mean"
        )
        self.assertIsNotNone(summary_art_m)
        self.assertEqual(summary_art_m["aggregation_method"], "mean")
        self.assertAlmostEqual(sum(summary_art_m["scores"].values()), 1.0, places=4)
        self.assertAlmostEqual(
            summary_art_m["article_groups"][0]["weight"],
            summary_art_m["article_groups"][1]["weight"],
            places=4,
        )

        # Test per-sentence relevance-weighted
        summary_sent_rw, results_sent_rw = verify_claim(
            claim, mock_evidence, self.verifier, grouping_strategy="per_sentence", aggregation_method="relevance_weighted"
        )
        self.assertIsNotNone(summary_sent_rw)
        self.assertEqual(summary_sent_rw["grouping_strategy"], "per_sentence")
        self.assertEqual(summary_sent_rw["aggregation_method"], "relevance_weighted")
        self.assertEqual(summary_sent_rw["evidence_sets_count"], 3)
        self.assertAlmostEqual(sum(summary_sent_rw["scores"].values()), 1.0, places=4)
        self.assertGreater(results_sent_rw[0]["weight"], results_sent_rw[1]["weight"])
        self.assertGreater(results_sent_rw[1]["weight"], results_sent_rw[2]["weight"])
        self.assertAlmostEqual(sum(r["weight"] for r in results_sent_rw), 1.0, places=4)

        # Test per-sentence mean
        summary_sent_m, _ = verify_claim(
            claim, mock_evidence, self.verifier, grouping_strategy="per_sentence", aggregation_method="mean"
        )
        self.assertIsNotNone(summary_sent_m)
        self.assertEqual(summary_sent_m["aggregation_method"], "mean")
        self.assertAlmostEqual(sum(summary_sent_m["scores"].values()), 1.0, places=4)

    def test_single_evidence_equivalence(self):
        claim = "Fox 2000 Pictures released the film Soul Food."
        single_evidence = [
            {
                "rank": 1,
                "article_id": "Soul_Food",
                "sentence_id": 0,
                "text": "Fox 2000 Pictures released the film on July 23 , 2010 .",
            }
        ]
        s_rw, _ = verify_claim(claim, single_evidence, self.verifier, aggregation_method="relevance_weighted")
        s_mean, _ = verify_claim(claim, single_evidence, self.verifier, aggregation_method="mean")
        self.assertEqual(s_rw["label"], s_mean["label"])
        self.assertAlmostEqual(s_rw["predicted_class_score"], s_mean["predicted_class_score"], places=6)
        for cls in ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]:
            self.assertAlmostEqual(s_rw["scores"][cls], s_mean["scores"][cls], places=6)

    def test_relevance_weighting_prevents_dilution(self):
        """Verify that relevance weighting preserves decisive top-ranked evidence over low-ranked distractors."""
        claim = "Fox 2000 Pictures released the film Soul Food."
        decisive_item = {
            "rank": 1,
            "article_id": "Soul_Food",
            "sentence_id": 0,
            "text": "Fox 2000 Pictures released the film on July 23 , 2010 .",
        }
        # Add 9 uninformative distractors at lower ranks
        distractors = [
            {
                "rank": i,
                "article_id": f"Distractor_{i}",
                "sentence_id": 0,
                "text": f"This is an uninformative background sentence {i} discussing general film production history.",
            }
            for i in range(2, 11)
        ]
        evidence = [decisive_item] + distractors

        s_rw, _ = verify_claim(claim, evidence, self.verifier, aggregation_method="relevance_weighted")
        s_mean, _ = verify_claim(claim, evidence, self.verifier, aggregation_method="mean")

        # In relevance_weighted, Rank 1 has weight 1.0 / (1 + 1/2 + 1/3 + ... + 1/10) ~ 0.34
        # In uniform mean, Rank 1 has weight 1/10 = 0.10
        # Consequently, SUPPORTS score must be substantially higher under relevance_weighted
        self.assertGreater(s_rw["scores"]["SUPPORTS"], s_mean["scores"]["SUPPORTS"])

    def test_representative_supports_case(self):
        claim = "Fox 2000 Pictures released the film Soul Food."
        evidence = [
            {
                "rank": 1,
                "article_id": "Soul_Food",
                "sentence_id": 0,
                "text": "Fox 2000 Pictures released the film on July 23 , 2010 .",
            }
        ]
        summary, _ = verify_claim(claim, evidence, self.verifier)
        self.assertEqual(summary["label"], "SUPPORTS")

    def test_representative_refutes_case(self):
        claim = "Savages was exclusively a German film."
        evidence = [
            {
                "rank": 1,
                "article_id": "Savages",
                "sentence_id": 0,
                "text": "For the German film director , see Michael Hofmann -LRB- director -RRB- .",
            }
        ]
        summary, _ = verify_claim(claim, evidence, self.verifier)
        self.assertEqual(summary["label"], "REFUTES")

    def test_representative_not_enough_info_case(self):
        claim = "Tilda Swinton is a vegan."
        evidence = [
            {
                "rank": 1,
                "article_id": "Tilda_Swinton",
                "sentence_id": 0,
                "text": "The film stars George Clooney , John Malkovich , Frances McDormand , Tilda Swinton , and Brad Pitt .",
            }
        ]
        summary, _ = verify_claim(claim, evidence, self.verifier)
        self.assertEqual(summary["label"], "NOT ENOUGH INFO")

    def test_carolina_panthers_case(self):
        claim = "The Carolina Panthers' inaugural NFL season was in 1995."
        evidence = retrieve_evidence(
            claim,
            self.vectorizer,
            self.matrix,
            self.corpus,
            self.embeddings,
            self.semantic_model,
            top_k=TOP_K,
            final_k=FINAL_EVIDENCE_K,
        )
        self.assertEqual(len(evidence), FINAL_EVIDENCE_K)
        # Verify rank 1 retrieved passage
        self.assertEqual(evidence[0]["article_id"], "1995_Carolina_Panthers_season")
        self.assertEqual(int(evidence[0]["sentence_id"]), 0)
        self.assertIn("inaugural season", evidence[0]["text"])

        # Test both relevance_weighted and mean aggregation
        summary_rw, results_rw = verify_claim(
            claim, evidence, self.verifier, aggregation_method="relevance_weighted"
        )
        summary_m, results_m = verify_claim(
            claim, evidence, self.verifier, aggregation_method="mean"
        )

        self.assertIsNotNone(summary_rw)
        self.assertIsNotNone(summary_m)
        self.assertEqual(summary_rw["evidence_count"], FINAL_EVIDENCE_K)
        self.assertAlmostEqual(sum(summary_rw["scores"].values()), 1.0, places=4)
        self.assertAlmostEqual(sum(summary_m["scores"].values()), 1.0, places=4)
        self.assertEqual(len(results_rw), FINAL_EVIDENCE_K)

        # Under relevance_weighted, the spurious distant REFUTES signal (e.g. Rank 14 = 0.964)
        # is significantly attenuated compared to unweighted mean
        self.assertLess(summary_rw["scores"]["REFUTES"], summary_m["scores"]["REFUTES"])

        # Test explanation and mismatch warning for Carolina Panthers
        exp = generate_explanation(claim, summary_rw, results_rw)
        self.assertEqual(exp["verdict"], "NOT ENOUGH INFO")
        self.assertTrue(exp["has_relevance_warning"])
        self.assertEqual(exp["verdict_headline"], "Evidence Retrieved but Unconfirmed")
        self.assertIn("highly relevant evidence passage", exp["explanation_text"])
        self.assertIn("1995 Carolina Panthers season", exp["explanation_text"])
        # Ensure generic "no factual details" message is NOT shown
        self.assertNotIn("do not contain verified factual details", exp["explanation_text"])
        # Ensure high similarity does NOT claim the claim is proven true
        self.assertIn("high retrieval similarity alone does not establish factual verification", exp["explanation_text"])

    def test_ordinary_not_enough_info_explanation(self):
        """Test ordinary NOT ENOUGH INFO case where corpus genuinely lacks factual details."""
        claim = "Tilda Swinton is a vegan."
        evidence = retrieve_evidence(
            claim,
            self.vectorizer,
            self.matrix,
            self.corpus,
            self.embeddings,
            self.semantic_model,
            top_k=TOP_K,
            final_k=FINAL_EVIDENCE_K,
        )
        summary, results = verify_claim(claim, evidence, self.verifier)
        self.assertEqual(summary["label"], "NOT ENOUGH INFO")

        exp = generate_explanation(claim, summary, results)
        self.assertEqual(exp["verdict"], "NOT ENOUGH INFO")
        self.assertFalse(exp["has_relevance_warning"])
        self.assertEqual(exp["verdict_headline"], "Insufficient Evidence in Corpus")
        self.assertIn("do not contain verified factual details", exp["explanation_text"])

    def test_generate_explanation_grounding(self):
        # 1. Empty input handling
        exp_empty = generate_explanation("", None, [])
        self.assertEqual(exp_empty["verdict"], "NOT ENOUGH INFO")
        self.assertEqual(exp_empty["verdict_headline"], "No Evidence Retrieved")

        # 2. SUPPORTS explanation grounding
        claim_sup = "Fox 2000 Pictures released the film Soul Food."
        mock_ev_sup = [
            {
                "rank": 1,
                "article_id": "Soul_Food",
                "sentence_id": 0,
                "text": "Fox 2000 Pictures released the film on July 23 , 2010 .",
                "retrieved_by": ["tfidf"],
                "retrieval_score": 0.032,
                "retrieval_score_type": "reciprocal_rank_fusion",
            }
        ]
        sum_sup, res_sup = verify_claim(claim_sup, mock_ev_sup, self.verifier)
        exp_sup = generate_explanation(claim_sup, sum_sup, res_sup)
        self.assertEqual(exp_sup["verdict"], "SUPPORTS")
        self.assertIn("Evidence Confirms", exp_sup["verdict_headline"])
        self.assertIn("Soul Food", exp_sup["explanation_text"])
        self.assertIn("Fox 2000 Pictures", exp_sup["explanation_text"])
        self.assertIsNotNone(exp_sup["key_passage"])

        # 3. REFUTES explanation grounding
        claim_ref = "Savages was exclusively a German film."
        mock_ev_ref = [
            {
                "rank": 1,
                "article_id": "Savages",
                "sentence_id": 0,
                "text": "For the German film director , see Michael Hofmann -LRB- director -RRB- .",
                "retrieved_by": ["semantic"],
                "retrieval_score": 0.032,
                "retrieval_score_type": "reciprocal_rank_fusion",
            }
        ]
        sum_ref, res_ref = verify_claim(claim_ref, mock_ev_ref, self.verifier)
        exp_ref = generate_explanation(claim_ref, sum_ref, res_ref)
        self.assertEqual(exp_ref["verdict"], "REFUTES")
        self.assertIn("Evidence Contradicts", exp_ref["verdict_headline"])
        self.assertIn("Savages", exp_ref["explanation_text"])

        # 4. NOT ENOUGH INFO explanation
        claim_nei = "Tilda Swinton is a vegan."
        mock_ev_nei = [
            {
                "rank": 1,
                "article_id": "Tilda_Swinton",
                "sentence_id": 0,
                "text": "The film stars George Clooney , John Malkovich , Frances McDormand , Tilda Swinton , and Brad Pitt .",
                "retrieved_by": ["tfidf"],
                "retrieval_score": 0.032,
                "retrieval_score_type": "reciprocal_rank_fusion",
            }
        ]
        sum_nei, res_nei = verify_claim(claim_nei, mock_ev_nei, self.verifier)
        exp_nei = generate_explanation(claim_nei, sum_nei, res_nei)
        self.assertEqual(exp_nei["verdict"], "NOT ENOUGH INFO")
        self.assertIn("Insufficient Evidence", exp_nei["verdict_headline"])

    def test_aggregation_distribution_and_group_consistency(self):
        """Regression test verifying mathematical aggregation consistency on 'The Earth orbits the Sun.'"""
        claim = "The Earth orbits the Sun."
        evidence = retrieve_evidence(
            claim,
            self.vectorizer,
            self.matrix,
            self.corpus,
            self.embeddings,
            self.semantic_model,
            top_k=TOP_K,
            final_k=FINAL_EVIDENCE_K,
        )
        self.assertEqual(len(evidence), FINAL_EVIDENCE_K)

        summary, results = verify_claim(
            claim,
            evidence,
            self.verifier,
            grouping_strategy="article_grouped",
            aggregation_method="relevance_weighted",
        )
        self.assertIsNotNone(summary)
        self.assertEqual(summary["label"], "SUPPORTS")

        classes = list(self.verifier.classes_)

        # 1. Verify mathematical probability pooling across article groups: sum(w_k * P_k) == summary["scores"]
        article_groups = summary["article_groups"]
        self.assertGreater(len(article_groups), 1)

        total_weight = sum(g["weight"] for g in article_groups)
        self.assertAlmostEqual(total_weight, 1.0, places=5)

        manual_pooled = {c: 0.0 for c in classes}
        for g in article_groups:
            w = g["weight"]
            for c in classes:
                manual_pooled[c] += w * g["class_scores"][c]

        for c in classes:
            self.assertAlmostEqual(summary["scores"][c], manual_pooled[c], places=5)

        # 2. Verify summary label and predicted class score match argmax
        best_class = max(summary["scores"], key=summary["scores"].get)
        self.assertEqual(summary["label"], best_class)
        self.assertAlmostEqual(summary["predicted_class_score"], summary["scores"][best_class], places=5)
        self.assertAlmostEqual(sum(summary["scores"].values()), 1.0, places=5)

        # 3. Verify Article Set 01 (Earth's_orbit) genuine classifier output and constituent sentences
        set_1 = article_groups[0]
        self.assertEqual(set_1["article_id"], "Earth's_orbit")
        self.assertEqual(set_1["predicted_label"], "NOT ENOUGH INFO")
        self.assertAlmostEqual(set_1["model_score"], 0.500, places=2)
        self.assertEqual(len(set_1["constituent_sentences"]), set_1["sentence_count"])

        # Check that constituent sentence 0 predicts SUPPORTS (> 0.70)
        sent_0 = [s for s in set_1["constituent_sentences"] if str(s["sentence_id"]) == "0"][0]
        self.assertEqual(sent_0["predicted_label"], "SUPPORTS")
        self.assertGreater(sent_0["model_score"], 0.70)

        # 4. Verify grounded explanation transparently notes group divergence
        explanation = generate_explanation(claim, summary, results)
        self.assertEqual(explanation["verdict"], "SUPPORTS")
        self.assertIn("Earth's orbit", explanation["explanation_text"])
        self.assertIn("NOT ENOUGH INFO", explanation["explanation_text"])
        self.assertIn("dilution", explanation["explanation_text"])


if __name__ == "__main__":
    unittest.main()
