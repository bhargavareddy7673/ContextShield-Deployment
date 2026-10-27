"""Unit and integration tests for ContextShield pipeline."""

import unittest
from pathlib import Path
import numpy as np

from pipeline import (
    DATA_DIR,
    FINAL_EVIDENCE_K,
    TOP_K,
    format_claim_evidence,
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
        # Second group is Team_B
        self.assertEqual(groups[1]["article_id"], "Team_B")
        self.assertEqual(groups[1]["sentence_count"], 1)
        self.assertEqual(groups[1]["text"], "Team B info .")


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
        # Test article-grouped strategy
        summary_art, results_art = verify_claim(
            claim, mock_evidence, self.verifier, grouping_strategy="article_grouped"
        )
        self.assertIsNotNone(summary_art)
        self.assertEqual(summary_art["grouping_strategy"], "article_grouped")
        self.assertEqual(summary_art["evidence_sets_count"], 2)  # Soul_Food and Other_Article
        self.assertAlmostEqual(sum(summary_art["scores"].values()), 1.0, places=4)
        self.assertEqual(len(results_art), 3)

        # Test per-sentence strategy
        summary_sent, results_sent = verify_claim(
            claim, mock_evidence, self.verifier, grouping_strategy="per_sentence"
        )
        self.assertIsNotNone(summary_sent)
        self.assertEqual(summary_sent["grouping_strategy"], "per_sentence")
        self.assertEqual(summary_sent["evidence_sets_count"], 3)
        self.assertAlmostEqual(sum(summary_sent["scores"].values()), 1.0, places=4)

        # Sentence-level results must have per-class scores summing to ~1.0
        for res in results_art:
            item_scores = res["class_scores"]
            self.assertAlmostEqual(sum(item_scores.values()), 1.0, places=4)
            best_item_label = max(item_scores.items(), key=lambda kv: kv[1])[0]
            self.assertEqual(res["predicted_label"], best_item_label)
            self.assertAlmostEqual(res["model_score"], item_scores[best_item_label], places=4)

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

        summary, results = verify_claim(claim, evidence, self.verifier)
        self.assertIsNotNone(summary)
        self.assertEqual(summary["evidence_count"], FINAL_EVIDENCE_K)
        self.assertIn(summary["label"], ["NOT ENOUGH INFO", "REFUTES", "SUPPORTS"])
        self.assertAlmostEqual(sum(summary["scores"].values()), 1.0, places=4)
        self.assertEqual(len(results), FINAL_EVIDENCE_K)


if __name__ == "__main__":
    unittest.main()
