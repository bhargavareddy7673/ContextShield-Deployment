"""Unit and integration tests for ContextShield pipeline."""

import unittest
from pathlib import Path
import numpy as np

from pipeline import (
    DATA_DIR,
    FINAL_EVIDENCE_K,
    TOP_K,
    format_claim_evidence,
    load_pipeline,
    retrieve_evidence,
    verify_claim,
)


class TestFormatting(unittest.TestCase):
    def test_format_claim_evidence_basic(self):
        pair = format_claim_evidence("Earth is round.", "Earth is an oblate spheroid.")
        self.assertEqual(pair, "Earth is round. [SEP] Earth is an oblate spheroid.")

    def test_format_claim_evidence_whitespace_stripping(self):
        pair = format_claim_evidence("  Claim text  ", "  Evidence text  \n")
        self.assertEqual(pair, "Claim text [SEP] Evidence text")


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
        # Query with characters/words not in vocabulary
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

    def test_aggregation_probability_distribution(self):
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
        ]
        summary, results = verify_claim(claim, mock_evidence, self.verifier)
        self.assertIsNotNone(summary)
        self.assertEqual(len(results), 2)

        # Probabilities must sum to ~1.0
        scores = summary["scores"]
        total_prob = sum(scores.values())
        self.assertAlmostEqual(total_prob, 1.0, places=4)

        # Summary label must be the argmax of the scores
        best_label = max(scores.items(), key=lambda kv: kv[1])[0]
        self.assertEqual(summary["label"], best_label)
        self.assertAlmostEqual(summary["predicted_class_score"], scores[best_label], places=4)
        self.assertGreaterEqual(summary["margin"], 0.0)

        # Each result item must have per-class scores summing to ~1.0
        for res in results:
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
