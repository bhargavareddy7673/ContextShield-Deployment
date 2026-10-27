"""Evaluate ContextShield verifier on saved development sets across grouping strategies.

Compares:
1. Old Baseline: [SEP] per-sentence mean pooling
2. Strategy 1: Training format per-sentence mean pooling (claim: ... evidence: ...)
3. Strategy 2: Training format article-grouped mean pooling (claim: ... evidence: ...)
4. Strategy 3: Training format all-retrieved joined (single concatenated input)

Evaluates on identical examples without modifying or regenerating any saved data or model artifacts.
"""

import json
from collections import defaultdict
from pathlib import Path
import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

from pipeline import DATA_DIR, format_claim_evidence, group_evidence_by_article

MODEL = joblib.load(DATA_DIR / "evidence_verifier_3class.joblib")
CLASSES = list(MODEL.classes_)


def evaluate_dataset(filename: str, batch_size: int = 5000):
    path = DATA_DIR / filename
    if not path.is_file():
        print(f"File not found: {path}")
        return

    print(f"\n" + "=" * 80)
    print(f"EVALUATING DATASET: {filename}")
    print(f"=" * 80)

    with open(path, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    y_true = [r["label"] for r in rows]
    print(f"Total claims: {len(rows)} (identical examples for all strategies)")

    def predict_batches(texts):
        probs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            probs.append(MODEL.predict_proba(batch))
        return np.vstack(probs) if probs else np.empty((0, len(CLASSES)))

    # 1. Old Baseline: [SEP] per-sentence mean
    print("Evaluating Baseline: [SEP] per-sentence mean...")
    sep_pairs, sep_lengths = [], []
    for r in rows:
        c = r["claim"].strip()
        ev = r.get("evidence", [])
        sep_pairs.extend(f"{c} [SEP] {item['text'].strip()}" for item in ev)
        sep_lengths.append(len(ev))
    sep_probs = predict_batches(sep_pairs)
    preds_sep = []
    offset = 0
    for length in sep_lengths:
        mean_p = sep_probs[offset:offset + length].mean(axis=0)
        preds_sep.append(CLASSES[int(np.argmax(mean_p))])
        offset += length

    # 2. Strategy 1: Training format per-sentence mean
    print("Evaluating Strategy 1: Training format per-sentence mean...")
    sent_pairs, sent_lengths = [], []
    for r in rows:
        c = r["claim"].strip()
        ev = r.get("evidence", [])
        sent_pairs.extend(format_claim_evidence(c, item["text"]) for item in ev)
        sent_lengths.append(len(ev))
    sent_probs = predict_batches(sent_pairs)
    preds_sent = []
    offset = 0
    for length in sent_lengths:
        mean_p = sent_probs[offset:offset + length].mean(axis=0)
        preds_sent.append(CLASSES[int(np.argmax(mean_p))])
        offset += length

    # 3. Strategy 2: Training format article-grouped mean
    print("Evaluating Strategy 2: Training format article-grouped mean...")
    art_pairs, art_lengths = [], []
    for r in rows:
        c = r["claim"].strip()
        ev = r.get("evidence", [])
        groups = group_evidence_by_article(ev)
        art_pairs.extend(format_claim_evidence(c, g["text"]) for g in groups)
        art_lengths.append(len(groups))
    art_probs = predict_batches(art_pairs)
    preds_art = []
    offset = 0
    for length in art_lengths:
        mean_p = art_probs[offset:offset + length].mean(axis=0)
        preds_art.append(CLASSES[int(np.argmax(mean_p))])
        offset += length

    # 4. Strategy 3: Training format all-retrieved joined
    print("Evaluating Strategy 3: Training format all-retrieved joined...")
    all_pairs = [format_claim_evidence(r["claim"], r.get("evidence_text", "")) for r in rows]
    all_probs = predict_batches(all_pairs)
    preds_all = [CLASSES[int(np.argmax(p))] for p in all_probs]

    strategies = [
        ("Old Baseline: [SEP] per-sentence mean", preds_sep),
        ("Strategy 1: Training format per-sentence mean", preds_sent),
        ("Strategy 2: Training format article-grouped mean", preds_art),
        ("Strategy 3: Training format all-retrieved joined", preds_all),
    ]

    print("\n" + "-" * 75)
    print(f"{'Strategy':<50} | {'Accuracy':<10} | {'Macro F1':<10}")
    print("-" * 75)
    for name, p in strategies:
        acc = accuracy_score(y_true, p)
        f1 = f1_score(y_true, p, average="macro")
        print(f"{name:<50} | {acc:<10.4f} | {f1:<10.4f}")

    print("\n" + "=" * 80)
    print("DETAILED PER-CLASS METRICS & CONFUSION MATRICES")
    print("=" * 80)
    for name, p in strategies:
        print(f"\n--- {name} ---")
        acc = accuracy_score(y_true, p)
        f1 = f1_score(y_true, p, average="macro")
        print(f"Accuracy: {acc:.6f} ({acc * 100:.2f}%) | Macro F1: {f1:.6f}")
        print(classification_report(y_true, p, target_names=CLASSES, digits=4))
        print(f"Confusion Matrix (Order: {CLASSES}):")
        print(confusion_matrix(y_true, p, labels=CLASSES))


if __name__ == "__main__":
    evaluate_dataset("retrieved_dev_top10.jsonl")
    evaluate_dataset("retrieved_dev_hybrid_first_semantic_append_top20.jsonl")
