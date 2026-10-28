"""Evaluate ContextShield verifier on saved development sets across grouping and aggregation strategies.

Compares:
1. Old Baseline: [SEP] per-sentence mean pooling
2. Strategy 1: Training format per-sentence mean pooling (claim: ... evidence: ...)
3. Strategy 2: Training format article-grouped mean pooling (claim: ... evidence: ...)
4. Strategy 3: Training format all-retrieved joined (single concatenated input)
5. Strategy 4: Training format article-grouped relevance-weighted (reciprocal rank 1/r)
6. Strategy 5: Training format per-sentence relevance-weighted (reciprocal rank 1/r)

Evaluates on identical examples without modifying or regenerating any saved data or model artifacts.
"""

import json
from pathlib import Path
import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

from pipeline import BASE_DIR, DATA_DIR, format_claim_evidence, group_evidence_by_article

MODEL = joblib.load(DATA_DIR / "evidence_verifier_3class.joblib")
CLASSES = list(MODEL.classes_)


def evaluate_dataset(filename: str, batch_size: int = 5000):
    path = DATA_DIR / filename
    if not path.is_file():
        print(f"File not found: {path}")
        return

    report_file = BASE_DIR / "dev_eval_results.txt"
    output_lines = []
    def log(msg=""):
        print(msg, flush=True)
        output_lines.append(str(msg))
        with open(report_file, "a", encoding="utf-8") as f:
            f.write(str(msg) + "\n")

    log("\n" + "=" * 80)
    log(f"EVALUATING DATASET: {filename}")
    log("=" * 80)

    with open(path, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    y_true = [r["label"] for r in rows]
    log(f"Total claims: {len(rows)} (identical examples for all strategies)")

    def predict_batches(texts):
        probs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            probs.append(MODEL.predict_proba(batch))
        return np.vstack(probs) if probs else np.empty((0, len(CLASSES)))

    # 1. Old Baseline: [SEP] per-sentence mean
    log("Evaluating Baseline: [SEP] per-sentence mean...")
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

    # 2. Strategy 1: Training format per-sentence mean (Baseline Uniform)
    log("Evaluating Strategy 1: Training format per-sentence mean...")
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

    # 3. Strategy 2: Training format article-grouped mean (Baseline Uniform)
    log("Evaluating Strategy 2: Training format article-grouped mean...")
    art_pairs, art_lengths, all_groups = [], [], []
    for r in rows:
        c = r["claim"].strip()
        ev = r.get("evidence", [])
        groups = group_evidence_by_article(ev)
        all_groups.append(groups)
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
    log("Evaluating Strategy 3: Training format all-retrieved joined...")
    all_pairs = [format_claim_evidence(r["claim"], r.get("evidence_text", "")) for r in rows]
    all_probs = predict_batches(all_pairs)
    preds_all = [CLASSES[int(np.argmax(p))] for p in all_probs]

    # 5. Strategy 4: Training format article-grouped relevance-weighted (reciprocal rank 1/r)
    log("Evaluating Strategy 4: Training format article-grouped relevance-weighted (1/r)...")
    preds_art_weighted = []
    offset = 0
    for length, groups in zip(art_lengths, all_groups):
        if length == 0:
            preds_art_weighted.append("NOT ENOUGH INFO")
            continue
        probs = art_probs[offset:offset + length]
        weights = np.array(
            [1.0 / max(1, g.get("min_rank", g.get("rank", idx))) for idx, g in enumerate(groups, start=1)],
            dtype=float,
        )
        w_sum = weights.sum()
        weighted_p = (probs * weights[:, None]).sum(axis=0) / (w_sum if w_sum > 0 else 1.0)
        preds_art_weighted.append(CLASSES[int(np.argmax(weighted_p))])
        offset += length

    # 6. Strategy 5: Training format per-sentence relevance-weighted (reciprocal rank 1/r)
    log("Evaluating Strategy 5: Training format per-sentence relevance-weighted (1/r)...")
    preds_sent_weighted = []
    offset = 0
    for length in sent_lengths:
        if length == 0:
            preds_sent_weighted.append("NOT ENOUGH INFO")
            continue
        probs = sent_probs[offset:offset + length]
        ranks = np.arange(1, length + 1, dtype=float)
        weights = 1.0 / ranks
        w_sum = weights.sum()
        weighted_p = (probs * weights[:, None]).sum(axis=0) / (w_sum if w_sum > 0 else 1.0)
        preds_sent_weighted.append(CLASSES[int(np.argmax(weighted_p))])
        offset += length

    strategies = [
        ("Old Baseline: [SEP] per-sentence mean", preds_sep),
        ("Strategy 1: Training format per-sentence mean", preds_sent),
        ("Strategy 2: Training format article-grouped mean", preds_art),
        ("Strategy 3: Training format all-retrieved joined", preds_all),
        ("Strategy 4: Article-grouped relevance-weighted (1/r)", preds_art_weighted),
        ("Strategy 5: Per-sentence relevance-weighted (1/r)", preds_sent_weighted),
    ]

    log("\n" + "-" * 75)
    log(f"{'Strategy':<50} | {'Accuracy':<10} | {'Macro F1':<10}")
    log("-" * 75)
    for name, p in strategies:
        acc = accuracy_score(y_true, p)
        f1 = f1_score(y_true, p, average="macro")
        log(f"{name:<50} | {acc:<10.4f} | {f1:<10.4f}")

    log("\n" + "=" * 80)
    log("DETAILED PER-CLASS METRICS & CONFUSION MATRICES")
    log("=" * 80)
    for name, p in strategies:
        log(f"\n--- {name} ---")
        acc = accuracy_score(y_true, p)
        f1 = f1_score(y_true, p, average="macro")
        log(f"Accuracy: {acc:.6f} ({acc * 100:.2f}%) | Macro F1: {f1:.6f}")
        log(classification_report(y_true, p, target_names=CLASSES, digits=4))
        log(f"Confusion Matrix (Order: {CLASSES}):")
        log(str(confusion_matrix(y_true, p, labels=CLASSES)))

    # Persist log to scratch directory for persistent record
    scratch_out = Path("/Users/bhargavareddykandukuri/.gemini/antigravity/brain/4029b289-6b3e-49d2-aa61-fd3b7c697eb4/scratch") / f"eval_{filename}.txt"
    try:
        with open(scratch_out, "w", encoding="utf-8") as out_f:
            out_f.write("\n".join(output_lines))
    except Exception:
        pass


if __name__ == "__main__":
    report_file = BASE_DIR / "dev_eval_results.txt"
    if report_file.exists():
        report_file.unlink()
    evaluate_dataset("retrieved_dev_top10.jsonl")
    evaluate_dataset("retrieved_dev_hybrid_first_semantic_append_top20.jsonl")
