
import json
from pathlib import Path

import joblib
import numpy as np
from scipy.stats import binomtest

DATA = Path(__file__).resolve().parent / "data"
MODEL = joblib.load(DATA / "evidence_verifier_3class.joblib")


def load_rows(filename):
    with (DATA / filename).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def correctness(rows):
    correct = []

    for start in range(0, len(rows), 200):
        batch = rows[start:start + 200]
        pairs, lengths = [], []

        for row in batch:
            evidence = row.get("evidence", [])
            if not evidence:
                raise ValueError("A claim has no evidence.")

            pairs.extend(
                f"{row['claim']} [SEP] {item['text']}"
                for item in evidence
            )
            lengths.append(len(evidence))

        probabilities = MODEL.predict_proba(pairs)
        offset = 0

        for row, length in zip(batch, lengths):
            mean_prob = probabilities[offset:offset + length].mean(axis=0)
            prediction = str(MODEL.classes_[int(np.argmax(mean_prob))])
            correct.append(prediction == row["label"])
            offset += length

        print(f"Processed {min(start + len(batch), len(rows))}/{len(rows)}")

    return correct


baseline = load_rows("retrieved_dev_top10.jsonl")
hybrid = load_rows("retrieved_dev_hybrid_first_semantic_append_top20.jsonl")

assert [r["claim_id"] for r in baseline] == [
    r["claim_id"] for r in hybrid
], "Claim IDs or order differ."

assert [r["label"] for r in baseline] == [
    r["label"] for r in hybrid
], "Gold labels differ."

print("Checking baseline...")
base_correct = correctness(baseline)

print("\nChecking hybrid...")
hybrid_correct = correctness(hybrid)

# b: baseline wrong, hybrid correct
# c: baseline correct, hybrid wrong
b = sum(not x and y for x, y in zip(base_correct, hybrid_correct))
c = sum(x and not y for x, y in zip(base_correct, hybrid_correct))

p_value = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0

print("\n--- PAIRED COMPARISON ---")
print("Claims:", len(base_correct))
print("Baseline wrong, hybrid correct:", b)
print("Baseline correct, hybrid wrong:", c)
print("Net additional correct predictions:", b - c)
print(f"McNemar exact two-sided p-value: {p_value:.6g}")
print("This is a development-set comparison.")
