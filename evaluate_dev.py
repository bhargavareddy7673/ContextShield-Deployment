"""Evaluate ContextShield verifier on saved development sets.

Computes accuracy, macro-F1, per-class metrics, and confusion matrices
without modifying or regenerating any saved data or model artifacts.
"""

import json
from pathlib import Path
import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

from pipeline import DATA_DIR, format_claim_evidence

MODEL = joblib.load(DATA_DIR / "evidence_verifier_3class.joblib")
CLASSES = list(MODEL.classes_)


def evaluate_dataset(filename: str):
    path = DATA_DIR / filename
    if not path.is_file():
        print(f"File not found: {path}")
        return

    print(f"\n=======================================================")
    print(f"Evaluating: {filename}")
    print(f"=======================================================")

    with open(path, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    y_true = [r["label"] for r in rows]
    y_pred = []

    batch_size = 500
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        pairs, lengths = [], []
        for r in batch:
            ev = r.get("evidence", [])
            claim = r["claim"]
            pairs.extend(format_claim_evidence(claim, item["text"]) for item in ev)
            lengths.append(len(ev))

        probs = MODEL.predict_proba(pairs)
        offset = 0
        for r, length in zip(batch, lengths):
            mean_prob = probs[offset:offset + length].mean(axis=0)
            pred = str(CLASSES[int(np.argmax(mean_prob))])
            y_pred.append(pred)
            offset += length

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")

    print(f"Total claims: {len(rows)}")
    print(f"Accuracy:     {acc:.6f} ({acc * 100:.2f}%)")
    print(f"Macro F1:     {macro_f1:.6f}")
    print("\nDetailed Per-Class Classification Report:")
    print(classification_report(y_true, y_pred, digits=4, target_names=CLASSES))
    print(f"Confusion Matrix (Rows=True, Columns=Predicted; order={CLASSES}):")
    print(confusion_matrix(y_true, y_pred, labels=CLASSES))


if __name__ == "__main__":
    evaluate_dataset("retrieved_dev_top10.jsonl")
    evaluate_dataset("retrieved_dev_hybrid_first_semantic_append_top20.jsonl")
