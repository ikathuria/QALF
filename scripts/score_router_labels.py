"""
Scores data/results/router_labeling_sheet.csv once gold_* columns have been
filled in by a human annotator: reports per-dimension accuracy for the 4D
complexity classifier and the intent classifier, plus (if a second
annotator column is present) inter-annotator agreement via Cohen's kappa.

Expects columns: pred_linguistic, pred_semantic, pred_modality,
pred_contextual, pred_intent, gold_linguistic, gold_semantic, gold_modality,
gold_contextual, gold_intent (empty gold_* rows are skipped, so partial
labeling is fine -- just run this on whatever's been filled in so far).

For inter-annotator agreement, add a second sheet with the same query_id
order and a `gold_*_annotator2` set of columns, then pass --second_sheet.

Usage:
    python scripts/score_router_labels.py --sheet data/results/router_labeling_sheet.csv
"""

import argparse
import csv

DIMENSIONS = ["linguistic", "semantic", "modality", "contextual"]


def load_rows(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def score(rows):
    labeled = [r for r in rows if r.get("gold_intent", "").strip()]
    if not labeled:
        print("No labeled rows found (gold_intent is empty for all rows) -- nothing to score yet.")
        return

    print(f"Scoring {len(labeled)} / {len(rows)} labeled rows.\n")

    for dim in DIMENSIONS:
        pred_col, gold_col = f"pred_{dim}", f"gold_{dim}"
        pairs = [
            (r[pred_col].strip(), r[gold_col].strip())
            for r in labeled
            if r.get(gold_col, "").strip()
        ]
        if not pairs:
            continue
        correct = sum(1 for p, g in pairs if p.lower() == g.lower())
        print(f"  {dim:12s} accuracy: {correct}/{len(pairs)} = {correct/len(pairs):.3f}")

    intent_pairs = [
        (r["pred_intent"].strip(), r["gold_intent"].strip())
        for r in labeled
        if r.get("gold_intent", "").strip()
    ]
    if intent_pairs:
        correct = sum(1 for p, g in intent_pairs if p.lower() == g.lower())
        print(f"\n  {'intent':12s} accuracy: {correct}/{len(intent_pairs)} = {correct/len(intent_pairs):.3f}")

        # Per-class breakdown, since 8-way intent accuracy alone can hide
        # a classifier that only works well on the majority class.
        from collections import defaultdict

        per_class = defaultdict(lambda: [0, 0])
        for p, g in intent_pairs:
            per_class[g][1] += 1
            if p.lower() == g.lower():
                per_class[g][0] += 1
        print("\n  Per-intent-class accuracy:")
        for cls, (c, t) in sorted(per_class.items()):
            print(f"    {cls:16s} {c}/{t} = {c/t:.3f}")


def cohens_kappa(labels_a, labels_b):
    """Simple two-annotator Cohen's kappa over categorical labels."""
    from collections import Counter

    n = len(labels_a)
    if n == 0:
        return None
    po = sum(1 for a, b in zip(labels_a, labels_b) if a == b) / n
    counts_a = Counter(labels_a)
    counts_b = Counter(labels_b)
    pe = sum((counts_a[k] / n) * (counts_b[k] / n) for k in set(labels_a) | set(labels_b))
    if pe == 1:
        return 1.0
    return (po - pe) / (1 - pe)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sheet", default="data/results/router_labeling_sheet.csv")
    parser.add_argument(
        "--second_sheet",
        default=None,
        help="Optional second annotator's labeled copy of the same sheet, "
        "same row order, for inter-annotator agreement on a subset.",
    )
    args = parser.parse_args()

    rows = load_rows(args.sheet)
    score(rows)

    if args.second_sheet:
        rows2 = load_rows(args.second_sheet)
        if len(rows2) != len(rows):
            print("\nWarning: second sheet has a different row count; skipping agreement calc.")
            return
        print("\nInter-annotator agreement (Cohen's kappa):")
        for dim in DIMENSIONS + ["intent"]:
            gold_col = f"gold_{dim}"
            a = [r.get(gold_col, "").strip() for r in rows]
            b = [r.get(gold_col, "").strip() for r in rows2]
            paired = [(x, y) for x, y in zip(a, b) if x and y]
            if not paired:
                continue
            xs, ys = zip(*paired)
            kappa = cohens_kappa(list(xs), list(ys))
            print(f"  {dim:12s} kappa={kappa:.3f} (n={len(paired)})")


if __name__ == "__main__":
    main()
