# Query Router Labeling Guide

`router_labeling_sheet.csv` has 147 real queries (88 from the DocBench subsample, ~59 or so from the robot manuals — exact split depends on the final ingested set) with the router's **predicted** 4D complexity and intent already filled in (`pred_*` columns). Fill in the matching `gold_*` columns with your own judgment of the correct label — this is what lets us compute real router accuracy instead of just trusting the classifier's own output.

You don't have to do all 147 in one sitting — the scoring script (`scripts/score_router_labels.py`) works on however many rows have `gold_intent` filled in, so partial labeling still produces a valid (if noisier) accuracy number. Aim for at least ~60-80 if time is tight; more is better for the confidence interval.

## Complexity dimensions (Low / Medium / High each)

Think of these as "how hard would this query be to answer correctly," not "how long is the sentence" — length is a proxy, not the definition.

- **linguistic**: syntactic complexity — how many clauses, how deep the grammatical structure. A short direct question ("What is the maximum payload?") is Low. A question with a subordinate clause or comparison ("Why did the arm stop, and is this the same fault as last week?") is Medium/High.
- **semantic**: how many distinct entities/facts must be resolved and connected. "What is X?" (one fact) is Low. "Compare X and Y across Z" (multiple entities + a relation) is High.
- **modality**: how much the query depends on non-text content — a table, diagram, or figure. A query answerable from prose alone is Low; one that requires reading a spec table or pointing at a diagram is Medium/High.
- **contextual**: how much the query depends on prior conversation context or specialized/expert domain knowledge (safety, security, adversarial terms). A standalone, general-knowledge question is Low; one using domain jargon or referring back to "the previous step" is Medium/High.

If you genuinely can't tell (ambiguous query), leave it blank rather than guessing — blank rows are just skipped by the scorer, not counted wrong.

## Intent (pick exactly one)

`factual_lookup`, `relationship`, `comparative`, `temporal`, `causal`, `definitional`, `visual_tabular`, `multi_hop`

- **factual_lookup**: a direct fact question ("What is the max speed?")
- **relationship**: asks who created/owns/developed something
- **comparative**: asks to compare, or uses "vs"/"greater"/"better"
- **temporal**: about timing, history, sequence, "before/after/since"
- **causal**: asks why something happened or what caused it
- **definitional**: asks for a definition/meaning
- **visual_tabular**: explicitly about a table, figure, diagram, chart
- **multi_hop**: requires chaining multiple facts/steps together

If a query could plausibly fit two categories, pick the one that best matches what a human would actually want emphasized in the answer, and note the ambiguity in the `notes` column.

## Inter-annotator agreement (optional but valuable)

If both you and Michael can label even a 30-40 row overlapping subset independently (same rows, each in your own copy of the sheet), pass both files to `scripts/score_router_labels.py --sheet your_copy.csv --second_sheet michaels_copy.csv` to get Cohen's kappa — this is exactly the kind of number Reviewer 2 wanted to see (routing isn't evaluated in isolation, and inter-annotator agreement shows the gold labels themselves are reliable).

## Running the scorer

```bash
python scripts/score_router_labels.py --sheet data/results/router_labeling_sheet.csv
```

Prints per-dimension accuracy, overall intent accuracy, and a per-intent-class breakdown (so we can see if the router is only good at the majority class).
