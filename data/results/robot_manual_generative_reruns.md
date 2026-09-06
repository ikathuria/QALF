# Robot-manual generative accuracy: before/after the chunk-relevance fix

`data/results/generator_evaluation.csv` stores only the most recent run's
robot-manual rows (resume-cache keeps one row per Subdir/Question). This file
records the two independent post-fix runs cited in the paper
(Section 5.3 / Discussion), since LLM generation temperature is 0.3 and
per-manual accuracy varies run to run at n=3-5 questions/manual.

## Before the fix (positional chunk selection, first k=3 by chunk_index)

| Manual | Accuracy |
|---|---|
| UR5e   | 60.0% |
| SICK   | 50.0% |
| KUKA   | 25.0% |
| Roomba | 20.0% |
| Spot   | 0.0%  |
| **Overall** | **33.3%** |

## After the fix (query-relevance chunk selection via vector.similarity.cosine)

| Manual | Run 1 | Run 2 |
|---|---|---|
| UR5e   | 60.0% | 80.0% |
| SICK   | 50.0% | 50.0% |
| KUKA   | 25.0% | 25.0% |
| Roomba | 40.0% | 40.0% |
| Spot   | 25.0% | 50.0% |
| **Overall** | **41.7%** | **50.0%** |

KUKA, Roomba, and SICK were stable across both runs; UR5e and Spot varied,
consistent with ordinary sampling noise at generation temperature 0.3 on only
3-5 questions per manual, not a further code change between runs.

Mean overall: (41.7 + 50.0) / 2 = **45.8%**, cited in the paper as the headline
number (up from 33.3%, still below DocBench's 76.4%).
