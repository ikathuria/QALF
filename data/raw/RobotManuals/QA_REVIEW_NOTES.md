# Robot Manual QA Corpus — Review Notes

24 QA pairs drafted across 5 real, publicly available, official manufacturer manuals for the IEEE HRI paper's supplementary "Robot/Equipment Manual QA" evaluation. **Please spot-check before these are used in any reported results** — I (Claude) authored the questions and answers directly from extracted PDF text, but did not independently verify against the original PDF layout in every case, particularly the one table-derived question flagged below.

**Husqvarna Automower was dropped from the corpus** (2026-09-05) after its text content reproducibly hung our ingestion pipeline's graph-extraction step across three separate attempts (once for ~17.5 hours, twice more killed after 6-25 minutes with zero progress logged), even after adding a request timeout and a `num_predict` generation cap — neither fix changed the outcome, and the hang occurred at the identical point every time, before any LLM call progress was logged. This points to a text-processing bug (likely regex catastrophic backtracking, given the manual's text had unusual bullet/warning-symbol characters) rather than a QALF retrieval-fusion issue, but there wasn't time to root-cause it before the deadline. Its 4 QA pairs (`Husqvarna_Automower_Manual_qa.jsonl`, `Husqvarna/Husqvarna_qa.jsonl`) are left in this directory for reference but are **not** part of the reported 24-pair / 5-manual corpus.

## Sources (all official manufacturer PDFs)
- `UR5e/` — Universal Robots UR5e User Manual — **truncated to a 12-page excerpt** (pages 25-36 of the original 205pp manual, covering the "Safety-related Functions and Interfaces" chapter our QA pairs are grounded in). The original full PDF was replaced in place with this excerpt to make ingestion tractable (full-manual MinerU parsing on this machine's CPU was projected at 1-2+ hours for this document alone).
- `Spot/` — Boston Dynamics Spot Instructions for Use v2.1.2 — **truncated to a 10-page excerpt** (pages 13-22 of the original 126pp manual, covering the safety-related stop section).
- `SICK/` — SICK AG "Safe Robotics" white paper on collaborative robot safety (8pp, full document, not truncated)
- `KUKA/` — KUKA Sunrise Cabinet Med Instructions for Use — **truncated to a 17-page excerpt** (pages 20-36 of the original 114pp manual, covering the stop-category section).
- `Husqvarna/` — Husqvarna Automower operator's manual (60pp, full document, not truncated)
- `Roomba/` — iRobot Roomba s9 owner's guide (6pp, full document, not truncated)

**Important:** UR5e, Spot, and KUKA are curated excerpts, not full manuals — verified (via `pdftotext` + grep) to still contain every quote the QA pairs below cite before truncating. This was a deliberate speed/tractability trade-off (GPU acceleration was attempted for the MinerU parsing step but segfaulted consistently across three different CUDA builds, likely a driver/hardware/RDP-session issue with the RTX 5090s available on this machine — reverted to CPU). The paper describes this truncation explicitly rather than implying full-manual scale.

## What to check
1. All 28 QA pairs are grounded in directly quoted text (see `evidence` field in each `*_qa.jsonl`) — please confirm the quotes are accurate and the answers correctly follow from them.
2. **One table-derived question** (`Roomba/Roomba_qa.jsonl`, the Filter maintenance-frequency question) was read from a `pdftotext -layout` extraction of a multi-column table; table extraction can misalign columns. Please verify this one directly against the PDF page before use.
3. I deliberately avoided drafting questions from other tables in these manuals (e.g., a garbled spec table in the Spot manual) because the layout-extracted numbers looked misaligned and I did not want to risk publishing an incorrect gold answer. If you want more table/spec-based (`multimodal-t`) questions for a stronger multimodal signal, someone should read the original PDF pages directly and add them.
4. No questions were drafted from images/diagrams (`multimodal-i` type) since I only had OCR'd/extracted text, not the actual page images, to verify against — if the paper wants that evidence type represented, someone with the rendered PDF pages should add a few.

## Format
Each `<Name>/<Name>_qa.jsonl` sits next to `<Name>/<Name>.pdf`, matching the same `question`/`answer`/`type`/`evidence` schema DocBench uses, so the existing evaluators (`retrieval_evaluator.py`, `generator_evaluator.py`, etc.) can run over this corpus without code changes — just point `--doc_bench_dir` at `data/raw/RobotManuals`.
