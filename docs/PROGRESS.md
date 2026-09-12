# Progress log

## 2026-09-12: Milestone 0, setup

- Read all three course phases (tokenizer, indexer, query processor), both reports,
  the corpus (5,368 docs, 15 files), the course qrels (50 topics, 686 FT911 lines,
  only 16 topics with a relevant FT911 document) and fetched the NIST Robust 2004
  qrels and topic files (71 evaluable topics, 186 relevant pairs in this slice).
- Found the course defect that motivates the rebuild: phase 3 wrote `Doc_<id>`
  instead of document numbers, so its output could never be scored.
- Wrote `docs/DESIGN.md`, `docs/DECISIONS.md`, the plan, copied the corpus to the
  git-ignored `data/raw/ft911/` with `SHA256SUMS`, committed the TREC files,
  stopwords, Porter golden vectors and the course notebooks for reference.
