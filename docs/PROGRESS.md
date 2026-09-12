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

## 2026-09-12: Milestone 1, pipeline complete

- `uv run pytest`: 218 passed in about 0.6 s. Porter matches all 23,531 golden words.
- `uv run python -m pipeline` writes 12 JSON files (largest well under 200 KB) plus
  `search/index.json.gz` (1,249,317 bytes) in 18 s and is byte-identical across runs
  except `generated_at`.
- Anchors: 5,368 documents, 1991-04-15 to 1991-05-14; 32,645 stems from 47,506 word
  forms; 71 evaluable topics, 186 relevant pairs, 1,844 judged documents, 36 topics
  with a single relevant document; BM25 title MAP 0.3696 (0.2887 to 0.4547) against
  0.2912 for the course weighting, paired difference +0.0784 (+0.0167 to +0.1406),
  40 wins, 17 losses, 14 ties; grid best 0.4167 at k1 1.5, b 0.0 (tuned on the scored
  topics); judged@10 0.47.
- Tasks 1 to 4 reviewed by a second agent each; fixes: multi-TEXT join and loud
  failure on malformed records, entity unescape test, duplicate-docno guard.
  Ledger: `.superpowers/sdd/2026-09-12-broadsheet/progress.md`.

## 2026-09-12: Milestone 2, site live with search and charts

- Skeleton, theme and CI (commit `5bd15fe`), CI green on the first run and Pages
  serving 200 at https://sahajm99.github.io/broadsheet/.
- Browser search (commit `ac52a99`): TypeScript Porter with 0 golden mismatches,
  BM25 and tf-idf cosine top-10 identical to the pipeline for all 243 topic titles
  with results, explain panel, about 1 ms per query after a 1.25 MB index load.
- Twelve charts (commits `79454aa`, `676b57d`) screenshotted in light, dark and at
  400 px by their implementers; no console errors.
- Live QA on the deployed page: 12 of 12 figures render, 0 error boxes, 0 unfilled
  `data-stat` spans, "junk bonds" ranks 314 articles in 1 ms, no console errors.
- Portfolio: commit `f73fa7e` on `sahajm99/portfolio` adds the Broadsheet card
  (category `data-engineering`, status live); Vercel deployment READY.

## 2026-09-12: Milestone 3, complete and verified

- Narrative for all six sections, the four-item "what changed" list, a ten-item
  limits list, the kicker chrome removed, the Channel tunnel chip in the markup,
  README final (commit `4e66ac6`); executed notebook with a leak and error check
  (commit `b1dce14`). CI green: https://github.com/sahajm99/broadsheet/actions/runs/34712252832.
- `uv run pytest`: 226 passed. `npm test`: 18 passed (Porter golden, tokenizer,
  browser top-10 parity for both rankers). `npm run typecheck` and `npm run build` clean.
- Live QA on https://sahajm99.github.io/broadsheet/ after the final deploy: 12 of 12
  figures render, 0 error boxes, 0 unfilled `data-stat` spans, 0 kickers, 4 change
  items, 10 limit items, no console errors at 1280 and 400 px; full-page screenshots
  in `.superpowers/sdd/2026-09-12-broadsheet/shots/live-*.png`.
- Not run in CI by design: the pipeline and the notebook (corpus not redistributable).
  A pipeline change requires a local rerun of both and a commit of the outputs.
- Per the user's standing instruction to skip thoroughness, only Tasks 1 to 4 were
  reviewed by a second agent; site tasks got controller screenshots and live QA.
