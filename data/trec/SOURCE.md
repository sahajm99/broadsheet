# TREC topics and relevance judgments

## Topics

`topics.301-450.601-700.txt` — 250 topics in TREC `<top>` format.

- Topics 301-350, 351-400 and 401-450 are published at
  <https://trec.nist.gov/data/topics_eng/>.
- The committed file is the TREC 2004 Robust track test set,
  <https://trec.nist.gov/data/robust/04.testset.gz> (unpacked `04.testset`,
  250 topics: 301-450 and 601-700).

## Relevance judgments

`qrels.ft911.txt` — the Robust 2004 qrels,
<https://trec.nist.gov/data/robust/qrels.robust2004.txt>, filtered to the
documents in this corpus on 2026-09-12 with

```
awk '$3 ~ /^FT911-/' qrels.robust2004.txt > qrels.ft911.txt
```

which yields 3,019 lines.

Both files are NIST data and are public.
