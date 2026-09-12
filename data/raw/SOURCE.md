# Corpus source

The corpus is the `ft911` subset of the Financial Times 1991 collection
(TREC disk 4): 15 files, `ft911_1` .. `ft911_15`, holding 5,368 `<DOC>`
records in TREC SGML.

It was provided as course material for CSCE 5200 (Information Retrieval),
University of North Texas, Fall 2024.

The content is licensed and is **not** redistributed with this repository:
`data/raw/ft911/` is git-ignored. To run the pipeline, place the 15 files
in `data/raw/ft911/` exactly as issued (no renaming, no re-encoding).

Integrity: `data/raw/SHA256SUMS` holds the SHA-256 of every file. Verify with

```
sha256sum -c data/raw/SHA256SUMS
```

from the repository root.
