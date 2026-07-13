# Table-context pipeline

This package contains the active MinerU + MinerU-Popo table-context workflow.

- `v2/` is the active scope-aware workflow.
- `config/` contains anonymous schemas that should be copied to local-only files.
- The older threshold-based Popo scripts in the parent directory remain available only for
  historical reproduction.

V2 preserves each body-text child once, attaches its complete parent, and exposes every table in
the paper to semantic review. Ranking scores and scope suggestions order work; they do not delete
evidence. Popo text nodes are primary parents, while uncovered MinerU `text`, `list`, and
`page_footnote` blocks remain available as fallback evidence.

Generated inventories, review packages, decisions, human annotations, PDFs, and OCR outputs must
remain outside this source directory and must not be committed.
