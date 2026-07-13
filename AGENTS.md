# Project Guidance: TabRefDetect

## Project Goal

This project studies whether a citing paper introduces errors when it references table content from a cited paper.

The core comparison task is:

1. Compare the numerical values in the citing-paper referenced table content against the corresponding table in the cited paper.
2. If the values are the same, verify whether the corresponding experimental conditions are also the same, including model names, dataset names, metrics, experimental settings, prompts, shots, splits, baselines, and any other table-scoped assumptions.
3. If the values differ, determine whether the difference is justified because the citing and cited tables describe genuinely different experiments, or whether the difference likely indicates a citation/reference error.

## Document Analysis Direction

The active direction is now based on MinerU + MinerU-Popo:

- MinerU parses PDFs and provides page-level content blocks, table HTML, captions, bounding boxes, images, and reading-order information.
- MinerU-Popo produces the current document text tree. For future table-text tree construction, Popo `type=text` nodes are the default parent blocks.
- Older PageIndex/GPT-reviewed text trees remain useful historical baselines, but they are no longer the default source for new table-text trees or annotation templates.
- Local code should consume Popo tree outputs and attach tables/text evidence in an auditable way. It should not silently rewrite, summarize, or replace the OCR evidence.
- A Popo parent node can contain several OCR paragraphs. Downstream code must therefore keep the full Popo parent text while also splitting it into smaller child spans for annotation and model training.
- The merged representation should preserve both the Popo document hierarchy and the original local evidence around each table.

For this project, tables must never be treated as isolated artifacts. A useful representation should include:

- The table body and caption.
- Accurate table position, including page number, bbox, and reading-order key.
- Nearby text around the table, especially paragraphs before and after the table.
- Section-level text that describes the table's experimental conditions.
- Page-level context when section boundaries are coarse or ambiguous.

## Table Anchors

The table caption/label stage is an automatic anchoring stage, not the main human annotation task.

- Caption resolution should determine a stable table anchor: canonical table number, title, page, bbox, MinerU content index, and Popo table node.
- Match MinerU/manual table anchors to Popo table nodes with a unique one-to-one
  page/bbox assignment. Never use list order as the deciding match signal. If
  table counts differ, geometry is weak, or more than one complete assignment
  exists, fail the document's anchor audit and do not generate its candidates.
- Debug flags from caption resolution indicate parser edge cases, such as shared caption blocks or recovered captions. They should not automatically become routine human-labeling tasks.
- Supplementary-evidence candidate recall is active in
  `Code/MinerU_PageIndex_TableTree/table_context_pipeline/v2`. Use Popo text
  nodes as primary parents and retain uncovered MinerU `text`, `list`, and
  `page_footnote` blocks as fallback parents.

## Important Constraints

- Do not discard non-table text when building intermediate outputs. The surrounding prose is essential for identifying datasets, models, metrics, and experimental conditions.
- When merging MinerU and Popo results, attach table leaves to the best matching Popo table/text node, but keep the assignment auditable with page number, bbox, caption, Popo block ids, and assignment reason.
- Popo parent nodes may be coarser than OCR paragraphs. Always preserve the complete parent text and record child span offsets.
- The active v2 code stage has no parent/child deletion thresholds, fallback
  floors, or per-table caps before semantic review. Scores and suggestions are
  ordering-only metadata.
- The former `120/60/11/3` policy is a historical compatibility workflow and
  is not an active default.
- Any future candidate-deletion threshold must have an explicit basis and
  human-label validation. Report at least label-`0` recall/precision/F1,
  macro-F1, per-table evidence coverage, false negatives, and false positives
  before promotion.
- Codex precision review uses a supplementary-context test: retain a child only when it adds information not already visible in the table that is needed to interpret, reproduce, or verify it.
- Codex should reject pure table pointers, direct result restatements, promotional efficacy/capability claims, related model/method innovation outside the table's intended content, and evidence belonging to another table or experiment. A result interpretation is retained only when it adds a necessary limitation, metric behavior, dataset difference, or cross-study qualification. A table may legitimately retain zero children.
- Existing human child labels are binding during Codex review. Codex predictions remain provisional and must never be written back as human gold.
- Preserve raw MinerU outputs whenever possible so downstream checks can be reproduced.
- When uploading PDFs to external services or MCP tools, upload only files explicitly approved by the user. Other local files require separate judgment and confirmation.
- Treat references/bibliography sections as outline-only by default. Keep their title and page range, but do not spend analysis effort attaching every reference entry as evidence unless the user explicitly asks for citation-list analysis.

## Expected Outputs

Preferred intermediate outputs should support downstream table-reference verification:

- A structured JSON tree containing Popo nodes, MinerU content leaves, and table leaves.
- For each table, a compact local context window with nearby text, figure/table captions, and page/order metadata.
- Human-readable Markdown summaries for quick inspection.
- Clear notes about uncertainty, especially when table placement or section assignment is inferred.
- Reference/bibliography sections should be compact outline nodes without detailed text leaves in normal table-reference verification workflows.
- Human annotation templates should preserve full Popo parent text, exact child text offsets, table anchors, page/block metadata, and empty human-label fields. Project convention remains `0 = correct/relevant`, `1 = incorrect/irrelevant`.
- Annotation-template summaries should state that v2 has no pre-review
  truncation. Any later calibrated model threshold must include its validation
  report and version.
- Final human-review templates should be compact. Keep `table_label`, `table_caption`, table body, full parent text, child text, offsets, hashes, Codex decision, and empty human label/rationale fields. Preserve full traceability in separate full-audit files.
- Table captions are table-anchor fields. They must be preserved or recovered from caption resolution, but they are not child candidates and should not be reviewed as Codex semantic decisions.

## Codex Review Workflow

- Code produces one complete, unique-child high-recall inventory per paper and
  must not delete candidates before human-label calibration proves recall and
  table-coverage guardrails.
- Codex performs semantic precision review over review packages. Use one agent per paper/package when packages are independent.
- Use a standard prompt with placeholders for package path, output path, and slug. Do not embed historical retained counts or old decision files in a normal review prompt.
- Previous Codex results are used only in explicit comparison/audit mode after a fresh run is materialized.
- If token budget is constrained, prepare packages once, review only missing package decision files, then materialize using existing decisions.

## Publishing Hygiene

- Publish code, Skill releases, and workflow documentation only after checking that raw PDFs, OCR text, generated annotations, decision outputs, secrets, and local-only configs are excluded.
- Replace absolute local paths with configurable examples before publishing reusable code.
- Review `git diff --stat` and targeted file diffs before committing.
- Keep public README language workflow-level and avoid paper-specific evidence, generated labels, or private dataset details.

## Working Style

- Favor transparent, auditable transformations over opaque summaries.
- Keep original page indices and bbox coordinates.
- Treat text, tables, captions, footnotes, and figures as evidence.
- When a generated result appears to omit evidence, inspect the raw MinerU output before concluding that parsing failed.

## Codebase Memory

- Use the installed `codebase-memory-mcp` when exploring the repository, tracing prior implementation decisions, or locating related modules and workflows.
- Treat retrieved memory as navigation and historical context. Verify it against the current files, tests, generated artifacts, and Git history before making changes or reporting conclusions.
- When a completed change introduces a durable architectural decision, workflow convention, threshold rationale, or compatibility constraint, record it through the memory system when that capability is available.
