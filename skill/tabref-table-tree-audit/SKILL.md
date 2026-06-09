---
name: tabref-table-tree-audit
description: Audit TabRefDetect table-text tree construction results. Use when Codex needs to verify table counts, table positions, bbox/page traceability, MinerU-to-tree attachment, PageIndex section assignment, manifest overrides, promoted/demoted table-like items, or generated table-text tree JSON/Markdown outputs.
---

# TabRef Table-Text Tree Audit

## Core Rule

Audit only the table-text tree construction layer unless the user explicitly asks for citation-error analysis. Do not start citing-paper vs cited-paper value comparison from this skill.

Use this skill after MinerU parsing and PageIndex/Codex text-tree construction have produced table-text tree outputs.

## Expected Inputs

Run from the TabRefDetect project root. The expected project layout is:

- Manifest: a local copy derived from
  `Code/MinerU_PageIndex_TableTree/manifest.example.json`
- Build script: `Code/MinerU_PageIndex_TableTree/build_table_text_tree_batch.py`
- Audit script: `Code/MinerU_PageIndex_TableTree/audit_batch_table_positions.py`
- Table-text trees, MinerU outputs, and PageIndex/Codex snapshots: directories
  configured by the manifest

## Workflow

1. Rebuild table-text trees when inputs or code changed:

```powershell
python Code/MinerU_PageIndex_TableTree/build_table_text_tree_batch.py `
  --manifest path\to\manifest.json
```

2. Run the batch table position/count audit:

```powershell
python Code/MinerU_PageIndex_TableTree/audit_batch_table_positions.py `
  --manifest path\to\manifest.json
```

3. Inspect the Markdown summary first:

`batch_table_text_tree/table_position_audits/batch_table_position_audit.md`

4. Inspect JSON when a table has `warn` or `fail`:

`batch_table_text_tree/table_position_audits/batch_table_position_audit.json`

## Pass Criteria

A table-text tree passes construction audit when:

- Tree table count equals accepted MinerU table-like item count.
- Every table leaf has `raw_content_index`.
- The source item exists and is table-like under current rules.
- Page index matches the source MinerU item.
- Bbox matches the source MinerU item.
- Assignment reason is present.
- Manifest overrides, if any, point to the actual parent node.
- Non-table figures mistakenly parsed as tables are not counted as table leaves.
- Tables parsed as `code` are promoted only when their caption explicitly contains a table label.

## Known Non-Fatal Warnings

Treat these as expected only when documented in output:

- `source_caption_empty`: MinerU found the table body but missed its caption. Resolve caption with `tabref-table-caption-resolver`.
- `caption_label_fallback`: label was inferred by order because local caption was absent.
- `source_code_promoted_to_table`: MinerU parsed a real table as `code`; acceptable if `code_caption` contains a table label.
- `figure_like_tables_excluded`: MinerU parsed a figure as `table`; acceptable if caption explicitly says Figure/Fig. and contains no Table label.

## Common Failure Modes

Check for these and fix code or manifest transparently:

- PageIndex page ranges are coarse, causing a table to attach to a neighboring section.
- Appendix sections share a page, so page-only attachment is insufficient.
- References/bibliography filtering accidentally hides real text because of substring matching. Use whole-word matching.
- Root/title contains words like `Coreference`; do not classify it as references.
- A single caption block contains multiple labels, such as `Table 7 ... Table 8 ...`.
- A true table is parsed as `code`.
- A figure, interface screenshot, or form is parsed as `table`.
- MinerU output directory is slug-based rather than PDF-stem-based.
- Chunked MinerU outputs need merged images and rewritten `img_path`.

## Fix Policy

- Prefer general rules in `build_table_text_tree_batch.py` for repeated patterns.
- Use manifest `table_parent_overrides` for document-specific section corrections.
- Always record corrections in `assignment_reason`, e.g. `manifest_override:Table 4->0016`.
- Do not silently edit generated JSON by hand.
- Preserve raw MinerU outputs and raw table items for reproducibility.
- If a new recurring failure mode appears, update this skill after fixing the workflow.

## Report Back

Summarize:

- Table count by document.
- Pass/warn/fail count.
- Any warn/fail table labels.
- Whether warnings are expected parser edge cases or unresolved issues.
- Paths to generated reports.
