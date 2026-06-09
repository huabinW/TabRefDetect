---
name: tabref-table-caption-resolver
description: Resolve and audit whether each table correctly corresponds to its table title and table number in TabRefDetect. Use when Codex needs to recover table captions from MinerU output, handle captions parsed as text, match Table/Tab./表/Table S labels to bboxes, disambiguate adjacent tables, or find full-document paragraphs that mention a table number.
---

# TabRef Table Caption Resolver

## Core Rule

Use rule-based evidence first. The goal is not to summarize the table, but to locate the correct table title, canonical table number, and numbered prose references that describe the table.

This skill complements `tabref-table-tree-audit`: run caption resolution after table count/position auditing, or whenever a table caption/label looks suspicious.

## Expected Inputs

Run from the TabRefDetect project root. The expected project layout is:

- Resolver script: `Code/MinerU_PageIndex_TableTree/resolve_table_captions.py`
- Manifest: a local copy derived from
  `Code/MinerU_PageIndex_TableTree/manifest.example.json`
- Table-text trees: the output directory configured by the manifest
- MinerU `content_list.json` files: the MinerU directory configured by the
  manifest

## Workflow

1. Run syntax validation if the resolver changed:

```powershell
python -m py_compile Code/MinerU_PageIndex_TableTree/resolve_table_captions.py
```

2. Run batch caption resolution:

```powershell
python Code/MinerU_PageIndex_TableTree/resolve_table_captions.py `
  --manifest path\to\manifest.json
```

3. Inspect the Markdown report first:

`batch_table_text_tree/table_caption_resolution/batch_table_caption_resolution.md`

4. Use the JSON for downstream code:

`batch_table_text_tree/table_caption_resolution/batch_table_caption_resolution.json`

5. When explaining or revising thresholds, consult:

`batch_table_text_tree/table_caption_resolution/caption_resolution_methodology.md`

## What to Resolve

For each table, identify:

- `canonical_label`, such as `Table 1`, `Table S1`, or `Table A1`.
- The best title/caption span beginning at that label.
- Whether the title came from MinerU `table_caption`, `code_caption`, or surrounding text.
- Confidence, interpretable evidence tags, and flags.
- Full-document paragraphs that mention the same table number.

## Label Patterns to Consider

Support at least:

- `Table 1`, `table 1`, `TABLE 1`
- `Tab. 1`, `Tab 1`
- `表 1`, `表1`
- `Table S1`, `Table S2`
- `Supplementary Table 1`
- `Supplemental Table 1`
- `Extended Data Table 1`
- Appendix-style labels such as `Table A1`
- Letter suffixes such as `Table 1a` or `Table 1b`

When new formats appear, update `TABLE_LABEL_RE` in `resolve_table_captions.py` and then update this skill.

## Candidate Sources

Search these sources in order, but keep all candidates for traceability:

- MinerU `table_caption`.
- MinerU `code_caption` when a true table is parsed as code.
- Text blocks immediately above the table.
- Text blocks immediately below the table.
- Neighboring content items in reading order.
- Same-page text containing a table label.
- Adjacent-page text when captions spill across pages.

## Disambiguation Rules

Use multiple signals together:

- Same page as table.
- Bbox relation: above, below, overlap, vertical distance.
- Horizontal overlap with the table bbox.
- MinerU reading-order distance from the table item.
- Exact or canonical label match to current table label.
- Caption-like position: label near the start of a text block.
- Previously used labels: avoid assigning the same label twice unless evidence requires it.
- Multiple labels in one text block: split candidate title spans by label positions.

These are deterministic rule-based heuristics grounded in document layout analysis and scientific figure/table extraction practice: caption cue detection, page-level layout regions, bounding-box proximity, alignment, and reading-order adjacency. Treat numeric thresholds as transparent operational definitions of those layout cues, not as learned probabilities.

For adjacent tables:

- Do not assign a caption only by nearest text if the text contains multiple labels.
- Prefer the candidate whose label matches the table's current/inferred label.
- If one table has empty MinerU caption and the next caption block contains two labels, assign the first label to the upper/previous table and the second label to the lower/next table when bbox/order supports it.

## Flags to Preserve

Do not hide ambiguity. Keep flags such as:

- `caption_recovered_from_surrounding_content`
- `source_text_contains_multiple_table_labels`
- `selected_label_differs_from_current_table_label`
- `nearby_competing_caption_candidate`
- `caption_bbox_missing_or_unusable`
- `no_caption_candidate`

These flags are anchor-debug signals. They are useful for improving parser rules, but they are not the main human annotation target.

## Numbered Paragraph Search

After selecting a canonical table label, search the full MinerU document for paragraphs mentioning that number. Exclude table/code/page-number items by default. Store matches as `full_document_references`.

These paragraphs may later help table-supplementary-evidence analysis because they often contain:

- Experimental setup.
- Dataset/model/metric explanations.
- Ablation descriptions.
- Notes that are not in the caption.
- Conditions required for later citation-error checking.

Do not implement or filter supplementary-evidence candidate paragraphs from this skill unless the user explicitly resumes that later stage.

## Common Edge Cases

- MinerU caption is empty but nearby text contains the title.
- MinerU caption is partly wrong or starts with a figure label.
- A caption has multiple table labels.
- A table is split visually but parsed as one item.
- Two small tables are stacked closely.
- A figure or interface screenshot is parsed as `table`.
- A real table is parsed as `code`.
- Caption is in Chinese or mixed Chinese/English.
- Caption appears below the table rather than above it.
- Caption appears on previous/next page.
- A prose paragraph references a table but is not the caption.

## Fix Policy

- Prefer improving `resolve_table_captions.py` rules for recurring patterns.
- Do not mutate raw MinerU output.
- Preserve all candidates and selected evidence in JSON.
- If a new recurring caption format or ambiguity appears, update both the resolver and this skill.

## Report Back

Summarize:

- Confidence counts by document.
- Tables with flags.
- Whether each flagged case is expected or unresolved.
- Paths to caption resolution JSON/Markdown.
