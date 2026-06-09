---
name: tabref-table-text-child-selector
description: Select and audit the smallest body-text child blocks that describe a specified table in a TabRefDetect table-text tree. Use when Codex needs to identify table references, experimental conditions, datasets, models, metrics, prompts, shots, splits, baselines, training settings, ablations, or result interpretations associated with a table; return each child with its complete parent paragraph, page, bbox, MinerU indices, selection evidence, and binary labels where 0 means correct/relevant and 1 means incorrect/irrelevant. Also use to compare predictions against user-provided annotations and refine the existing selector without creating a replacement skill.
---

# TabRef Table Text Child Selector

Current version: `0.2.2`.

Find the minimal body-text spans that describe a table while preserving their complete parent paragraphs and original MinerU traceability.

## Scope

- Select table-related child blocks from a local table-text tree.
- Preserve table caption/body, parent paragraph, section, page, bbox, reading order, and MinerU indices.
- Label correct/relevant relations `0`; label incorrect/irrelevant relations `1`.
- Separate deterministic high-recall candidate generation from Codex semantic precision review.
- Treat code-stage child labels as candidates, not final semantic labels.
- Compare later user annotations against current predictions and update this same Skill.

Do not:

- Summarize, truncate, or rewrite OCR evidence.
- Infer citation errors between citing and cited papers.
- Compare table values across papers.
- Replace manual gold annotations with heuristic predictions.
- Create another child-selector Skill when refining this workflow.

## Workflow

1. Locate the full candidate dataset and manual parent annotations.
2. Confirm parent labels use `0 = correct/relevant`, `1 = incorrect/irrelevant`.
3. Run `scripts/select_table_description_child_blocks.py`.
4. Preserve every child under a correct parent so the code stage maximizes recall.
5. Run `scripts/prepare_codex_child_review_packages.py`.
6. Run `scripts/run_codex_child_semantic_review.py`, or read each package in the current Codex session and write its decisions.
7. Run `scripts/materialize_codex_child_review_results.py` to validate completeness and produce final outputs.
8. Report code-stage candidate counts separately from Codex final labels and human gold.

Default project command:

```powershell
python skill\tabref-table-text-child-selector\scripts\select_table_description_child_blocks.py `
  --manifest Code\MinerU_PageIndex_TableTree\manifest.json
```

The bundled wrappers call the reusable implementation under
`Code/MinerU_PageIndex_TableTree/`. Run them from the repository root.

## Selection Logic

For a parent labeled `1`, assign all children label `1`.

For a parent labeled `0`, preserve every child and score it using:

- Exact `Table N` or `Tab. N` reference.
- Lexical overlap with table caption and table body.
- Lexical overlap with the manual parent-selection rationale.
- Evidence-family match: dataset, model, metric, training, method, or result.
- Experimental-condition and result-description cues.

The score only orders review work. It must not remove children or become the final label.

## Codex Precision Review

Judge each high-recall candidate from the table caption/body, complete parent paragraph, and exact child text.

Assign `0` when the child:

- Introduces what the table contains.
- Supplies table-scoped datasets, models, metrics, prompts, shots, splits, baselines, training settings, or other experimental conditions.
- Interprets, compares, qualifies, or limits the table results.
- Provides evidence useful to later citation-reference judgment.

Assign `1` when the child is generic background, a transition or heading fragment, redundant, unrelated, or insufficiently table-scoped.

Record `semantic_role`, `citation_support`, and a concise rationale. The code may validate and materialize these fields, but it must not invent them.

Always return:

- `child_text`, exact offsets, and child label.
- `full_parent_text` and parent label.
- Table anchor and table body.
- Page, bbox, section, reading-order, and MinerU indices.
- Selection signals, reason, source, and review status.

## Outputs

Maintain four distinct artifacts:

- Full candidate dataset with parent and child labels.
- JSONL training samples containing table, parent, and child together.
- Codex review packages containing complete evidence.
- Final Codex-reviewed JSON/JSONL.
- Human-readable Markdown listing retained label `0` and Codex-demoted label `1` separately after every table.

Treat the score as an interpretable ranking aid, not a calibrated probability.

Mark code-stage candidates:

```text
child_label_source = code_high_recall_v1
child_review_status = awaiting_codex_semantic_review
```

Mark final model decisions:

```text
final_child_label_source = codex_semantic_review_v1
```

Never present code candidates or Codex decisions as human gold labels.

The automated launcher starts one Codex CLI review per paper and materializes
results only after every expected decision file exists and passes validation.

## Updating With User Gold

When the user supplies child-level annotations:

1. Preserve the supplied annotation unchanged as a gold artifact.
2. Match records using document slug, canonical table label, parent raw indices, and child offsets/text.
3. Produce false-positive and false-negative audits by evidence family and selection signal.
4. Modify the bundled selector and this workflow in place.
5. Re-run the existing five-paper regression set.
6. Increment the version in both this file and `references/version.json`.

Use semantic versioning:

- Patch: thresholds, bug fixes, output metadata.
- Minor: new signals, annotation schema, or selection behavior.
- Major: incompatible label or input/output contract.

Read `references/output-schema.md` when modifying output fields.
