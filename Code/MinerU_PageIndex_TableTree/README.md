# MinerU + PageIndex Table-Text Tree

It supports two structure sources:

1. Existing PageIndex/GPT-reviewed structure snapshots combined with MinerU
   evidence.
2. A fully local PageIndex-style structure pipeline that replaces PageIndex
   LLM calls with MinerU TOC and heading evidence.

The local path does not call an LLM, upload PDFs, or generate node summaries.

## Prerequisites

- Python 3.10 or later.
- A working [MinerU](https://github.com/opendatalab/MinerU) CLI for local PDF
  parsing.
- No PageIndex package or service is required for the fully local path.

The structure stages are based on the processing flow published by
[PageIndex](https://github.com/VectifyAI/PageIndex). This implementation uses
MinerU evidence in place of PageIndex's LLM extraction and verification calls.

## Data boundary

The repository contains only reusable code and a placeholder manifest.
Do not commit PDFs, MinerU OCR output, generated trees, table HTML, captions,
model responses, credentials, or document-specific overrides.

Generated outputs preserve document text, table bodies, captions, bboxes, and
page positions, so they must be treated as private experiment artifacts.

## Layout

| File | Purpose |
|---|---|
| `run_mineru_batch.py` | Run MinerU for documents listed in the manifest. |
| `build_mineru_pageindex_text_trees.py` | Build local PageIndex-style text trees from MinerU evidence. |
| `build_local_text_tree_batch.py` | Internal helpers for TOC, heading, page, and table-anchor evidence. |
| `build_table_text_tree_batch.py` | Merge external PageIndex structure snapshots with MinerU leaves. |
| `build_local_table_text_tree_batch.py` | Merge local PageIndex-style trees with MinerU leaves. |
| `validate_text_tree_batch.py` | Validate node IDs and page ranges. |
| `resolve_table_captions.py` | Resolve and audit table labels/captions. |
| `resolve_local_table_captions.py` | Run caption resolution on local table-text trees. |
| `audit_batch_table_positions.py` | Audit table count, page, bbox, and assignment traceability. |
| `build_local_paragraph_table_text_tree_batch.py` | Build full, untruncated MinerU OCR paragraph leaves. |
| `materialize_manual_table_body_text_annotations.py` | Convert private parent-paragraph selections into traceable annotations. |
| `build_table_text_classifier_candidates.py` | Build table-parent-child classification candidates. |
| `select_table_description_child_blocks.py` | Preserve high-recall child candidates and compute transparent ranking signals. |
| `prepare_codex_child_review_packages.py` | Package table, parent, and child evidence for Codex semantic review. |
| `run_codex_child_semantic_review.py` | Launch one Codex precision-review task per document. |
| `materialize_codex_child_review_results.py` | Validate and materialize final Codex labels. |
| `popo_workflow_helpers.py` | Shared MinerU-Popo text/table scoring helpers. |
| `prepare_popo_strict_human_child_annotation_packages.py` | Build a Popo strict high-recall child-candidate annotation pool. |
| `prepare_popo_codex_precision_review_packages.py` | Package Popo strict candidates for Codex semantic precision review. |
| `run_popo_codex_precision_review.py` | Launch or dry-run one Codex review per Popo package. |
| `materialize_popo_codex_precision_review_results.py` | Validate Popo Codex decisions and write slim annotation plus full-audit outputs. |
| `run_local_table_text_training_pipeline.py` | Run the deterministic preparation stages in sequence. |

## Configuration

Copy `manifest.example.json` to `manifest.json` and edit only local paths:

```powershell
Copy-Item manifest.example.json manifest.json
```

Minimal document entry:

```json
{
  "slug": "example_document",
  "pdf_path": "inputs/example_document.pdf",
  "pageindex_doc_name": "example_document.pdf",
  "page_count": 12,
  "table_parent_overrides": {}
}
```

`page_idx` from MinerU is treated as zero-based. Tree `start_index` and
`end_index` values are one-based PDF page numbers.

## Fully local workflow

Run from this directory:

```powershell
python run_mineru_batch.py
python build_mineru_pageindex_text_trees.py
python validate_text_tree_batch.py
python build_local_table_text_tree_batch.py
python resolve_local_table_captions.py
```

The local text-tree builder follows the open-source PageIndex processing shape:

```text
check_toc
  -> process_toc_with_page_numbers | process_no_toc
  -> validate_and_truncate_physical_indices
  -> add_preface_if_needed
  -> post_processing / list_to_tree
```

MinerU TOC, heading, reading-order, page, and bbox evidence replaces the LLM
extraction and verification calls. Node summaries are omitted. Full section
content is attached later as MinerU text, media, and table leaves.

## External PageIndex snapshot workflow

Place one approved structure JSON per document at:

```text
outputs/table_text_tree/pageindex_structures/<slug>.json
```

Each file must contain a top-level `structure` list with `title`, `node_id`,
`start_index`, and `end_index` fields. Then run:

```powershell
python build_table_text_tree_batch.py
python audit_batch_table_positions.py
python resolve_table_captions.py
```

Only upload PDFs to PageIndex or another external service when the document has
been explicitly approved for upload.

## Table-description child selection

The table-description workflow separates recall and precision:

1. MinerU OCR is attached as full, untruncated parent paragraphs.
2. Private manual selections identify relevant table-parent relations.
3. Deterministic code preserves every child under a relevant parent to maximize
   recall. Rule scores are ranking aids, not final labels.
4. The active candidate policy marks `send_to_semantic_review`, which controls
   the semantic-review queue without deleting preserved child evidence.
5. Codex reads the table, complete parent paragraph, and exact child block to
   assign the semantic precision label.
6. The materializer verifies review keys, child IDs, SHA-256 hashes, labels,
   semantic roles, and decision completeness.

Label semantics are:

```text
0 = correct/relevant
1 = incorrect/irrelevant
```

Create a private selections file from the released schema:

```powershell
Copy-Item manual_table_body_text_selections.example.json `
  manual_table_body_text_selections.json
```

The selections file contains document-specific annotations and is excluded from
version control. Run the deterministic stages with:

```powershell
python run_local_table_text_training_pipeline.py `
  --manifest manifest.json `
  --selections manual_table_body_text_selections.json `
  --allow-draft
```

Then perform and validate the Codex precision stage:

```powershell
python run_codex_child_semantic_review.py --manifest manifest.json
python materialize_codex_child_review_results.py --manifest manifest.json
```

### MinerU-Popo strict workflow

For the MinerU-Popo branch, use Popo `type=text` nodes as candidate parents and
keep the code stage recall-oriented:

```powershell
python prepare_popo_strict_human_child_annotation_packages.py
python prepare_popo_codex_precision_review_packages.py
python run_popo_codex_precision_review.py --dry-run
python materialize_popo_codex_precision_review_results.py
```

The normal review path is one package per Codex agent with the standard prompt
from `skill/tabref-table-text-child-selector/references/five-agent-review-prompt.md`.
The materializer writes a compact `*.slim.json` template for human annotation
and separate full-audit JSON files for traceability. Captions are preserved as
table-anchor fields and are not reviewed as child candidates.

The corresponding Codex Skills are published under the repository-level
[`skill/`](../../skill/) directory.

## Output policy

The module `.gitignore` excludes:

- local manifests;
- private manual parent-paragraph selections;
- input PDFs;
- MinerU outputs;
- generated document trees;
- logs and caches.

Before publishing changes, also scan staged files for absolute paths, API keys,
document text, table content, and dataset identifiers.
