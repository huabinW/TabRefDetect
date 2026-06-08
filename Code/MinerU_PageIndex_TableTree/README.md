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

## Output policy

The module `.gitignore` excludes:

- local manifests;
- input PDFs;
- MinerU outputs;
- generated document trees;
- logs and caches.

Before publishing changes, also scan staged files for absolute paths, API keys,
document text, table content, and dataset identifiers.
