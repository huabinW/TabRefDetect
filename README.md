# TabRefDetect

**Numerical Discrepancy Detection in Cross-Document Table Citations**

TabRefDetect is an automated framework designed to detect whether a citing paper's table introduces a quotation error when referencing numerical values from a cited paper's table. It provides both a closed-source multimodal large language model (MLLM) reasoning strategy and a three-stage fine-tuning strategy based on open-source MLLMs.

---

## Project Highlights

- **Research problem**: detects numerical and contextual inconsistencies in
  cross-document scientific table citations.
- **Document intelligence pipeline**: combines MinerU OCR, PageIndex-style
  document structure, table-text trees, and auditable evidence preservation.
- **Agent engineering**: includes a LangGraph workflow agent for table-context
  identification with checkpointing, parallel review, human feedback, and
  Skill approval gates.
- **Structured agent memory**: stores stable rules in core JSON, growing
  human-feedback experience in local SQLite, and only injects a budget-limited
  active memory pack into each run.
- **Hybrid review design**: separates deterministic high-recall candidate
  generation from Codex or future local-model semantic precision review.
- **Model-ready outputs**: preserves table anchors, full parent paragraphs,
  child spans, hashes, page metadata, and labels for downstream classifier
  training.

## Key Engineering Facts

- Built a **14-node LangGraph workflow agent** for auditable scientific
  table-context identification.
- Supports **4 review modes**: `prepare`, `codex`, `existing`, and `manual`.
- Supports **3 learning modes**: `off`, `analyze`, and `propose`.
- Implements **paper-level parallel review** using LangGraph `Send`.
- Includes **controlled self-learning** from human annotations with recall and
  table-coverage guardrails.
- Adds **scalable local memory**: core rules, SQLite long-term memory, and
  retrieved active memory packs instead of full-history loading.
- Maintains **Skill version governance**: pending proposals, explicit human
  approval, and history snapshots.
- Ships with unit tests covering routing, graph compilation, parallel fan-out,
  learning metrics, structured memory, and Skill approval behavior.

---

##  Abstract

[**Objective**] The task of numerical discrepancy detection in cross-document table citations aims to determine whether a citing paper's table introduces a quotation error when referencing numerical values from a cited paper's table. To address the low efficiency and the oversight in manual verification, this study develops an automated detection strategy.

[**Methods**] First, we constructed a human-annotated dataset for this task, named **TabRefError**. This dataset contains numerical alignment samples across diverse table types. Second, we proposed a reasoning strategy based on closed-source multimodal large language models and a three-stage fine-tuning strategy based on open-source multimodal large language models.

[**Results**] Experimental results indicate that the closed-source model achieves a macro F1 score of 0.81. The fine-tuned open-source model achieves a macro F1 score of 0.78, performing comparably to the closed-source model. This validates the effectiveness of the fine-tuning strategy for this task.

[**Limitations**] Although this study achieves automatic detection of numerical discrepancies, discrepancies in experimental settings also should be concerned. The automatic classification of discrepancy causes has not been accomplished.

[**Conclusions**] This research provides a dataset and technical foundation for numerical discrepancy detection in cross-document table quotations. This research also contributes practical value for intelligent editorial review and research evaluation tasks. Future work will include contextual information to achieve automatic classification of the causes of numerical discrepancies.

---

##  Context-Aware Table Reference Analysis

Numerical equality alone is not sufficient to determine whether a table
reference is correct. The same value may refer to different models, datasets,
metrics, prompts, numbers of shots, data splits, baselines, or experimental
settings. Conversely, different values may be justified when the citing and
cited papers report genuinely different experiments.

TabRefDetect therefore includes a document-structure and evidence-construction
pipeline based on **MinerU and PageIndex**. Its research purpose is to preserve
the context required to distinguish justified experimental differences from
probable table citation errors.

### MinerU + PageIndex workflow

- **MinerU** parses each approved PDF into page-level text blocks, table HTML,
  captions, images, bounding boxes, footnotes, and reading-order information.
- **PageIndex** provides the hierarchical document structure and page ranges
  needed to locate tables within sections and subsections.
- The structure tree is reviewed or constructed from available document
  evidence, then combined with MinerU output to form an auditable
  **table-text tree**.
- Each table is attached to the best matching section while retaining its
  canonical label, caption, page number, bbox, MinerU content index, assignment
  reason, and nearby text.

Tables are not treated as isolated artifacts. The resulting representation
keeps the table body together with the surrounding prose that may describe its
datasets, models, metrics, prompts, splits, baselines, and other experimental
conditions. References and bibliography sections are retained as compact
outline nodes by default so that analysis remains focused on table-related
evidence.

This representation supports the next stage of the research: human annotation
and classification of whether a numerical difference is an error, a valid
experimental variation, or a difference whose cause requires additional
context.

The reusable implementation is available in
[`Code/MinerU_PageIndex_TableTree/`](Code/MinerU_PageIndex_TableTree). The
released module includes both:

1. merging approved PageIndex structure snapshots with local MinerU evidence;
2. a fully local PageIndex-style structure builder based on MinerU TOC and
   heading evidence;
3. full OCR paragraph-tree construction and auditable table-parent-child
   candidate generation;
4. a two-stage table-description selector in which deterministic code
   maximizes recall and Codex performs semantic precision review.

The second-stage selector keeps the complete parent paragraph together with
each candidate child block. It uses `0 = correct/relevant` and
`1 = incorrect/irrelevant`. Codex decisions remain provisional silver labels
until they are compared with human annotations.

Reusable Codex workflows are released in [`skill/`](skill/):

- [`tabref-table-tree-audit`](skill/tabref-table-tree-audit) audits table
  counts, positions, traceability, and section assignments.
- [`tabref-table-caption-resolver`](skill/tabref-table-caption-resolver)
  recovers and audits canonical table labels and captions.
- [`tabref-table-text-child-selector`](skill/tabref-table-text-child-selector)
  manages high-recall child generation and Codex semantic precision review.

The repository contains reusable code and empty/example schemas only. PDFs,
MinerU OCR outputs, generated trees, review packages, model decisions, and
paper-specific annotations are not included.

---

##  Open-Source Project Files

This repository includes standard open-source project files:

- [`LICENSE`](LICENSE): Apache-2.0 license.
- [`CONTRIBUTING.md`](CONTRIBUTING.md): contribution workflow and data-safety rules.
- [`SECURITY.md`](SECURITY.md): how to report credential, data-leakage, or unsafe-execution issues.
- [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md): collaboration expectations.
- [`CITATION.cff`](CITATION.cff): citation metadata for GitHub and reference managers.
- [`CHANGELOG.md`](CHANGELOG.md): public-facing release notes.
- [`.github/`](.github): issue templates and pull request checklist.

Please do not submit private PDFs, OCR outputs, prompts, model responses,
human annotations, API keys, local paths, or server credentials in issues or
pull requests.

---

##  Dataset: TabRefError

We introduce **TabRefError**, a human-annotated dataset for numerical discrepancy detection in cross-document table citations. It contains numerical alignment samples across diverse table types.

> **Note:** Since our human annotations cannot guarantee absolute correctness, we currently release only a **portion of the dataset** to ensure quality.

### How to Access

| Access Method | Description |
|---|---|
| **GitHub Release** | A partial subset of the TabRefError dataset is available for download via the [GitHub Releases](https://github.com/huabinW/TabRefDetect/releases) page. |
| **Full Dataset** | For the complete dataset, please contact the authors via email. |

For full dataset access, please contact the corresponding author.

---

##  Getting Started

Install the required Python packages:

```bash
pip install -r requirements.txt
```

### 1. API-based Inference

If you wish to use the closed-source model reasoning strategy, please refer to the API client code:

 [`API_client/`](https://github.com/huabinW/TabRefDetect/tree/main/API_client)

This module contains scripts and instructions for calling closed-source multimodal large language models to perform numerical discrepancy detection.

### 2. Core Pipeline and Model Training

The document-analysis pipeline, training scripts, and supporting experiment
code are available in:

 [`Code/`](https://github.com/huabinW/TabRefDetect/tree/main/Code)

This directory includes the MinerU + PageIndex table-text tree pipeline,
open-source MLLM fine-tuning, OCR+GLM fact checking, and MLLM+SciBERT
classification utilities.

### 3. Codex Skills

The reusable Codex Skills for table-tree auditing, caption resolution, and
table-description child selection are available in:

[`skill/`](https://github.com/huabinW/TabRefDetect/tree/main/skill)

### 4. Workflow Agent

The LangGraph workflow agent for auditable table-context identification is
available in:

[`agent/tabref_context_agent/`](https://github.com/huabinW/TabRefDetect/tree/main/agent/tabref_context_agent)

This agent orchestrates deterministic table-context preprocessing, review
package preparation, Codex or existing-decision routing, paper-level parallel
review, and a controlled human-feedback learning loop. It demonstrates a
research-grade agent workflow with checkpoints, auditable evidence, Skill
proposal approval, structured local memory, and a path toward replacing Codex
review with a local classifier. It releases reusable code and sanitized
configuration templates only; local runtime outputs and annotations are not
included.

---

