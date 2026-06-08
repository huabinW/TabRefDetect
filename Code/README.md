# TabRefDetect Research Code

This directory contains reusable, data-independent implementations for the
core TabRefDetect workflow and its supporting experiments.

- [`MinerU_PageIndex_TableTree/`](MinerU_PageIndex_TableTree): MinerU parsing,
  PageIndex-style structure construction, and auditable table-text tree
  merging. This is a core component for preserving the experimental context
  around cited tables.
- [`OCR_GLM/`](OCR_GLM): OCR result integration and GLM pairwise fact checking.
- [`MLLM_SciBERT/`](MLLM_SciBERT): MLLM-output conversion and SciBERT training.

The export excludes datasets, OCR results, prompts containing sample content,
model predictions, checkpoints, server credentials, and machine-specific
paths.
