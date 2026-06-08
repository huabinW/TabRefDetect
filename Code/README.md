# Additional Experiment Code

This directory contains reusable, data-independent implementations added for
the TabRefDetect experiments.

- [`OCR_GLM/`](OCR_GLM): OCR result integration and GLM pairwise fact checking.
- [`MLLM_SciBERT/`](MLLM_SciBERT): MLLM-output conversion and SciBERT training.
- [`MinerU_PageIndex_TableTree/`](MinerU_PageIndex_TableTree): MinerU parsing,
  PageIndex-style structure construction, and auditable table-text tree merging.

The export excludes datasets, OCR results, prompts containing sample content,
model predictions, checkpoints, server credentials, and machine-specific
paths.
