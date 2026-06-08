# OCR + GLM

Reusable OCR-text and GLM fact-checking utilities for cross-document table
citation verification. No dataset files, model outputs, credentials, or local
machine paths are included.

## Files

- `ocr_glm46vflash_reintegrate.py`: parses OCR table structures, builds target
  claims, creates pairwise prompts, calls an OpenAI-compatible local server,
  and writes audit/full result files incrementally.
- `ocr_glm46vflash_direct_author_content.py`: bypasses claim re-extraction and
  directly uses an existing `author_related_content` field.
- `calculate_metrics.py`: computes accuracy and per-class/macro
  precision-recall-F1 from JSON or JSONL results.

## Expected OCR schema

The scripts expect records containing identifiers, labels, `citekey_analysis`,
and `originalkey_analysis`. The direct-input script reads:

```text
originalkey_analysis[].analysis.author_related_content
originalkey_analysis[].analysis.caption
citekey_analysis[].description
```

The reintegration script additionally supports OCR table HTML under the
original-key analysis structure.

## Example

Generate prompts:

```bash
python ocr_glm46vflash_reintegrate.py \
  --ocr-file data/full_results.json \
  --output-dir outputs/prompts
```

Run pairwise inference through a local OpenAI-compatible server:

```bash
python ocr_glm46vflash_reintegrate.py \
  --input-prompts outputs/prompts/ocr_glm46vflash_extraction_prompts.jsonl \
  --prompt-kind pairwise \
  --run-model \
  --api-base http://localhost:8000/v1 \
  --model your-served-model \
  --output-dir outputs/pairwise \
  --resume
```

Evaluate:

```bash
python calculate_metrics.py \
  --input outputs/pairwise/ocr_glm46vflash_pairwise_model_results.jsonl
```
