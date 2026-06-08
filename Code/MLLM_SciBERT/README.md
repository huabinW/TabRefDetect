# MLLM + SciBERT

Utilities for converting MLLM semantic-matching outputs into SciBERT
classification datasets and training a binary sentence-pair classifier.
Dataset records and experiment outputs are intentionally excluded.

## Files

- `build_scibert_from_semantic_data.py`: extracts ground-truth and citation
  sections from semantic SFT JSONL and writes single-text and sentence-pair
  SciBERT datasets.
- `build_scibert_cross_model_dataset.py`: builds a cross-model train/validation
  setting from configurable source file templates.
- `train_scibert_semantic.py`: trains and evaluates SciBERT with optional
  evidence-aware chunking, class weights, and group-level aggregation.

## Labels

```text
0 = factually supported / match
1 = unsupported or inconsistent / mismatch
```

## Example

Convert semantic data:

```bash
python build_scibert_from_semantic_data.py \
  --semantic-dir data/semantic \
  --model-source mllm \
  --fold 0 \
  --output-dir outputs/dataset/fold0
```

Train:

```bash
python train_scibert_semantic.py \
  --train-file outputs/dataset/fold0/train_scibert_pair.jsonl \
  --val-file outputs/dataset/fold0/val_scibert_pair.jsonl \
  --mode pair \
  --model-name allenai/scibert_scivocab_uncased \
  --output-dir outputs/model/fold0
```
