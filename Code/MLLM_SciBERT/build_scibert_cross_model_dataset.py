import argparse
import json
import re
from collections import Counter
from pathlib import Path

from build_scibert_from_semantic_data import (
    convert_row as convert_semantic_row,
    label_distribution,
    read_jsonl,
    score_summary,
    write_json,
    write_jsonl,
)


def normalize_space(value):
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def unique_texts(values):
    seen = set()
    output = []
    for value in values:
        text = normalize_space(value)
        if not text or text in seen:
            continue
        seen.add(text)
        output.append(text)
    return output


def label_to_int(value):
    text = str(value).strip()
    if text not in {"0", "1"}:
        raise ValueError(f"Expected label 0 or 1, got {value!r}")
    return int(text)


def convert_glm_row(row, source_file):
    sample_id = normalize_space(row.get("id") or row.get("paper_id"))
    label = label_to_int(row.get("label"))
    matching_results = row.get("matching_results") or []
    if not isinstance(matching_results, list):
        matching_results = []

    cited_claims = unique_texts(
        result.get("claim") for result in matching_results if isinstance(result, dict)
    )
    if not cited_claims:
        cited_claims = unique_texts(row.get("claim") or [])

    author_related_contents = []
    captions = []
    for result in matching_results:
        if not isinstance(result, dict):
            continue
        original_item = result.get("originalkey_item") or {}
        if not isinstance(original_item, dict):
            continue
        author_related_contents.append(original_item.get("author_related_content"))
        captions.append(original_item.get("caption_annotations"))

    author_related_contents = unique_texts(author_related_contents)
    captions = unique_texts(captions)

    cited_facts_text = "\n\n".join(
        f"cited evidence {index + 1}: {claim}" for index, claim in enumerate(cited_claims)
    )
    citing_author_related_content = "\n\n".join(author_related_contents)
    citing_caption_text = "\n\n".join(captions)

    text_a = f"cited paper facts: {cited_facts_text}" if cited_facts_text else "cited paper facts:"
    text_b_parts = []
    if citing_author_related_content:
        text_b_parts.append(f"originalkey author_related_content: {citing_author_related_content}")
    else:
        text_b_parts.append("originalkey author_related_content:")
    if citing_caption_text:
        text_b_parts.append(f"originalkey table caption: {citing_caption_text}")
    text_b = "\n\n".join(text_b_parts)
    text = f"[CITED_FACTS]\n{text_a}\n\n[SEP]\n\n[ORIGINALKEY_CITING]\n{text_b}"

    return {
        "id": sample_id,
        "label": label,
        "text": text,
        "text_a": text_a,
        "text_b": text_b,
        "cited_facts_text": cited_facts_text,
        "citing_author_related_content": citing_author_related_content,
        "citing_caption_text": citing_caption_text,
        "source_file": source_file,
        "extraction_status": "structured_glm_results",
        "num_cited_claims": len(cited_claims),
        "num_author_related_blocks": len(author_related_contents),
        "num_caption_blocks": len(captions),
        "num_matching_pairs": len(matching_results),
        "cited_facts_char_length": len(cited_facts_text),
        "citing_author_related_char_length": len(citing_author_related_content),
        "citing_caption_char_length": len(citing_caption_text),
        "text_char_length": len(text),
    }


def to_minimal_text_row(row):
    return {
        "id": row["id"],
        "text": row["text"],
        "label": row["label"],
    }


def to_minimal_pair_row(row):
    return {
        "id": row["id"],
        "text_a": row["text_a"],
        "text_b": row["text_b"],
        "label": row["label"],
    }


def load_json_list(path):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise TypeError(f"{path} must contain a JSON list")
    return data


def build_dataset(
    root,
    output_dir,
    fold,
    train_source,
    test_source,
    train_pattern,
    validation_pattern,
):
    semantic_train_path = root / train_pattern.format(source=train_source, fold=fold)
    validation_path = root / validation_pattern.format(source=test_source, fold=fold)
    if not semantic_train_path.exists():
        raise FileNotFoundError(semantic_train_path)
    if not validation_path.exists():
        raise FileNotFoundError(validation_path)

    train_rows = [
        convert_semantic_row(row, semantic_train_path.name)
        for row in read_jsonl(semantic_train_path)
    ]
    val_rows = [
        convert_glm_row(row, validation_path.name)
        for row in load_json_list(validation_path)
    ]

    train_ids = {row["id"] for row in train_rows}
    val_ids = {row["id"] for row in val_rows}
    overlap = sorted(train_ids & val_ids)
    if overlap:
        raise ValueError(f"Train/val id overlap detected: {overlap[:10]}")

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "train.jsonl", train_rows)
    write_jsonl(output_dir / "val.jsonl", val_rows)
    write_json(output_dir / "train.json", train_rows)
    write_json(output_dir / "val.json", val_rows)

    write_jsonl(output_dir / "train_scibert_text.jsonl", [to_minimal_text_row(row) for row in train_rows])
    write_jsonl(output_dir / "val_scibert_text.jsonl", [to_minimal_text_row(row) for row in val_rows])
    write_jsonl(output_dir / "train_scibert_pair.jsonl", [to_minimal_pair_row(row) for row in train_rows])
    write_jsonl(output_dir / "val_scibert_pair.jsonl", [to_minimal_pair_row(row) for row in val_rows])

    metadata = {
        "setting": "cross_model_test",
        "train_source": train_source,
        "test_source": test_source,
        "fold": fold,
        "task": "Train SciBERT on one MLLM semantic dataset and validate on another MLLM output format",
        "label_definition": {
            "0": "match=true / citation is factually supported",
            "1": "match=false / citation is not factually supported",
        },
        "source_files": {
            "train": semantic_train_path.name,
            "val": validation_path.name,
        },
        "validation_fields_used_as_input": [
            "text_a = matching_results[].claim = 被引文献事实池",
            "text_b = matching_results[].originalkey_item.author_related_content = originalkey/施引文献 author_related_content",
            "text_b += matching_results[].originalkey_item.caption_annotations = originalkey/施引表格注释",
        ],
        "validation_fields_excluded_from_input": [
            "result",
            "explanation",
            "matching_results[].match_score",
            "matching_results[].explanation",
            "matching_results[].originalkey_item.description",
        ],
        "train_samples": len(train_rows),
        "val_samples": len(val_rows),
        "train_label_distribution": label_distribution(train_rows),
        "val_label_distribution": label_distribution(val_rows),
        "train_score_summary": score_summary(train_rows),
        "val_extraction_status": dict(Counter(row["extraction_status"] for row in val_rows)),
        "trainable_files": {
            "sentence_pair": {
                "train": "train_scibert_pair.jsonl",
                "val": "val_scibert_pair.jsonl",
                "required_columns": ["text_a", "text_b", "label"],
                "recommended_for_scibert": True,
            },
            "single_text": {
                "train": "train_scibert_text.jsonl",
                "val": "val_scibert_text.jsonl",
                "required_columns": ["text", "label"],
            },
            "full_with_metadata": {
                "train": "train.jsonl",
                "val": "val.jsonl",
            },
        },
        "val_avg_text_chars": round(sum(row["text_char_length"] for row in val_rows) / max(len(val_rows), 1), 2),
        "val_avg_cited_claims": round(sum(row["num_cited_claims"] for row in val_rows) / max(len(val_rows), 1), 2),
    }
    write_json(output_dir / "metadata.json", metadata)
    return metadata


def parse_args():
    parser = argparse.ArgumentParser(description="Build cross-model SciBERT datasets from configurable MLLM outputs.")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output-root", type=Path, default=Path.cwd() / "outputs" / "cross_model")
    parser.add_argument("--fold", type=int, default=None, help="Build one fold. If omitted, builds all folds 0-4.")
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--train-source", default="train_mllm")
    parser.add_argument("--test-source", default="validation_mllm")
    parser.add_argument(
        "--train-pattern",
        default="semantic_data/sft_semantic_match_{source}_fold{fold}.jsonl",
        help="Path template relative to --root. Supports {source} and {fold}.",
    )
    parser.add_argument(
        "--validation-pattern",
        default="{source}_fold{fold}_val.json",
        help="Path template relative to --root. Supports {source} and {fold}.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    folds = [args.fold] if args.fold is not None else list(range(args.num_folds))
    for fold in folds:
        output_dir = args.output_root / f"fold{fold}"
        metadata = build_dataset(
            args.root,
            output_dir,
            fold,
            args.train_source,
            args.test_source,
            args.train_pattern,
            args.validation_pattern,
        )
        print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
