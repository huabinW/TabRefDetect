import argparse
import json
import re
from collections import Counter
from pathlib import Path


def normalize_space(value):
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def read_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return rows


def parse_target(output_text):
    try:
        target = json.loads(output_text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", output_text, flags=re.S)
        if not match:
            raise
        target = json.loads(match.group(0))

    if "match" not in target:
        raise ValueError(f"Missing match field in output: {output_text[:200]}")
    match_value = target["match"]
    if isinstance(match_value, str):
        match_value = match_value.strip().lower()
        if match_value in {"true", "yes", "1"}:
            match_value = True
        elif match_value in {"false", "no", "0"}:
            match_value = False
        else:
            raise ValueError(f"Unsupported match value: {target['match']!r}")
    if not isinstance(match_value, bool):
        raise ValueError(f"Unsupported match value: {target['match']!r}")

    return {
        "match": match_value,
        "label": 0 if match_value else 1,
        "score": target.get("score"),
    }


def extract_between(text, start_marker, end_marker):
    pattern = re.escape(start_marker) + r"\s*(.*?)\s*" + re.escape(end_marker)
    match = re.search(pattern, text, flags=re.S)
    return normalize_space(match.group(1)) if match else ""


def extract_tagged_section(text, english_marker):
    pattern = rf"<[^<>]*{re.escape(english_marker)}[^<>]*>\s*(.*?)\s*</[^<>]*{re.escape(english_marker)}[^<>]*>"
    match = re.search(pattern, text, flags=re.S)
    return normalize_space(match.group(1)) if match else ""


def extract_caption_info(text):
    match = re.search(r"【表格注释信息】\s*(.*?)(?:\n\s*【|$)", text, flags=re.S)
    if not match:
        return ""
    return normalize_space(match.group(1))


def convert_row(row, source_file):
    input_text = row.get("input", "")
    target = parse_target(row.get("output", ""))

    cited_facts_text = extract_tagged_section(input_text, "Ground Truth")
    citing_author_related_content = extract_tagged_section(input_text, "Citation Sentence")
    citing_caption_text = extract_caption_info(input_text)

    if not cited_facts_text or not citing_author_related_content:
        compact_input = normalize_space(input_text)
        text = compact_input
        text_a = compact_input
        text_b = ""
        extraction_status = "fallback_raw_input"
    else:
        # Domain mapping:
        # text_a = cited-paper/table facts from the prompt Ground Truth section.
        # text_b = originalkey/citing-paper evidence: author_related_content plus citing-table caption.
        text_a = f"cited paper facts: {cited_facts_text}"
        text_b_parts = [f"originalkey author_related_content: {citing_author_related_content}"]
        if citing_caption_text:
            text_b_parts.append(f"originalkey table caption: {citing_caption_text}")
        text_b = "\n\n".join(text_b_parts)
        text = f"[CITED_FACTS]\n{text_a}\n\n[SEP]\n\n[ORIGINALKEY_CITING]\n{text_b}"
        extraction_status = "structured_sections"

    return {
        "id": normalize_space(row.get("id")),
        "label": target["label"],
        "text": text,
        "text_a": text_a,
        "text_b": text_b,
        "match": target["match"],
        "score": target["score"],
        "cited_facts_text": cited_facts_text,
        "citing_author_related_content": citing_author_related_content,
        "citing_caption_text": citing_caption_text,
        "source_file": source_file,
        "extraction_status": extraction_status,
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


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def label_distribution(rows):
    counts = Counter(row["label"] for row in rows)
    return {str(label): counts.get(label, 0) for label in [0, 1]}


def score_summary(rows):
    values = [row["score"] for row in rows if isinstance(row["score"], (int, float))]
    if not values:
        return {}
    return {
        "min": min(values),
        "max": max(values),
        "avg": round(sum(values) / len(values), 4),
    }


def build_dataset(semantic_dir, output_dir, model_source, fold):
    train_path = semantic_dir / f"sft_semantic_match_{model_source}_fold{fold}.jsonl"
    val_path = semantic_dir / f"sft_semantic_val_{model_source}_fold{fold}.jsonl"
    if not train_path.exists():
        raise FileNotFoundError(train_path)
    if not val_path.exists():
        raise FileNotFoundError(val_path)

    train_rows = [convert_row(row, train_path.name) for row in read_jsonl(train_path)]
    val_rows = [convert_row(row, val_path.name) for row in read_jsonl(val_path)]

    train_ids = {row["id"] for row in train_rows}
    val_ids = {row["id"] for row in val_rows}
    overlap = sorted(train_ids & val_ids)
    if overlap:
        raise ValueError(f"Train/val id overlap detected: {overlap[:10]}")

    write_jsonl(output_dir / "train.jsonl", train_rows)
    write_jsonl(output_dir / "val.jsonl", val_rows)
    write_json(output_dir / "train.json", train_rows)
    write_json(output_dir / "val.json", val_rows)

    write_jsonl(output_dir / "train_scibert_text.jsonl", [to_minimal_text_row(row) for row in train_rows])
    write_jsonl(output_dir / "val_scibert_text.jsonl", [to_minimal_text_row(row) for row in val_rows])
    write_jsonl(output_dir / "train_scibert_pair.jsonl", [to_minimal_pair_row(row) for row in train_rows])
    write_jsonl(output_dir / "val_scibert_pair.jsonl", [to_minimal_pair_row(row) for row in val_rows])

    metadata = {
        "source": "semantic_data",
        "model_source": model_source,
        "fold": fold,
        "task": "SciBERT binary classification for semantic matching",
        "label_definition": {
            "0": "match=true / citation is factually supported",
            "1": "match=false / citation is not factually supported",
        },
        "source_files": {
            "train": train_path.name,
            "val": val_path.name,
        },
        "fields_used_as_input": [
            "text_a = cited_facts_text = input.<原文事实 (Ground Truth)> = 被引文献事实",
            "text_b = citing_author_related_content = input.<待核查的引用句 (Citation Sentence)> = originalkey/施引文献 author_related_content",
            "text_b += citing_caption_text = input.【表格注释信息】 = originalkey/施引表格注释",
        ],
        "fields_excluded_from_input": [
            "instruction",
            "output.explanation",
            "output.score",
        ],
        "train_samples": len(train_rows),
        "val_samples": len(val_rows),
        "trainable_files": {
            "single_text": {
                "train": "train_scibert_text.jsonl",
                "val": "val_scibert_text.jsonl",
                "required_columns": ["text", "label"],
            },
            "sentence_pair": {
                "train": "train_scibert_pair.jsonl",
                "val": "val_scibert_pair.jsonl",
                "required_columns": ["text_a", "text_b", "label"],
                "recommended_for_scibert": True,
            },
            "full_with_metadata": {
                "train": "train.jsonl",
                "val": "val.jsonl",
            },
        },
        "train_label_distribution": label_distribution(train_rows),
        "val_label_distribution": label_distribution(val_rows),
        "train_score_summary": score_summary(train_rows),
        "val_score_summary": score_summary(val_rows),
        "train_extraction_status": dict(Counter(row["extraction_status"] for row in train_rows)),
        "val_extraction_status": dict(Counter(row["extraction_status"] for row in val_rows)),
        "train_avg_text_chars": round(sum(row["text_char_length"] for row in train_rows) / max(len(train_rows), 1), 2),
        "val_avg_text_chars": round(sum(row["text_char_length"] for row in val_rows) / max(len(val_rows), 1), 2),
    }
    write_json(output_dir / "metadata.json", metadata)
    return metadata


def parse_args():
    parser = argparse.ArgumentParser(description="Convert semantic SFT JSONL data into SciBERT classification data.")
    parser.add_argument(
        "--semantic-dir",
        type=Path,
        default=Path.cwd() / "semantic_data",
        help="Directory containing sft_semantic_match_* and sft_semantic_val_* JSONL files.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--model-source", default="mllm")
    parser.add_argument("--fold", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = Path.cwd() / "scibert_datasets" / "semantic_data" / args.model_source / f"fold{args.fold}"
    metadata = build_dataset(args.semantic_dir, output_dir, args.model_source, args.fold)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
