import argparse
import json
import logging
from pathlib import Path

import ocr_glm46vflash_reintegrate as base


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


DEFAULT_OCR_FILE = Path.cwd() / "data" / "full_results.json"
DEFAULT_FALLBACK_OCR_FILE = None
DEFAULT_OUTPUT_DIR = Path.cwd() / "outputs" / "direct_author_content"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run OCR + GLM4.6V-Flash pairwise checking by directly using the existing "
            "originalkey author_related_content field without extraction preprocessing."
        )
    )
    parser.add_argument("--ocr-file", type=Path, default=DEFAULT_OCR_FILE)
    parser.add_argument(
        "--fallback-ocr-file",
        type=Path,
        default=DEFAULT_FALLBACK_OCR_FILE,
        help="Used only when originalkey_analysis is missing from the primary record.",
    )
    parser.add_argument(
        "--input-prompts",
        type=Path,
        default=None,
        help="Run pairwise model calls from a previously generated direct prompt JSONL.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-model", action="store_true")
    parser.add_argument("--api-base", default="http://localhost:8000/v1")
    parser.add_argument("--api-key", default="your-api-key")
    parser.add_argument("--model", default="glm-4.6v")
    parser.add_argument("--max-tokens", type=int, default=8000)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all samples.")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--max-citekey-chars", type=int, default=0, help="0 means no truncation.")
    parser.add_argument("--match-threshold", type=float, default=0.5)
    parser.add_argument("--resume", action="store_true")
    parser.set_defaults(prompt_kind="pairwise")
    return parser.parse_args()


def direct_field_text(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return json.dumps(value, ensure_ascii=False)


def get_original_analysis(item, fallback_by_id):
    original = base.get_first_original_analysis(item)
    fallback_used = False
    if original is None:
        sample_id = item.get("id") or item.get("paper_id")
        original = base.get_first_original_analysis(fallback_by_id.get(sample_id) or {})
        fallback_used = original is not None
    return original, fallback_used


def build_direct_prompt_row(item, fallback_by_id, max_citekey_chars):
    original, fallback_used = get_original_analysis(item, fallback_by_id)
    payload = (original or {}).get("analysis") or {}
    author_related_content = direct_field_text(payload.get("author_related_content"))
    caption = direct_field_text(payload.get("caption") or payload.get("caption_annotations"))
    cited_facts_text = base.join_citekey_facts(
        item.get("citekey_analysis"),
        max_citekey_chars,
    )

    sample_id = item.get("id") or item.get("paper_id")
    tab_id = (original or {}).get("tab_id", "")
    system_prompt = base.build_system_prompt("pairwise")

    return {
        "id": sample_id,
        "paper_id": item.get("paper_id") or sample_id,
        "citekey": item.get("citekey"),
        "originalkey": item.get("originalkey"),
        "author": item.get("author"),
        "label": item.get("label"),
        "charts": item.get("charts") or [],
        "processing_status": item.get("processing_status"),
        "error_message": item.get("error_message") or "",
        "task": "pairwise_citekey_originalkey_direct_author_content",
        "prompt_kind": "pairwise",
        "system_prompt": system_prompt,
        "prompt_inputs": {
            "tab_id": tab_id,
            "caption": caption,
            "cited_facts_text": cited_facts_text,
            "target_claims_text": author_related_content,
            "direct_author_related_content": author_related_content,
            "author_related_content_source": (
                "originalkey_analysis[].analysis.author_related_content"
            ),
            "source_quality": {
                "processing_status": item.get("processing_status"),
                "used_fallback_originalkey_analysis": fallback_used,
                "direct_author_related_content_nonempty": bool(author_related_content),
                "num_citekey_evidences": len(
                    base.split_citekey_evidences(cited_facts_text)
                ),
                "error_message": item.get("error_message") or "",
            },
        },
    }


def generate_direct_prompts(args):
    data = base.load_json_list(args.ocr_file)

    fallback_data = []
    if (
        args.fallback_ocr_file
        and str(args.fallback_ocr_file) not in {"", "."}
        and args.fallback_ocr_file.exists()
    ):
        fallback_data = base.load_json_list(args.fallback_ocr_file)
    fallback_by_id = {
        item.get("id") or item.get("paper_id"): item for item in fallback_data
    }

    selected = data[args.start_index :]
    if args.max_samples > 0:
        selected = selected[: args.max_samples]

    rows = [
        build_direct_prompt_row(
            item,
            fallback_by_id=fallback_by_id,
            max_citekey_chars=args.max_citekey_chars,
        )
        for item in selected
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prompt_jsonl = args.output_dir / "ocr_glm46vflash_direct_author_content_prompts.jsonl"
    prompt_json = args.output_dir / "ocr_glm46vflash_direct_author_content_prompts.json"
    base.write_jsonl(prompt_jsonl, rows)
    base.write_json(prompt_json, rows)

    empty_ids = [
        row.get("id")
        for row in rows
        if not row.get("prompt_inputs", {}).get("target_claims_text")
    ]
    no_evidence_ids = [
        row.get("id")
        for row in rows
        if not row.get("prompt_inputs", {}).get("cited_facts_text")
    ]
    summary = {
        "workflow": "direct_existing_author_related_content",
        "ocr_file": str(args.ocr_file),
        "fallback_ocr_file": str(args.fallback_ocr_file),
        "total_source_records": len(data),
        "selected_records": len(rows),
        "records_with_author_related_content": len(rows) - len(empty_ids),
        "records_without_author_related_content": len(empty_ids),
        "records_without_citekey_evidence": len(no_evidence_ids),
        "empty_author_related_content_ids": empty_ids,
        "no_citekey_evidence_ids": no_evidence_ids,
        "prompt_jsonl": str(prompt_jsonl),
        "prompt_json": str(prompt_json),
        "model_preprocessing_used": False,
    }
    base.write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return prompt_jsonl


def main():
    args = parse_args()

    if args.input_prompts is not None:
        base.run_existing_prompts(args)
        return

    prompt_jsonl = generate_direct_prompts(args)
    if args.run_model:
        args.input_prompts = prompt_jsonl
        base.run_existing_prompts(args)


if __name__ == "__main__":
    main()
