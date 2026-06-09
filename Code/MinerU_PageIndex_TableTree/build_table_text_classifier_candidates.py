import argparse
import json
import re
from pathlib import Path

from build_table_text_tree_batch import DEFAULT_MANIFEST, load_json


DEFAULT_ANNOTATIONS = None

SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])(?=\s+[\"'(\[]*[A-Z0-9])")
SECTION_PRIORITY_RE = re.compile(
    r"\b(?:experiment(?:al)?|setup|method|dataset|corpus|evaluation|results?|discussion|"
    r"analysis|ablation|implementation|training|benchmark|performance|appendix|hyperparameter)\b",
    re.I,
)
CONDITION_RE = re.compile(
    r"\b(?:dataset|corpus|model|metric|benchmark|prompt|shot|split|baseline|optimizer|"
    r"learning rate|batch size|epoch|step|hyperparameter|accuracy|precision|recall|"
    r"f[- ]?score|f1|bleu|pass@1|hardware|gpu|pretrain|finetun|training|evaluation)\b",
    re.I,
)
AUTHOR_RE = re.compile(r"\b(?:university|department|institute|@[\w.-]+|correspondence)\b", re.I)
LICENSE_RE = re.compile(r"\b(?:creative commons|apache|mit license|license:|code release)\b", re.I)
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{3,}")
STOPWORDS = {
    "about", "after", "also", "among", "based", "before", "between", "from", "have",
    "into", "more", "most", "only", "other", "over", "show", "shown", "table", "than",
    "that", "their", "these", "this", "those", "through", "using", "were", "which", "with",
}


def split_long_span(text, start, end, target=480):
    spans = []
    cursor = start
    while end - cursor > target:
        lower = cursor + int(target * 0.65)
        upper = min(end, cursor + int(target * 1.25))
        window = text[lower:upper]
        matches = list(re.finditer(r"[;,](?:\s+|$)", window))
        if not matches:
            break
        split_at = lower + matches[-1].end()
        spans.append((cursor, split_at))
        cursor = split_at
    spans.append((cursor, end))
    return spans


def split_child_spans(text):
    boundaries = [0]
    boundaries.extend(match.end() for match in SENTENCE_BOUNDARY_RE.finditer(text))
    boundaries.append(len(text))
    spans = []
    for left, right in zip(boundaries, boundaries[1:]):
        if right <= left:
            continue
        spans.extend(split_long_span(text, left, right))
    return [(left, right) for left, right in spans if text[left:right].strip()]


def table_reference_pattern(label):
    suffix = re.sub(r"(?i)^table\s*", "", label).strip()
    return re.compile(rf"\b(?:Table|Tab\.)\s*{re.escape(suffix)}\b", re.I)


def exclusion_reason(paragraph):
    text = paragraph["text"]
    if paragraph["page_start"] == 1 and len(text) < 600 and AUTHOR_RE.search(text):
        return "front_page_author_or_affiliation"
    if len(text) < 500 and LICENSE_RE.search(text):
        return "license_or_release_metadata"
    return None


def lexical_tokens(text):
    return {
        token.lower()
        for token in TOKEN_RE.findall(re.sub(r"<[^>]+>", " ", text or ""))
        if token.lower() not in STOPWORDS
    }


def candidate_signals(table, paragraph):
    label = table["canonical_label"]
    table_page = table["page_index"]
    table_raw_index = table["raw_content_index"]
    page_distance = min(abs(page - table_page) for page in paragraph["page_indices"])
    raw_distance = min(abs(index - table_raw_index) for index in paragraph["raw_content_indices"])
    exact_reference = bool(table_reference_pattern(label).search(paragraph["text"]))
    same_section = paragraph["parent_node_id"] == table["parent_node_id"]
    section_priority = bool(SECTION_PRIORITY_RE.search(paragraph["parent_title"] or ""))
    condition_rich = bool(CONDITION_RE.search(paragraph["text"]))
    close_reading_order = raw_distance <= 8
    table_tokens = lexical_tokens(
        " ".join(
            value or ""
            for value in (
                table.get("caption"),
                table.get("table_html"),
                table.get("table_code_body"),
            )
        )
    )
    paragraph_tokens = lexical_tokens(
        f"{paragraph.get('parent_title') or ''} {paragraph['text']}"
    )
    lexical_overlap = sorted(table_tokens & paragraph_tokens)
    lexical_overlap_score = min(30, len(lexical_overlap) * 5)

    score = 0
    score += 100 if exact_reference else 0
    score += 40 if same_section else 0
    score += 35 if page_distance == 0 else 25 if page_distance == 1 else 15 if page_distance == 2 else 0
    score += 20 if close_reading_order else 0
    score += 20 if section_priority else 0
    score += 10 if condition_rich else 0
    score += lexical_overlap_score

    if exact_reference:
        tier = "tier_1_explicit_reference"
    elif same_section or page_distance <= 1 or close_reading_order:
        tier = "tier_2_local_context"
    elif section_priority or condition_rich or lexical_overlap:
        tier = "tier_3_experiment_condition"
    else:
        tier = "tier_4_background"

    return {
        "priority_score": score,
        "priority_tier": tier,
        "exact_table_reference": exact_reference,
        "same_tree_section": same_section,
        "page_distance": page_distance,
        "raw_content_distance": raw_distance,
        "close_reading_order": close_reading_order,
        "experiment_section": section_priority,
        "condition_rich_text": condition_rich,
        "table_text_lexical_overlap": lexical_overlap,
        "table_text_lexical_overlap_score": lexical_overlap_score,
    }


def build_document(document_annotation):
    tree_path = Path(document_annotation["paragraph_tree_source"])
    tree_payload = load_json(tree_path)
    paragraphs = tree_payload["paragraph_corpus"]
    positives = {
        table["table_anchor"]["canonical_label"]: {
            evidence["parent_paragraph_id"]
            for evidence in table["manual_body_text_evidence"]
        }
        for table in document_annotation["tables"]
    }

    output_tables = []
    for annotated_table in document_annotation["tables"]:
        table = annotated_table["table_anchor"]
        label = table["canonical_label"]
        candidates = []
        for paragraph in paragraphs:
            exclusion = exclusion_reason(paragraph)
            signals = candidate_signals(table, paragraph)
            if exclusion:
                signals["priority_tier"] = "excluded_non_body"
                signals["priority_score"] = -1

            parent_label = 0 if paragraph["node_id"] in positives[label] else 1
            children = []
            for child_index, (start, end) in enumerate(split_child_spans(paragraph["text"]), start=1):
                children.append(
                    {
                        "child_id": f"{paragraph['node_id']}-child-{child_index:03d}",
                        "char_start": start,
                        "char_end": end,
                        "child_text": paragraph["text"][start:end],
                        "child_label": None,
                    }
                )

            candidates.append(
                {
                    "candidate_id": f"{label.lower().replace(' ', '-')}-{paragraph['node_id']}",
                    "eligible": exclusion is None,
                    "exclusion_reason": exclusion,
                    "manual_parent_label": parent_label,
                    "label_scope": "parent_paragraph",
                    "parent_paragraph_id": paragraph["node_id"],
                    "full_parent_text": paragraph["text"],
                    "raw_content_indices": paragraph["raw_content_indices"],
                    "page_indices": paragraph["page_indices"],
                    "bboxes": paragraph["bboxes"],
                    "parent_node_id": paragraph["parent_node_id"],
                    "parent_title": paragraph["parent_title"],
                    "candidate_signals": signals,
                    "children": children,
                }
            )

        candidates.sort(
            key=lambda item: (
                -item["candidate_signals"]["priority_score"],
                item["page_indices"][0],
                item["raw_content_indices"][0],
            )
        )
        output_tables.append(
            {
                "table_anchor": table,
                "candidate_count": len(candidates),
                "positive_parent_count": len(positives[label]),
                "candidates": candidates,
            }
        )

    return {
        "slug": document_annotation["slug"],
        "source_pdf": document_annotation["source_pdf"],
        "candidate_policy": {
            "recall": "All local paragraph-tree parent paragraphs are retained.",
            "ranking": "Rule-based metadata only; manual labels do not affect priority.",
            "label_semantics": {
                "0": "correct: the parent paragraph supplies evidence for the table",
                "1": "incorrect: the parent paragraph does not supply evidence for the table",
            },
            "child_labels": "null until a separate child-level annotation pass.",
        },
        "table_count": len(output_tables),
        "tables": output_tables,
    }


def audit_dataset(dataset):
    failures = []
    for document in dataset["documents"]:
        for table in document["tables"]:
            positives = sum(
                candidate["manual_parent_label"] == 0 for candidate in table["candidates"]
            )
            if positives != table["positive_parent_count"]:
                failures.append(
                    f"{document['slug']} {table['table_anchor']['canonical_label']}: "
                    f"{positives} positives != {table['positive_parent_count']}"
                )
            for candidate in table["candidates"]:
                parent = candidate["full_parent_text"]
                for child in candidate["children"]:
                    if parent[child["char_start"]:child["char_end"]] != child["child_text"]:
                        failures.append(f"{candidate['candidate_id']}: child offsets changed text")
    return failures


def main():
    parser = argparse.ArgumentParser(description="Build table-parent-child classifier candidates.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    args = parser.parse_args()

    manifest = load_json(args.manifest)
    annotations_path = args.annotations or (
        Path(manifest["output_root"])
        / "manual_table_body_text_annotations"
        / "batch_manual_table_body_text_annotations.json"
    )
    annotations = load_json(annotations_path)
    documents = [build_document(document) for document in annotations["documents"]]
    dataset = {
        "dataset_version": "1.1",
        "label_semantics": {
            "manual_parent_label": {
                "0": "correct/relevant table-text relation",
                "1": "incorrect/irrelevant table-text relation",
            },
            "child_label": None,
        },
        "label_source": str(annotations_path),
        "document_count": len(documents),
        "table_count": sum(document["table_count"] for document in documents),
        "documents": documents,
    }
    failures = audit_dataset(dataset)
    if failures:
        raise ValueError("\n".join(failures))

    output_dir = Path(manifest["output_root"]) / "classifier_candidate_dataset"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "batch_table_text_classifier_candidates.json"
    output_path.write_text(json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8")
    jsonl_path = output_dir / "batch_table_text_classifier_parent_child_samples.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for document in documents:
            for table in document["tables"]:
                for candidate in table["candidates"]:
                    for child in candidate["children"]:
                        sample = {
                            "slug": document["slug"],
                            "table": table["table_anchor"],
                            "candidate_id": candidate["candidate_id"],
                            "eligible": candidate["eligible"],
                            "manual_parent_label": candidate["manual_parent_label"],
                            "child_label": child["child_label"],
                            "parent": {
                                "parent_paragraph_id": candidate["parent_paragraph_id"],
                                "full_parent_text": candidate["full_parent_text"],
                                "raw_content_indices": candidate["raw_content_indices"],
                                "page_indices": candidate["page_indices"],
                                "bboxes": candidate["bboxes"],
                                "parent_node_id": candidate["parent_node_id"],
                                "parent_title": candidate["parent_title"],
                            },
                            "child": child,
                            "candidate_signals": candidate["candidate_signals"],
                        }
                        handle.write(json.dumps(sample, ensure_ascii=False) + "\n")

    summary = {
        "output": str(output_path),
        "jsonl_output": str(jsonl_path),
        "documents": dataset["document_count"],
        "tables": dataset["table_count"],
        "parent_candidates": sum(
            table["candidate_count"]
            for document in documents
            for table in document["tables"]
        ),
        "positive_parent_relations": sum(
            table["positive_parent_count"]
            for document in documents
            for table in document["tables"]
        ),
        "child_candidates": sum(
            len(candidate["children"])
            for document in documents
            for table in document["tables"]
            for candidate in table["candidates"]
        ),
        "audit_failures": failures,
    }
    (output_dir / "batch_table_text_classifier_candidates_status.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
