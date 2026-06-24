from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from materialize_popo_codex_precision_review_results import (
    DEFAULT_CAPTION_RESOLUTION,
    apply_caption_resolution,
    load_caption_resolution,
)


BASE = Path(__file__).resolve().parent
DEFAULT_INPUT = (
    BASE
    / "batch_table_text_tree"
    / "mineru_popo_comparison"
    / "popo_human_child_annotations_strict"
    / "batch_popo_strict_human_child_annotation_template.json"
)
DEFAULT_OUTPUT_DIR = (
    BASE
    / "batch_table_text_tree"
    / "mineru_popo_comparison"
    / "popo_codex_precision_v2"
    / "review_packages"
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_document_package(
    source: dict[str, Any], document: dict[str, Any]
) -> dict[str, Any]:
    tables = []
    item_count = 0
    for table in document["tables"]:
        packaged_table = {
            "table_label": table["table_label"],
            "table_anchor": copy.deepcopy(table["table_anchor"]),
            "popo_table_node": copy.deepcopy(table.get("popo_table_node")),
            "parents": [],
        }
        for parent in table["parents"]:
            packaged_parent = {
                "candidate_id": parent["candidate_id"],
                "parent_node_id": parent.get("parent_node_id"),
                "parent_title": parent.get("parent_title"),
                "parent_ancestor_titles": copy.deepcopy(
                    parent.get("parent_ancestor_titles") or []
                ),
                "parent_page_indices": copy.deepcopy(
                    parent.get("parent_page_indices") or []
                ),
                "parent_raw_content_indices": copy.deepcopy(
                    parent.get("parent_raw_content_indices") or []
                ),
                "full_parent_text": parent["full_parent_text"],
                "candidate_signals": copy.deepcopy(
                    parent.get("candidate_signals") or {}
                ),
                "review_items": [],
            }
            for child in parent["children"]:
                packaged_parent["review_items"].append(
                    {
                        "review_key": child["review_key"],
                        "child_id": child["child_id"],
                        "child_text": child["child_text"],
                        "child_text_sha256": child["child_text_sha256"],
                        "char_start": child["char_start"],
                        "char_end": child["char_end"],
                        "code_selection_score": child.get("code_selection_score"),
                        "code_selection_signals": copy.deepcopy(
                            child.get("code_selection_signals") or {}
                        ),
                        "existing_human_label": child.get("human_label"),
                        "existing_human_rationale": child.get("human_rationale"),
                        "existing_citation_support": child.get("citation_support"),
                    }
                )
                item_count += 1
            packaged_table["parents"].append(packaged_parent)
        tables.append(packaged_table)

    return {
        "schema_version": "2.0",
        "review_stage": "codex_table_scope_precision_review_v2",
        "source_workflow": "mineru_popo_strict_high_recall",
        "source_annotation_file": str(DEFAULT_INPUT),
        "slug": document["slug"],
        "review_item_count": item_count,
        "label_semantics": {
            "0": (
                "retain: the child supplies table-scoped information that a later "
                "citation verifier may need to interpret or validate this table"
            ),
            "1": (
                "demote: the child is a table pointer, generic description, "
                "off-scope background, another table's evidence, or otherwise "
                "unlikely to be information for which this table would be cited"
            ),
        },
        "decision_policy": [
            "Read the table caption/body, complete Popo parent text, and exact child text.",
            "Use label 0 only for supplementary information not already visible in the table that is needed to interpret, reproduce, or verify it.",
            "Use label 0 for table-scoped datasets, splits, prompts, shots, metrics, evaluation protocols, baseline definitions, configurations, training settings, constraints, and necessary limitations or cross-study qualifications.",
            "Use label 1 for text that only says the table shows, summarizes, reports, lists, or compares something already visible.",
            "Use label 1 for model innovations, method background, implementation details, or general performance claims that are topically related but are not part of the table's intended content.",
            "Use label 1 when the text belongs to another table, figure, experiment, dataset, model variant, or evaluation scope.",
            "Use label 1 for direct result restatements, trend summaries, and promotional efficacy or capability claims derived from values already visible in the table.",
            "A result interpretation is label 0 only when it supplies a necessary limitation, metric behavior, dataset difference, or cross-study comparability qualification that is not visible in the table.",
            "A relevant parent does not make every child relevant. Judge the exact child span.",
            "A table may legitimately retain zero child spans when its body and caption are self-contained.",
            "Do not use code_selection_score as the decision; it is only a high-recall ranking signal.",
            "Existing human labels are gold constraints and must be copied exactly.",
        ],
        "human_feedback_examples": [
            {
                "principle": "pure_table_description",
                "label": 1,
                "reason": "A sentence that only says the main parameters are summarized in the table adds no citation-verification evidence.",
            },
            {
                "principle": "related_innovation_but_not_table_scope",
                "label": 1,
                "reason": "A model innovation may supplement general model knowledge but should be excluded when the table only presents model names or parameters and would not be cited for that innovation.",
            },
            {
                "principle": "not_the_table_intended_content",
                "label": 1,
                "reason": "Topical proximity is insufficient when the child does not express what the table is intended to establish.",
            },
            {
                "principle": "result_restatement_is_not_supplementary",
                "label": 1,
                "reason": "A sentence that repeats a gain, gap, trend, or best score already visible in the table is descriptive rather than supplementary.",
            },
        ],
        "allowed_semantic_roles": [
            "experimental_condition",
            "dataset",
            "model",
            "metric",
            "training_setting",
            "method",
            "result_interpretation",
            "comparison",
            "limitation",
            "other_support",
            "irrelevant",
        ],
        "allowed_citation_support": ["direct", "indirect", "none"],
        "tables": tables,
    }


def render_markdown(package: dict[str, Any]) -> str:
    lines = [
        f"# Codex Table-Scope Precision Review: {package['slug']}",
        "",
        f"- Review items: {package['review_item_count']}",
        "- Code stage: high recall only",
        "- Semantic stage: Codex precision review",
        "",
    ]
    for table in package["tables"]:
        anchor = table["table_anchor"]
        lines.extend(
            [
                f"## {table['table_label']}",
                "",
                f"Caption: {anchor.get('caption') or ''}",
                "",
                f"Table body: {anchor.get('table_html') or anchor.get('table_code_body') or ''}",
                "",
            ]
        )
        for parent in table["parents"]:
            lines.extend(
                [
                    f"### {parent['candidate_id']}",
                    "",
                    f"Section: {parent.get('parent_title') or ''}",
                    "",
                    f"Parent text: {parent['full_parent_text']}",
                    "",
                ]
            )
            for child in parent["review_items"]:
                lines.extend(
                    [
                        f"- Review key: `{child['review_key']}`",
                        f"  - Code score: {child.get('code_selection_score')}",
                        f"  - Existing human label: {child.get('existing_human_label')}",
                        f"  - Existing human rationale: {child.get('existing_human_rationale') or ''}",
                        f"  - Child: {child['child_text']}",
                        "",
                    ]
                )
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare Popo strict high-recall candidates for Codex precision review."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--caption-resolution",
        type=Path,
        default=DEFAULT_CAPTION_RESOLUTION,
        help="Optional caption-resolution JSON used to preserve table captions.",
    )
    args = parser.parse_args()

    source = load_json(args.input)
    source = apply_caption_resolution(
        source,
        load_caption_resolution(args.caption_resolution),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    documents = []
    total_items = 0
    for document in source["documents"]:
        package = build_document_package(source, document)
        slug = document["slug"]
        package_path = args.output_dir / f"{slug}_codex_precision_v2_package.json"
        markdown_path = args.output_dir / f"{slug}_codex_precision_v2_package.md"
        decision_path = args.output_dir / f"{slug}_codex_precision_v2_decisions.json"
        write_json(package_path, package)
        markdown_path.write_text(render_markdown(package), encoding="utf-8")
        documents.append(
            {
                "slug": slug,
                "review_items": package["review_item_count"],
                "package_json": str(package_path),
                "package_markdown": str(markdown_path),
                "expected_decisions": str(decision_path),
            }
        )
        total_items += package["review_item_count"]

    if total_items != source["candidate_child_count"]:
        raise ValueError(
            f"Prepared {total_items} items, expected {source['candidate_child_count']}"
        )
    status = {
        "schema_version": "2.0",
        "review_stage": "codex_table_scope_precision_review_v2",
        "source": str(args.input),
        "caption_resolution_input": str(args.caption_resolution),
        "source_candidate_children": source["candidate_child_count"],
        "prepared_review_items": total_items,
        "documents": documents,
    }
    status_path = args.output_dir / "codex_precision_v2_package_status.json"
    write_json(status_path, status)
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
