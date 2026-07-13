from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

try:
    from .prepare_scope_review_packages import (
        ALLOWED_EVIDENCE_ROLES,
        DEFAULT_INPUT,
        DEFAULT_RUN_DIR,
        REVIEW_STAGE,
        build_document_package,
        load_json,
        paper_id,
        write_json,
    )
except ImportError:  # pragma: no cover - supports direct script execution
    from prepare_scope_review_packages import (
        ALLOWED_EVIDENCE_ROLES,
        DEFAULT_INPUT,
        DEFAULT_RUN_DIR,
        REVIEW_STAGE,
        build_document_package,
        load_json,
        paper_id,
        write_json,
    )


DEFAULT_DECISION_DIR = DEFAULT_RUN_DIR / "scope_review_packages"
DEFAULT_OUTPUT_DIR = DEFAULT_RUN_DIR / "scope_review_results"
DEFAULT_DECISION_GLOB = "*_scope_review_decisions.json"

DECISION_SCHEMA_VERSION = "scope-review-decisions-v2"
FULL_AUDIT_SCHEMA_VERSION = "scope-review-full-audit-v2"
SELECTED_SCHEMA_VERSION = "scope-review-selected-relations-v2"
HUMAN_SCHEMA_VERSION = "scope-review-human-template-v2"


def _is_confidence(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and 0 <= value <= 1


def validate_decision_payload(
    package: dict[str, Any],
    payload: dict[str, Any],
    source_name: str = "<decision payload>",
) -> dict[str, dict[str, Any]]:
    """Validate complete one-decision-per-child coverage for one paper."""
    failures: list[str] = []
    payload_paper_id = payload.get("paper_id") or payload.get("slug")
    if payload_paper_id != package["paper_id"]:
        failures.append(
            f"paper_id mismatch: expected {package['paper_id']!r}, got {payload_paper_id!r}"
        )
    if payload.get("review_stage") != REVIEW_STAGE:
        failures.append(f"review_stage must be {REVIEW_STAGE!r}")
    if payload.get("schema_version") != DECISION_SCHEMA_VERSION:
        failures.append(f"schema_version must be {DECISION_SCHEMA_VERSION!r}")
    decisions = payload.get("decisions")
    if not isinstance(decisions, list):
        raise ValueError(f"{source_name}: decisions must be a list")

    expected = {item["child_id"]: item for item in package["review_items"]}
    known_tables = set(package["review_eligible_table_ids"])
    normalized: dict[str, dict[str, Any]] = {}
    for index, decision in enumerate(decisions):
        prefix = f"decision[{index}]"
        if not isinstance(decision, dict):
            failures.append(f"{prefix}: must be an object")
            continue
        child_id = decision.get("child_id")
        if child_id not in expected:
            failures.append(f"{prefix}: unknown child_id {child_id!r}")
            continue
        if child_id in normalized:
            failures.append(f"{prefix}: duplicate child_id {child_id!r}")
            continue
        item = expected[child_id]
        if decision.get("parent_id") != item["parent_id"]:
            failures.append(
                f"{prefix} {child_id}: parent_id must be {item['parent_id']!r}"
            )
        relations = decision.get("relevant_tables")
        if not isinstance(relations, list):
            failures.append(f"{prefix} {child_id}: relevant_tables must be a list")
            continue

        seen_tables: set[str] = set()
        clean_relations: list[dict[str, Any]] = []
        for relation_index, relation in enumerate(relations):
            relation_prefix = f"{prefix} {child_id} relevant_tables[{relation_index}]"
            if not isinstance(relation, dict):
                failures.append(f"{relation_prefix}: must be an object")
                continue
            table_id = relation.get("table_id")
            if table_id not in known_tables:
                failures.append(f"{relation_prefix}: unknown table_id {table_id!r}")
            elif table_id in seen_tables:
                failures.append(f"{relation_prefix}: duplicate table_id {table_id!r}")
            else:
                seen_tables.add(table_id)
            if relation.get("label") != 0:
                failures.append(f"{relation_prefix}: retained relation label must be 0")
            if not _is_confidence(relation.get("confidence")):
                failures.append(f"{relation_prefix}: confidence must be in [0, 1]")
            role = relation.get("evidence_role")
            if role not in ALLOWED_EVIDENCE_ROLES:
                failures.append(
                    f"{relation_prefix}: evidence_role must be one of {ALLOWED_EVIDENCE_ROLES}"
                )
            rationale = relation.get("rationale")
            if not isinstance(rationale, str) or not rationale.strip():
                failures.append(f"{relation_prefix}: rationale is required")
            clean_relations.append(copy.deepcopy(relation))

        rejection_reason = decision.get("rejection_reason")
        if not relations and (not isinstance(rejection_reason, str) or not rejection_reason.strip()):
            failures.append(f"{prefix} {child_id}: rejection_reason is required when no table is retained")

        normalized[child_id] = {
            "child_id": child_id,
            "parent_id": decision.get("parent_id"),
            "relevant_tables": clean_relations,
            "rejection_reason": rejection_reason,
        }

    missing = sorted(set(expected) - set(normalized))
    if missing:
        failures.append(f"missing decisions for {len(missing)} children: {missing[:5]}")
    if failures:
        raise ValueError(f"{source_name}:\n" + "\n".join(failures))
    return normalized


def load_decision_payloads(
    decision_dir: Path,
    expected_paper_ids: set[str],
    decision_glob: str = DEFAULT_DECISION_GLOB,
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    paths = sorted(decision_dir.glob(decision_glob))
    if not paths:
        raise ValueError(
            f"no decision files matching {decision_glob!r} in {decision_dir}"
        )
    payloads: dict[str, dict[str, Any]] = {}
    sources: dict[str, str] = {}
    failures: list[str] = []
    for path in paths:
        payload = load_json(path)
        if not isinstance(payload, dict):
            failures.append(f"{path}: decision file must contain an object")
            continue
        doc_id = payload.get("paper_id") or payload.get("slug")
        if not isinstance(doc_id, str) or not doc_id:
            failures.append(f"{path}: missing paper_id")
            continue
        if doc_id not in expected_paper_ids:
            failures.append(f"{path}: unknown paper_id {doc_id!r}")
            continue
        if doc_id in payloads:
            failures.append(f"{path}: duplicate decision file for {doc_id!r}")
            continue
        payloads[doc_id] = payload
        sources[doc_id] = str(path)
    missing = sorted(expected_paper_ids - set(payloads))
    if missing:
        failures.append(f"missing decision files for papers: {missing}")
    if failures:
        raise ValueError("\n".join(failures))
    return payloads, sources


def _table_summary(table: dict[str, Any], selected_count: int) -> dict[str, Any]:
    return {
        "table_id": table["table_id"],
        "table_label": table["table_label"],
        "table_caption": table["table_caption"],
        "page": table.get("page"),
        "bbox": copy.deepcopy(table.get("bbox")),
        "selected_relation_count": selected_count,
    }


def _relation_row(
    package: dict[str, Any],
    table: dict[str, Any],
    item: dict[str, Any],
    relation: dict[str, Any],
    decision_source: str | None,
) -> dict[str, Any]:
    return {
        "paper_id": package["paper_id"],
        "table_id": table["table_id"],
        "table_label": table["table_label"],
        "table_caption": table["table_caption"],
        "table_body": copy.deepcopy(table["table_body"]),
        "parent_id": item["parent_id"],
        "full_parent_text": item["full_parent_text"],
        "full_parent_text_sha256": item["full_parent_text_sha256"],
        "child_id": item["child_id"],
        "char_start": item["char_start"],
        "char_end": item["char_end"],
        "child_text": item["child_text"],
        "child_text_sha256": item["child_text_sha256"],
        "page": copy.deepcopy(item.get("page")),
        "bbox": copy.deepcopy(item.get("bbox")),
        "source": copy.deepcopy(item.get("source")),
        "scope_ids": copy.deepcopy(item["scope_ids"]),
        "table_suggestions": copy.deepcopy(item["table_suggestions"]),
        "subagent_label": 0,
        "subagent_confidence": relation["confidence"],
        "subagent_role": relation["evidence_role"],
        "subagent_rationale": relation["rationale"],
        "decision_source_file": decision_source,
    }


def _human_row(relation: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "paper_id",
        "table_id",
        "table_label",
        "table_caption",
        "table_body",
        "parent_id",
        "full_parent_text",
        "full_parent_text_sha256",
        "child_id",
        "char_start",
        "char_end",
        "child_text",
        "child_text_sha256",
        "page",
        "bbox",
        "source",
        "subagent_confidence",
        "subagent_role",
        "subagent_rationale",
    ]
    row = {key: copy.deepcopy(relation.get(key)) for key in keys}
    row["human_label"] = None
    row["human_rationale"] = ""
    return row


def materialize_scope_reviews(
    inventory: dict[str, Any],
    decision_payloads: Mapping[str, dict[str, Any]],
    decision_sources: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    documents = inventory.get("documents") if isinstance(inventory, dict) else None
    if not isinstance(documents, list):
        raise ValueError("inventory must contain a documents list")
    decision_sources = decision_sources or {}

    packages: list[dict[str, Any]] = []
    expected_papers: set[str] = set()
    for document in documents:
        package = build_document_package(inventory, document)
        if package["paper_id"] in expected_papers:
            raise ValueError(f"duplicate paper_id {package['paper_id']!r}")
        expected_papers.add(package["paper_id"])
        packages.append(package)
    unknown_papers = sorted(set(decision_payloads) - expected_papers)
    missing_papers = sorted(expected_papers - set(decision_payloads))
    if unknown_papers or missing_papers:
        raise ValueError(
            f"decision paper coverage mismatch; missing={missing_papers}, unknown={unknown_papers}"
        )

    audit_documents: list[dict[str, Any]] = []
    selected_documents: list[dict[str, Any]] = []
    human_documents: list[dict[str, Any]] = []
    all_relations: list[dict[str, Any]] = []
    decision_counts: Counter[str] = Counter()

    for package in packages:
        doc_id = package["paper_id"]
        decisions = validate_decision_payload(
            package,
            decision_payloads[doc_id],
            decision_sources.get(doc_id, f"<memory:{doc_id}>")
        )
        tables_by_id = {table["table_id"]: table for table in package["tables"]}
        relations_by_table: dict[str, list[dict[str, Any]]] = {
            table_id: [] for table_id in tables_by_id
        }
        audit_children: list[dict[str, Any]] = []
        for item in package["review_items"]:
            decision = copy.deepcopy(decisions[item["child_id"]])
            decision["decision_source_file"] = decision_sources.get(doc_id)
            audit_item = copy.deepcopy(item)
            audit_item["subagent_decision"] = decision
            audit_children.append(audit_item)
            if decision["relevant_tables"]:
                decision_counts["selected_children"] += 1
            else:
                decision_counts["rejected_children"] += 1
            for relation in decision["relevant_tables"]:
                row = _relation_row(
                    package,
                    tables_by_id[relation["table_id"]],
                    item,
                    relation,
                    decision_sources.get(doc_id),
                )
                relations_by_table[relation["table_id"]].append(row)
                all_relations.append(row)

        table_summaries = [
            _table_summary(table, len(relations_by_table[table["table_id"]]))
            for table in package["tables"]
        ]
        audit_documents.append(
            {
                "paper_id": doc_id,
                "title": package.get("title"),
                "inventory_metadata": copy.deepcopy(package.get("inventory_metadata")),
                "tables": copy.deepcopy(package["tables"]),
                "table_summaries": copy.deepcopy(table_summaries),
                "scopes": copy.deepcopy(package["scopes"]),
                "parents": copy.deepcopy(package["parents"]),
                "children": audit_children,
            }
        )
        selected_documents.append(
            {
                "paper_id": doc_id,
                "tables": copy.deepcopy(table_summaries),
            }
        )
        human_documents.append(
            {
                "paper_id": doc_id,
                "tables": [
                    {
                        **_table_summary(table, len(relations_by_table[table["table_id"]])),
                        "table_body": copy.deepcopy(table["table_body"]),
                        "records": [
                            _human_row(row)
                            for row in relations_by_table[table["table_id"]]
                        ],
                    }
                    for table in package["tables"]
                ],
            }
        )

    table_count = sum(len(package["tables"]) for package in packages)
    child_count = sum(len(package["review_items"]) for package in packages)
    tables_with_relations = sum(
        summary["selected_relation_count"] > 0
        for document in selected_documents
        for summary in document["tables"]
    )
    full_audit = {
        "schema_version": FULL_AUDIT_SCHEMA_VERSION,
        "review_stage": REVIEW_STAGE,
        "document_count": len(packages),
        "table_count": table_count,
        "child_count": child_count,
        "selected_relation_count": len(all_relations),
        "documents": audit_documents,
    }
    selected = {
        "schema_version": SELECTED_SCHEMA_VERSION,
        "review_stage": REVIEW_STAGE,
        "document_count": len(packages),
        "table_count": table_count,
        "tables_with_relations": tables_with_relations,
        "selected_relation_count": len(all_relations),
        "documents": selected_documents,
        "relations": all_relations,
    }
    human_template = {
        "schema_version": HUMAN_SCHEMA_VERSION,
        "annotation_stage": "human_review_after_scope_subagent_v2",
        "label_semantics": "0 = correct/relevant, 1 = incorrect/irrelevant",
        "caption_policy": "table_caption is anchor metadata and is not a decision item",
        "subagent_predictions_are_provisional": True,
        "document_count": len(packages),
        "table_count": table_count,
        "candidate_relation_count": len(all_relations),
        "documents": human_documents,
    }
    status = {
        "schema_version": "scope-review-materialization-status-v2",
        "review_stage": REVIEW_STAGE,
        "document_count": len(packages),
        "table_count": table_count,
        "reviewed_child_count": child_count,
        "selected_child_count": decision_counts["selected_children"],
        "rejected_child_count": decision_counts["rejected_children"],
        "selected_relation_count": len(all_relations),
        "tables_with_relations": tables_with_relations,
        "tables_without_relations": table_count - tables_with_relations,
        "validation_failures": [],
    }
    return {
        "full_audit": full_audit,
        "selected_relations": selected,
        "human_template": human_template,
        "status": status,
    }


def _display(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def render_subagent_markdown(outputs: dict[str, Any]) -> str:
    audit = outputs["full_audit"]
    relations = outputs["selected_relations"]["relations"]
    by_paper_table: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for relation in relations:
        by_paper_table.setdefault((relation["paper_id"], relation["table_id"]), []).append(relation)
    lines = [
        "# Scope Subagent Review Results",
        "",
        f"- Reviewed unique children: {audit['child_count']}",
        f"- Selected table-child relations: {len(relations)}",
        "- Captions are table anchors, not reviewed children.",
        "",
    ]
    for document in audit["documents"]:
        lines.extend([f"## {document['paper_id']}", ""])
        for table in document["tables"]:
            rows = by_paper_table.get((document["paper_id"], table["table_id"]), [])
            lines.extend(
                [
                    f"### {table['table_label']} (`{table['table_id']}`)",
                    "",
                    f"Caption: {table['table_caption']}",
                    "",
                    f"Selected relations: {len(rows)}",
                    "",
                ]
            )
            if not rows:
                lines.extend(["None.", ""])
            for index, row in enumerate(rows, 1):
                lines.extend(
                    [
                        f"{index}. `{row['child_id']}`: {row['child_text']}",
                        f"   - Confidence: {row['subagent_confidence']}",
                        f"   - Role: {row['subagent_role']}",
                        f"   - Rationale: {row['subagent_rationale']}",
                        f"   - Parent: {row['full_parent_text']}",
                        "",
                    ]
                )
        rejected = [
            child for child in document["children"]
            if not child["subagent_decision"]["relevant_tables"]
        ]
        lines.extend(["### Rejected Children", ""])
        if not rejected:
            lines.extend(["None.", ""])
        for child in rejected:
            lines.extend(
                [
                    f"- `{child['child_id']}`: {child['child_text']}",
                    f"  - Reason: {child['subagent_decision']['rejection_reason']}",
                    "",
                ]
            )
    return "\n".join(lines).rstrip() + "\n"


def render_human_markdown(template: dict[str, Any]) -> str:
    lines = [
        "# Scope Review Human Annotation Template",
        "",
        "Fill `human_label` with 0 or 1 and add `human_rationale`. Subagent predictions are provisional.",
        "Captions are table anchors and are not annotation decisions.",
        "",
    ]
    for document in template["documents"]:
        lines.extend([f"## {document['paper_id']}", ""])
        for table in document["tables"]:
            lines.extend(
                [
                    f"### {table['table_label']} (`{table['table_id']}`)",
                    "",
                    f"Caption: {table['table_caption']}",
                    "",
                    "Table body:",
                    "",
                    _display(table.get("table_body")),
                    "",
                    f"Candidates: {table['selected_relation_count']}",
                    "",
                ]
            )
            if not table["records"]:
                lines.extend(["None.", ""])
            for index, row in enumerate(table["records"], 1):
                lines.extend(
                    [
                        f"{index}. `{row['child_id']}`: {row['child_text']}",
                        f"   - Parent id: `{row['parent_id']}`",
                        f"   - Full parent: {row['full_parent_text']}",
                        f"   - Offsets: [{row['char_start']}, {row['char_end']})",
                        f"   - Page: {_display(row.get('page'))}",
                        f"   - BBox: {_display(row.get('bbox'))}",
                        f"   - Source: {_display(row.get('source'))}",
                        f"   - Subagent confidence: {row['subagent_confidence']}",
                        f"   - Subagent role: {row['subagent_role']}",
                        f"   - Subagent rationale: {row['subagent_rationale']}",
                        "   - Human label:",
                        "   - Human rationale:",
                        "",
                    ]
                )
    return "\n".join(lines).rstrip() + "\n"


def write_materialized_outputs(
    outputs: dict[str, Any],
    output_dir: Path,
    input_path: Path | None = None,
    decision_dir: Path | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "full_audit_json": output_dir / "scope_review_full_audit.json",
        "selected_relations_json": output_dir / "scope_review_selected_relations.json",
        "subagent_markdown": output_dir / "scope_review_subagent_readable.md",
        "human_template_json": output_dir / "scope_review_human_template.json",
        "human_template_markdown": output_dir / "scope_review_human_template.md",
        "status_json": output_dir / "scope_review_materialization_status.json",
    }
    write_json(paths["full_audit_json"], outputs["full_audit"])
    write_json(paths["selected_relations_json"], outputs["selected_relations"])
    paths["subagent_markdown"].write_text(
        render_subagent_markdown(outputs), encoding="utf-8"
    )
    write_json(paths["human_template_json"], outputs["human_template"])
    paths["human_template_markdown"].write_text(
        render_human_markdown(outputs["human_template"]), encoding="utf-8"
    )
    status = copy.deepcopy(outputs["status"])
    status.update(
        {
            "source_inventory": str(input_path) if input_path else None,
            "decision_dir": str(decision_dir) if decision_dir else None,
            "outputs": {key: str(path) for key, path in paths.items() if key != "status_json"},
        }
    )
    write_json(paths["status_json"], status)
    return status


def materialize_from_paths(
    input_path: Path,
    decision_dir: Path,
    output_dir: Path,
    decision_glob: str = DEFAULT_DECISION_GLOB,
) -> dict[str, Any]:
    inventory = load_json(input_path)
    documents = inventory.get("documents") if isinstance(inventory, dict) else None
    if not isinstance(documents, list):
        raise ValueError("inventory must contain a documents list")
    expected_paper_ids = {paper_id(document) for document in documents}
    if len(expected_paper_ids) != len(documents):
        raise ValueError("inventory contains duplicate paper ids")
    payloads, sources = load_decision_payloads(
        decision_dir, expected_paper_ids, decision_glob
    )
    outputs = materialize_scope_reviews(inventory, payloads, sources)
    return write_materialized_outputs(
        outputs,
        output_dir,
        input_path=input_path,
        decision_dir=decision_dir,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate scope-review decisions and materialize audit, selected-relation, and human-review artifacts."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to batch_scope_candidate_inventory.json.",
    )
    parser.add_argument(
        "--decision-dir",
        type=Path,
        default=DEFAULT_DECISION_DIR,
        help="Directory containing fresh per-paper scope review decision JSON files.",
    )
    parser.add_argument(
        "--decision-glob",
        default=DEFAULT_DECISION_GLOB,
        help="Glob used inside --decision-dir (default: *_scope_review_decisions.json).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for validated materialized outputs.",
    )
    args = parser.parse_args()
    status = materialize_from_paths(
        args.input, args.decision_dir, args.output_dir, args.decision_glob
    )
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
