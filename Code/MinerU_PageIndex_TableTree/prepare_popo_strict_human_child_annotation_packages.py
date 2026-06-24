import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent

from popo_workflow_helpers import (
    candidate_signals,
    child_score,
    extract_popo_nodes,
    load_json,
    table_anchor_by_order,
)


DEFAULT_MANUAL = (
    PROJECT_ROOT
    / "batch_table_text_tree"
    / "manual_table_body_text_annotations"
    / "batch_manual_table_body_text_annotations.json"
)
DEFAULT_TREE_DIR = (
    PROJECT_ROOT / "batch_table_text_tree" / "mineru_popo_comparison" / "popo_build_tree"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "batch_table_text_tree"
    / "mineru_popo_comparison"
    / "popo_human_child_annotations_strict"
)
DEFAULT_THRESHOLD_SENSITIVITY = (
    PROJECT_ROOT
    / "batch_table_text_tree"
    / "mineru_popo_comparison"
    / "popo_workflow_full_comparison"
    / "popo_threshold_sensitivity.json"
)

POLICY_VERSION = "popo-strict-human-template-v1"
NON_EVIDENCE_TITLE_RE = re.compile(
    r"\b(references?|bibliography|acknowledg|author list|contributions?)\b|^preface$",
    re.I,
)


def text_hash(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def review_key(row: dict[str, Any]) -> str:
    return "::".join(
        (
            row["slug"],
            row["table_anchor"]["canonical_label"],
            row["candidate_id"],
            row["child"]["child_id"],
        )
    )


def compact_anchor(anchor: dict[str, Any]) -> dict[str, Any]:
    return {
        "table_node_id": anchor.get("table_node_id"),
        "canonical_label": anchor.get("canonical_label"),
        "caption": anchor.get("caption"),
        "page_index": anchor.get("page_index"),
        "bbox": anchor.get("bbox"),
        "order_key": anchor.get("order_key"),
        "raw_content_index": anchor.get("raw_content_index"),
        "parent_node_id": anchor.get("parent_node_id"),
        "parent_title": anchor.get("parent_title"),
        "assignment_reason": anchor.get("assignment_reason"),
        "table_html": anchor.get("table_html"),
        "table_code_body": anchor.get("table_code_body"),
        "image_path": anchor.get("image_path"),
    }


def is_non_evidence_parent(text_node: dict[str, Any]) -> bool:
    title_text = " ".join(
        str(value or "")
        for value in [text_node.get("title"), *(text_node.get("ancestor_titles") or [])]
    ).strip()
    return bool(NON_EVIDENCE_TITLE_RE.search(title_text))


def safe_distance(value: Any) -> int:
    if value is None:
        return 999999
    return int(value)


def rank_row(row: dict[str, Any]) -> tuple[Any, ...]:
    child_signals = row["child"]["child_selection_signals"]
    parent_signals = row["candidate_signals"]
    family_hits = child_signals.get("evidence_family_hits") or []
    return (
        1 if child_signals.get("exact_table_reference") else 0,
        int(child_signals.get("score") or 0),
        int(parent_signals.get("priority_score") or 0),
        1 if child_signals.get("condition_match") else 0,
        1 if child_signals.get("result_match") else 0,
        len(family_hits),
        -safe_distance(parent_signals.get("page_distance")),
        -safe_distance(parent_signals.get("block_distance")),
        len(row["child"].get("child_text") or ""),
    )


def make_child_row(
    slug: str,
    table_anchor: dict[str, Any],
    popo_table: dict[str, Any],
    text_node: dict[str, Any],
    child: dict[str, Any],
    parent_signals: dict[str, Any],
    selection_reason: str,
) -> dict[str, Any]:
    full_parent_text = text_node.get("content") or ""
    child_text = child.get("child_text") or ""
    if child_text != full_parent_text[child["char_start"] : child["char_end"]]:
        raise ValueError(
            f"Child offset mismatch: {slug} {table_anchor['canonical_label']} "
            f"{text_node['node_id']} {child['child_id']}"
        )
    signals = child_score(table_anchor, child_text)
    return {
        "schema_version": "1.0",
        "source_workflow": "mineru_popo",
        "candidate_policy_version": POLICY_VERSION,
        "slug": slug,
        "table_anchor": compact_anchor(table_anchor),
        "popo_table_node": {
            "node_id": popo_table.get("node_id"),
            "block_ids": popo_table.get("block_ids"),
            "page_indices": popo_table.get("page_indices"),
            "ancestor_titles": popo_table.get("ancestor_titles"),
        },
        "candidate_id": f"{table_anchor['canonical_label'].lower().replace(' ', '-')}-{text_node['node_id']}",
        "parent_paragraph_id": text_node["node_id"],
        "parent_node_id": text_node["node_id"],
        "parent_title": text_node.get("title"),
        "parent_ancestor_titles": text_node.get("ancestor_titles") or [],
        "parent_page_indices": text_node.get("page_indices") or [],
        "parent_raw_content_indices": text_node.get("block_ids") or [],
        "parent_evidence_type": "popo_text_node",
        "parent_manual_rationale": None,
        "full_parent_text": full_parent_text,
        "candidate_signals": parent_signals,
        "child": {
            "child_id": child["child_id"],
            "char_start": child["char_start"],
            "char_end": child["char_end"],
            "child_text": child_text,
            "child_label": None,
            "child_label_source": "awaiting_human_gold",
            "child_review_status": "awaiting_human_annotation",
            "child_selection_signals": signals,
            "send_to_semantic_review": True,
            "candidate_policy_version": POLICY_VERSION,
            "selection_reason": selection_reason,
        },
    }


def table_rows(
    slug: str,
    table_anchor: dict[str, Any],
    popo_table: dict[str, Any],
    text_nodes: list[dict[str, Any]],
    parent_score_threshold: int,
    child_score_threshold: int,
) -> list[dict[str, Any]]:
    rows = []
    for text_node in text_nodes:
        if is_non_evidence_parent(text_node):
            continue
        parent_signals = candidate_signals(table_anchor, popo_table, text_node)
        parent_selected = (
            parent_signals["exact_table_reference"]
            or parent_signals["priority_score"] >= parent_score_threshold
            or parent_signals["priority_tier"]
            in {"tier_1_explicit_reference", "tier_2_local_context"}
        )
        if not parent_selected:
            continue
        for child in text_node.get("child_spans") or []:
            signals = child_score(table_anchor, child["child_text"])
            if signals["exact_table_reference"] or signals["score"] >= child_score_threshold:
                rows.append(
                    make_child_row(
                        slug,
                        table_anchor,
                        popo_table,
                        text_node,
                        child,
                        parent_signals,
                        "popo_strict_threshold_or_explicit_reference",
                    )
                )
    return rows


def choose_table_rows(
    rows: list[dict[str, Any]],
    all_parent_rows: list[dict[str, Any]],
    min_children_per_table: int,
    max_children_per_table: int,
) -> list[dict[str, Any]]:
    ranked = sorted(rows, key=rank_row, reverse=True)
    selected = ranked[:max_children_per_table]
    selected_keys = {(row["candidate_id"], row["child"]["child_id"]) for row in selected}

    if len(selected) < min_children_per_table:
        fallback = sorted(all_parent_rows, key=rank_row, reverse=True)
        for row in fallback:
            key = (row["candidate_id"], row["child"]["child_id"])
            if key in selected_keys:
                continue
            row = json.loads(json.dumps(row, ensure_ascii=False))
            row["child"]["selection_reason"] = "popo_table_coverage_fallback"
            selected.append(row)
            selected_keys.add(key)
            if len(selected) >= min_children_per_table:
                break

    selected.sort(
        key=lambda row: (
            row["slug"],
            row["table_anchor"]["canonical_label"],
            min(row.get("parent_raw_content_indices") or [999999]),
            row["child"]["char_start"],
        )
    )
    return selected


def all_parent_child_rows(
    slug: str,
    table_anchor: dict[str, Any],
    popo_table: dict[str, Any],
    text_nodes: list[dict[str, Any]],
    parent_score_threshold: int,
) -> list[dict[str, Any]]:
    rows = []
    for text_node in text_nodes:
        if is_non_evidence_parent(text_node):
            continue
        parent_signals = candidate_signals(table_anchor, popo_table, text_node)
        parent_selected = (
            parent_signals["exact_table_reference"]
            or parent_signals["priority_score"] >= parent_score_threshold
            or parent_signals["priority_tier"]
            in {"tier_1_explicit_reference", "tier_2_local_context"}
        )
        if not parent_selected:
            continue
        for child in text_node.get("child_spans") or []:
            rows.append(
                make_child_row(
                    slug,
                    table_anchor,
                    popo_table,
                    text_node,
                    child,
                    parent_signals,
                    "popo_parent_selected_fallback_pool",
                )
            )
    return rows


def build_selected_rows(
    manual: dict[str, Any],
    tree_dir: Path,
    parent_score_threshold: int,
    child_score_threshold: int,
    min_children_per_table: int,
    max_children_per_table: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected_rows = []
    audit_failures = []
    for document in manual.get("documents") or []:
        slug = document["slug"]
        tree_path = tree_dir / f"{slug}.json"
        if not tree_path.exists():
            audit_failures.append({"slug": slug, "reason": "missing_popo_tree", "path": str(tree_path)})
            continue
        tree = load_json(tree_path)
        text_nodes, table_nodes = extract_popo_nodes(tree, slug)
        anchors = table_anchor_by_order(document)
        if len(anchors) != len(table_nodes):
            audit_failures.append(
                {
                    "slug": slug,
                    "reason": "table_count_mismatch",
                    "manual_table_count": len(anchors),
                    "popo_table_count": len(table_nodes),
                }
            )
        for table_index, table_anchor in enumerate(anchors):
            if table_index >= len(table_nodes):
                continue
            popo_table = table_nodes[table_index]
            threshold_rows = table_rows(
                slug,
                table_anchor,
                popo_table,
                text_nodes,
                parent_score_threshold,
                child_score_threshold,
            )
            fallback_pool = all_parent_child_rows(
                slug,
                table_anchor,
                popo_table,
                text_nodes,
                parent_score_threshold,
            )
            selected_rows.extend(
                choose_table_rows(
                    threshold_rows,
                    fallback_pool,
                    min_children_per_table,
                    max_children_per_table,
                )
            )
    return selected_rows, audit_failures


def load_threshold_sensitivity(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = load_json(path)
    return list(payload.get("rows") or [])


def selected_child_count_for_cap(
    manual: dict[str, Any],
    tree_dir: Path,
    parent_score_threshold: int,
    child_score_threshold: int,
    min_children_per_table: int,
    cap: int,
) -> int:
    rows, _ = build_selected_rows(
        manual,
        tree_dir,
        parent_score_threshold,
        child_score_threshold,
        min_children_per_table,
        cap,
    )
    return len(rows)


def build_threshold_rationale(
    manual: dict[str, Any],
    tree_dir: Path,
    threshold_sensitivity_path: Path,
    parent_score_threshold: int,
    child_score_threshold: int,
    min_children_per_table: int,
    max_children_per_table: int,
) -> dict[str, Any]:
    sensitivity_rows = load_threshold_sensitivity(threshold_sensitivity_path)
    selected_pair = None
    for row in sensitivity_rows:
        if (
            row.get("parent_score_threshold") == parent_score_threshold
            and row.get("child_score_threshold") == child_score_threshold
        ):
            selected_pair = row
            break

    cap_values = sorted({8, 9, 10, 11, 12, max_children_per_table})
    cap_sensitivity = [
        {
            "max_children_per_table": cap,
            "candidate_children": selected_child_count_for_cap(
                manual,
                tree_dir,
                parent_score_threshold,
                child_score_threshold,
                min_children_per_table,
                cap,
            ),
        }
        for cap in cap_values
    ]

    return {
        "basis_type": "calibration_on_current_five_paper_development_set",
        "threshold_sensitivity_source": str(threshold_sensitivity_path),
        "available_parent_child_threshold_rows": sensitivity_rows,
        "selected_parent_child_threshold_row": selected_pair,
        "parent_child_threshold_reason": [
            "The 120/60 pair is chosen from the Popo sensitivity grid, not by intuition.",
            "It keeps the same historical manual-evidence preservation audit as looser settings: 188/190 covered in the five-paper development set.",
            "It reduces projected code-only children from 913 at 100/40 to 433, while avoiding the more aggressive 140/80 row that drops to 233 before human labels are available.",
            "The selected pair should be re-estimated after human child-level labels are available.",
        ],
        "cap_sensitivity_at_selected_thresholds": cap_sensitivity,
        "per_table_cap_reason": [
            "The cap controls human annotation workload after the 120/60 recall filter.",
            "At the selected thresholds, cap 9 gives 253 children, cap 10 gives 272, cap 11 gives 290, and cap 12 gives 305 on the current five-paper set.",
            "Cap 11 is selected because it stays close to the requested roughly-300 annotation budget while retaining more candidates than cap 9 or cap 10.",
        ],
        "min_children_per_table_reason": [
            "The minimum of 3 is a coverage guardrail, not a learned threshold.",
            "It prevents a table from entering the human template with only one narrow evidence style when a fallback pool exists.",
            "It should be validated against human labels and adjusted if per-table false positives dominate.",
        ],
    }


def child_template(row: dict[str, Any]) -> dict[str, Any]:
    child = row["child"]
    signals = child.get("child_selection_signals") or {}
    return {
        "review_key": review_key(row),
        "child_id": child["child_id"],
        "child_text": child["child_text"],
        "child_text_sha256": text_hash(child["child_text"]),
        "char_start": child["char_start"],
        "char_end": child["char_end"],
        "code_selection_score": signals.get("score"),
        "code_selection_signals": signals,
        "selection_reason": child.get("selection_reason"),
        "human_label": None,
        "semantic_role": None,
        "citation_support": None,
        "human_rationale": None,
    }


def build_document(slug: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    table_groups = defaultdict(list)
    for row in rows:
        table_groups[row["table_anchor"]["canonical_label"]].append(row)

    tables = []
    for table_label, table_rows_for_label in sorted(table_groups.items()):
        parents = defaultdict(list)
        for row in table_rows_for_label:
            parents[row["candidate_id"]].append(row)
        parent_items = []
        for candidate_id, parent_rows in sorted(parents.items()):
            first = parent_rows[0]
            parent_items.append(
                {
                    "candidate_id": candidate_id,
                    "parent_node_id": first["parent_node_id"],
                    "parent_title": first["parent_title"],
                    "parent_ancestor_titles": first["parent_ancestor_titles"],
                    "parent_page_indices": first["parent_page_indices"],
                    "parent_raw_content_indices": first["parent_raw_content_indices"],
                    "parent_evidence_type": first["parent_evidence_type"],
                    "full_parent_text": first["full_parent_text"],
                    "full_parent_text_sha256": text_hash(first["full_parent_text"]),
                    "candidate_signals": first["candidate_signals"],
                    "children": [child_template(row) for row in parent_rows],
                }
            )
        tables.append(
            {
                "table_label": table_label,
                "table_anchor": table_rows_for_label[0]["table_anchor"],
                "popo_table_node": table_rows_for_label[0]["popo_table_node"],
                "parents": parent_items,
            }
        )
    return {"slug": slug, "table_count": len(tables), "tables": tables}


def build_batch(rows: list[dict[str, Any]], source_path: Path, policy: dict[str, Any]) -> dict[str, Any]:
    by_slug = defaultdict(list)
    for row in rows:
        by_slug[row["slug"]].append(row)
    documents = [build_document(slug, slug_rows) for slug, slug_rows in sorted(by_slug.items())]
    return {
        "schema_version": "1.0",
        "annotation_stage": "human_child_semantic_gold_popo",
        "source_candidates": str(source_path),
        "source_workflow": "mineru_popo",
        "label_semantics": {
            "0": "correct/relevant child block for the table",
            "1": "incorrect/irrelevant child block for the table",
        },
        "annotation_policy": [
            "Judge each child from the table, the complete Popo parent text node, and the exact child text.",
            "Use 0 when the child introduces, supplements, constrains, or interprets the table.",
            "Use 0 for table-scoped datasets, models, metrics, prompts, shots, splits, baselines, training settings, or result interpretation.",
            "Use 1 when the child is generic, redundant, unrelated, or too weak to support later citation-reference judgment.",
            "Do not use code_selection_score as the label. It is only a ranking aid.",
        ],
        "prefilter_policy": policy,
        "document_count": len(documents),
        "table_count": sum(document["table_count"] for document in documents),
        "candidate_child_count": len(rows),
        "documents": documents,
    }


def render_markdown(document: dict[str, Any]) -> str:
    lines = [
        f"# Popo Human Child Annotation Template: {document['slug']}",
        "",
        "Label 0 = correct/relevant. Label 1 = incorrect/irrelevant.",
        "Scores are only ranking aids. Fill the JSON file for machine-readable labels.",
        "",
    ]
    for table in document["tables"]:
        anchor = table["table_anchor"]
        lines.extend(
            [
                f"## {table['table_label']}",
                "",
                f"Caption: {anchor.get('caption') or ''}",
                f"Old MinerU page: {anchor.get('page_index')}",
                f"Popo table pages: {table.get('popo_table_node', {}).get('page_indices')}",
                "",
            ]
        )
        for parent in table["parents"]:
            lines.extend(
                [
                    f"### {parent['candidate_id']}",
                    "",
                    f"Section: {parent.get('parent_title') or ''}",
                    f"Ancestor path: {' > '.join(parent.get('parent_ancestor_titles') or [])}",
                    f"Parent pages: {parent.get('parent_page_indices')}",
                    f"Parent block ids: {parent.get('parent_raw_content_indices')}",
                    f"Parent score: {parent.get('candidate_signals', {}).get('priority_score')}",
                    "",
                    f"Parent text: {parent['full_parent_text']}",
                    "",
                ]
            )
            for child in parent["children"]:
                lines.extend(
                    [
                        f"- Review key: `{child['review_key']}`",
                        f"  - Score: {child.get('code_selection_score')}",
                        f"  - Signals: exact_ref={child.get('code_selection_signals', {}).get('exact_table_reference')}, "
                        f"families={child.get('code_selection_signals', {}).get('evidence_family_hits')}",
                        f"  - Child: {child['child_text'].strip()}",
                        "  - Human label: ",
                        "",
                    ]
                )
    return "\n".join(lines).rstrip() + "\n"


def write_outputs(
    rows: list[dict[str, Any]],
    audit_failures: list[dict[str, Any]],
    output_dir: Path,
    policy: dict[str, Any],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_path = output_dir / "popo_strict_selected_child_blocks.json"
    selected_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    batch = build_batch(rows, selected_path, policy)
    batch_path = output_dir / "batch_popo_strict_human_child_annotation_template.json"
    batch_path.write_text(json.dumps(batch, ensure_ascii=False, indent=2), encoding="utf-8")

    documents = []
    counts_by_doc = defaultdict(int)
    counts_by_table = defaultdict(int)
    for document in batch["documents"]:
        json_path = output_dir / f"{document['slug']}_popo_strict_human_child_annotation_template.json"
        md_path = output_dir / f"{document['slug']}_popo_strict_human_child_annotation_template.md"
        document_payload = {
            "schema_version": batch["schema_version"],
            "annotation_stage": batch["annotation_stage"],
            "source_candidates": str(selected_path),
            "source_workflow": batch["source_workflow"],
            "label_semantics": batch["label_semantics"],
            "annotation_policy": batch["annotation_policy"],
            "prefilter_policy": policy,
            "documents": [document],
        }
        json_path.write_text(json.dumps(document_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        md_path.write_text(render_markdown(document), encoding="utf-8")
        child_count = sum(
            len(parent["children"])
            for table in document["tables"]
            for parent in table["parents"]
        )
        counts_by_doc[document["slug"]] = child_count
        for table in document["tables"]:
            table_child_count = sum(len(parent["children"]) for parent in table["parents"])
            counts_by_table[f"{document['slug']}::{table['table_label']}"] = table_child_count
        documents.append(
            {
                "slug": document["slug"],
                "tables": document["table_count"],
                "candidate_children": child_count,
                "template_json": str(json_path),
                "readable_markdown": str(md_path),
            }
        )

    summary_path = output_dir / "popo_strict_human_child_annotation_summary.md"
    summary_path.write_text(render_summary(batch, documents, policy, audit_failures), encoding="utf-8")

    status = {
        "schema_version": "1.0",
        "source_workflow": "mineru_popo",
        "strict_selected_child_blocks": str(selected_path),
        "batch_template": str(batch_path),
        "summary_markdown": str(summary_path),
        "candidate_children": len(rows),
        "documents": documents,
        "counts_by_document": dict(sorted(counts_by_doc.items())),
        "counts_by_table": dict(sorted(counts_by_table.items())),
        "prefilter_policy": policy,
        "audit_failures": audit_failures,
    }
    status_path = output_dir / "popo_strict_human_child_annotation_package_status.json"
    status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    return status


def render_summary(
    batch: dict[str, Any],
    documents: list[dict[str, Any]],
    policy: dict[str, Any],
    audit_failures: list[dict[str, Any]],
) -> str:
    lines = [
        "# Popo Strict Human Child Annotation Summary",
        "",
        f"- Source workflow: {batch['source_workflow']}",
        f"- Documents: {batch['document_count']}",
        f"- Tables: {batch['table_count']}",
        f"- Candidate children for human review: {batch['candidate_child_count']}",
        f"- Parent threshold: {policy['parent_score_threshold']}",
        f"- Child threshold: {policy['child_score_threshold']}",
        f"- Per-table child cap: {policy['max_children_per_table']}",
        f"- Per-table fallback minimum: {policy['min_children_per_table']}",
        "",
        "## Documents",
        "",
        "| document | tables | candidate children | markdown |",
        "| --- | ---: | ---: | --- |",
    ]
    for item in documents:
        lines.append(
            f"| {item['slug']} | {item['tables']} | {item['candidate_children']} | {item['readable_markdown']} |"
        )
    rationale = policy.get("threshold_rationale") or {}
    if rationale:
        selected = rationale.get("selected_parent_child_threshold_row") or {}
        lines.extend(
            [
                "",
                "## Threshold Rationale",
                "",
                f"- Basis: {rationale.get('basis_type')}",
                f"- Sensitivity source: {rationale.get('threshold_sensitivity_source')}",
                f"- Selected 120/60 row projected children: {selected.get('projected_selected_children')}",
                f"- Selected 120/60 evidence coverage: {selected.get('manual_evidence_covered')}/{selected.get('manual_evidence_count')}",
                "",
                "### Parent/Child Threshold",
                "",
            ]
        )
        for reason in rationale.get("parent_child_threshold_reason") or []:
            lines.append(f"- {reason}")
        lines.extend(
            [
                "",
                "### Per-Table Cap Sensitivity",
                "",
                "| max children per table | total template children |",
                "| ---: | ---: |",
            ]
        )
        for row in rationale.get("cap_sensitivity_at_selected_thresholds") or []:
            lines.append(f"| {row['max_children_per_table']} | {row['candidate_children']} |")
        lines.extend(["", "### Coverage Floor", ""])
        for reason in rationale.get("min_children_per_table_reason") or []:
            lines.append(f"- {reason}")
    lines.extend(["", "## Audit", ""])
    if audit_failures:
        for failure in audit_failures:
            lines.append(f"- {failure}")
    else:
        lines.append("- No Popo table-count audit failures were reported.")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare a strict human child annotation template from MinerU-Popo table-text trees."
    )
    parser.add_argument("--manual", type=Path, default=DEFAULT_MANUAL)
    parser.add_argument("--tree-dir", type=Path, default=DEFAULT_TREE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--threshold-sensitivity", type=Path, default=DEFAULT_THRESHOLD_SENSITIVITY)
    parser.add_argument("--parent-score-threshold", type=int, default=120)
    parser.add_argument("--child-score-threshold", type=int, default=60)
    parser.add_argument("--min-children-per-table", type=int, default=3)
    parser.add_argument("--max-children-per-table", type=int, default=11)
    args = parser.parse_args()

    manual = load_json(args.manual)
    threshold_rationale = build_threshold_rationale(
        manual,
        args.tree_dir,
        args.threshold_sensitivity,
        args.parent_score_threshold,
        args.child_score_threshold,
        args.min_children_per_table,
        args.max_children_per_table,
    )
    policy = {
        "schema_version": "1.0",
        "policy_name": POLICY_VERSION,
        "source_tree": str(args.tree_dir),
        "manual_table_anchors": str(args.manual),
        "parent_unit": "MinerU-Popo type=text node",
        "child_unit": "sentence-like spans split from the Popo text-node content",
        "parent_score_threshold": args.parent_score_threshold,
        "child_score_threshold": args.child_score_threshold,
        "min_children_per_table": args.min_children_per_table,
        "max_children_per_table": args.max_children_per_table,
        "threshold_rationale": threshold_rationale,
        "rule": [
            "Use Popo text nodes as parent blocks.",
            "Keep children with explicit table reference or child score above threshold.",
            "Rank by explicit table reference, child score, parent priority, evidence family, and proximity.",
            "Cap each table to keep the first human annotation round manageable.",
            "Fallback to highest-ranked children when a table would otherwise have too few candidates.",
            "Exclude reference/bibliography/preface/acknowledgement-like parent nodes from this first annotation template.",
        ],
    }
    rows, audit_failures = build_selected_rows(
        manual,
        args.tree_dir,
        args.parent_score_threshold,
        args.child_score_threshold,
        args.min_children_per_table,
        args.max_children_per_table,
    )
    status = write_outputs(rows, audit_failures, args.output_dir, policy)
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
