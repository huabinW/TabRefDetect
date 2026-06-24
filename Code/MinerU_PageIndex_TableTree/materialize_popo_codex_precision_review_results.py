from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any


BASE = Path(__file__).resolve().parent
DEFAULT_INPUT = (
    BASE
    / "batch_table_text_tree"
    / "mineru_popo_comparison"
    / "popo_human_child_annotations_strict"
    / "batch_popo_strict_human_child_annotation_template.json"
)
DEFAULT_DECISION_DIR = (
    BASE
    / "batch_table_text_tree"
    / "mineru_popo_comparison"
    / "popo_codex_precision_v2"
    / "review_packages"
)
DEFAULT_OUTPUT_DIR = (
    BASE
    / "batch_table_text_tree"
    / "mineru_popo_comparison"
    / "popo_codex_precision_v2"
    / "results"
)
DEFAULT_CAPTION_RESOLUTION = (
    BASE
    / "batch_table_text_tree"
    / "local_table_caption_resolution"
    / "batch_local_table_caption_resolution.json"
)
ALLOWED_ROLES = {
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
}
ALLOWED_SUPPORT = {"direct", "indirect", "none"}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def iter_children(payload: dict[str, Any]):
    for document in payload["documents"]:
        for table in document["tables"]:
            for parent in table["parents"]:
                for child in parent["children"]:
                    yield document, table, parent, child


def normalize_decision(decision: dict[str, Any], path: Path) -> dict[str, Any]:
    normalized = dict(decision)
    if "codex_label" not in normalized:
        if "label" in normalized:
            normalized["codex_label"] = normalized["label"]
        elif "final_child_label" in normalized:
            normalized["codex_label"] = normalized["final_child_label"]
    if "rationale" not in normalized and "codex_rationale" in normalized:
        normalized["rationale"] = normalized["codex_rationale"]
    normalized["decision_source_file"] = str(path)
    return normalized


def load_decisions(decision_dir: Path) -> tuple[dict[str, dict[str, Any]], list[Path]]:
    decisions: dict[str, dict[str, Any]] = {}
    paths = sorted(decision_dir.glob("*_codex_precision_v2_decisions.json"))
    for path in paths:
        payload = load_json(path)
        if payload.get("review_stage") != "codex_table_scope_precision_review_v2":
            raise ValueError(f"{path}: unexpected review_stage")
        for decision in payload.get("decisions", []):
            key = decision.get("review_key")
            if not key:
                raise ValueError(f"{path}: decision missing review_key")
            if key in decisions:
                raise ValueError(f"Duplicate decision: {key}")
            decisions[key] = normalize_decision(decision, path)
    return decisions, paths


def load_caption_resolution(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    if not path.exists():
        return {}
    payload = load_json(path)
    overrides: dict[tuple[str, str], dict[str, Any]] = {}
    if not isinstance(payload, list):
        return overrides
    for document in payload:
        slug = document.get("slug")
        if not slug:
            continue
        for resolution in document.get("resolutions", []):
            table = resolution.get("table") or {}
            selected = resolution.get("selected_caption") or {}
            label = (
                table.get("current_caption_label")
                or selected.get("canonical_label")
                or selected.get("raw_label")
            )
            caption = selected.get("title_candidate") or table.get("title")
            if not label or not caption:
                continue
            overrides[(slug, label)] = {
                "caption": caption,
                "confidence": resolution.get("confidence"),
                "anchor_status": resolution.get("anchor_status"),
                "anchor_debug_required": resolution.get("anchor_debug_required"),
                "selected_caption": selected,
                "table": table,
            }
    return overrides


def apply_caption_resolution(
    source: dict[str, Any],
    caption_overrides: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    output = copy.deepcopy(source)
    for document in output.get("documents", []):
        slug = document.get("slug")
        for table in document.get("tables", []):
            label = table.get("table_label") or (
                table.get("table_anchor") or {}
            ).get("canonical_label")
            anchor = table.setdefault("table_anchor", {})
            override = caption_overrides.get((slug, label))
            if not override:
                anchor.setdefault("caption_resolution_status", "not_checked")
                continue
            original_caption = anchor.get("caption") or ""
            resolved_caption = override.get("caption") or ""
            if resolved_caption and resolved_caption != original_caption:
                anchor["raw_caption_before_resolution"] = original_caption
                anchor["caption"] = resolved_caption
                anchor["caption_resolution_status"] = "resolved_from_caption_stage"
            else:
                anchor["caption_resolution_status"] = "kept_existing"
            anchor["caption_resolution_confidence"] = override.get("confidence")
            anchor["caption_anchor_status"] = override.get("anchor_status")
            anchor["caption_anchor_debug_required"] = override.get(
                "anchor_debug_required"
            )
            selected = override.get("selected_caption") or {}
            anchor["caption_source"] = selected.get("source")
            anchor["caption_source_content_index"] = selected.get("content_index")
    return output


def validate_decision(child: dict[str, Any], decision: dict[str, Any]) -> list[str]:
    failures = []
    if decision.get("child_id") != child["child_id"]:
        failures.append("child_id mismatch")
    if decision.get("child_text_sha256") != child["child_text_sha256"]:
        failures.append("child_text_sha256 mismatch")
    label = decision.get("codex_label")
    if label not in (0, 1):
        failures.append("codex_label must be 0 or 1")
    if decision.get("semantic_role") not in ALLOWED_ROLES:
        failures.append("invalid semantic_role")
    if decision.get("citation_support") not in ALLOWED_SUPPORT:
        failures.append("invalid citation_support")
    if not str(decision.get("rationale") or "").strip():
        failures.append("rationale is required")
    if label == 1 and decision.get("semantic_role") != "irrelevant":
        failures.append("label 1 must use semantic_role=irrelevant")
    if label == 1 and decision.get("citation_support") != "none":
        failures.append("label 1 must use citation_support=none")
    if label == 0 and decision.get("semantic_role") == "irrelevant":
        failures.append("label 0 cannot use semantic_role=irrelevant")
    human_label = child.get("human_label")
    if human_label is not None and label != human_label:
        failures.append("Codex decision conflicts with existing human gold")
    return failures


def infer_decision_source(
    decisions: dict[str, dict[str, Any]],
) -> tuple[str, str]:
    if any(decision.get("reaudit_source") for decision in decisions.values()):
        return (
            "codex_precision_v2_plus_supplementary_context_reaudit_v3",
            "codex_supplementary_context_reaudit_v3",
        )
    return (
        "codex_skill_0_5_0_fresh_supplementary_review",
        "codex_skill_0_5_0_fresh_supplementary_review",
    )


def build_retained_template(
    source: dict[str, Any],
    decisions: dict[str, dict[str, Any]],
    policy_version: str,
) -> dict[str, Any]:
    output = copy.deepcopy(source)
    output["annotation_stage"] = "human_gold_after_codex_table_scope_precision_v2"
    output["source_candidates"] = str(DEFAULT_INPUT)
    output["codex_review_policy"] = {
        "version": policy_version,
        "code_stage": "All strict candidates were reviewed; no code precision filter was applied.",
        "semantic_stage": (
            "Codex retained only supplementary information not already visible "
            "in the table that is needed to interpret, reproduce, or verify it."
        ),
        "human_gold_status": (
            "Codex labels are provisional predictions. human_label remains the "
            "field for user gold annotations."
        ),
    }
    retained_count = 0
    tables_with_candidates = 0
    for document in output["documents"]:
        document_count = 0
        for table in document["tables"]:
            retained_parents = []
            table_count = 0
            for parent in table["parents"]:
                retained_children = []
                for child in parent["children"]:
                    decision = decisions[child["review_key"]]
                    if decision["codex_label"] == 0:
                        child["codex_review"] = copy.deepcopy(decision)
                        child["codex_label"] = 0
                        child["codex_label_source"] = decision.get(
                            "reaudit_source",
                            "codex_table_scope_precision_review_v2",
                        )
                        retained_children.append(child)
                        table_count += 1
                if retained_children:
                    parent["children"] = retained_children
                    retained_parents.append(parent)
            table["parents"] = retained_parents
            table["candidate_child_count"] = table_count
            document_count += table_count
            if table_count:
                tables_with_candidates += 1
        document["candidate_child_count"] = document_count
        retained_count += document_count
    output["candidate_child_count"] = retained_count
    output["tables_with_candidates"] = tables_with_candidates
    return output


def render_readable(template: dict[str, Any]) -> str:
    lines = [
        "# Codex Supplementary-Context Human Annotation Template",
        "",
        "Only Codex label-0 candidates are listed here.",
        "Codex decisions are provisional; fill human_label with 0 or 1.",
        "",
    ]
    for document in template["documents"]:
        lines.extend(
            [
                f"## {document['slug']}",
                "",
                f"Candidates: {document['candidate_child_count']}",
                "",
            ]
        )
        for table in document["tables"]:
            lines.extend(
                [
                    f"### {table['table_label']}",
                    "",
                    f"Caption: {table['table_anchor'].get('caption') or ''}",
                    f"Candidates: {table['candidate_child_count']}",
                    "",
                ]
            )
            for parent in table["parents"]:
                lines.extend(
                    [
                        f"#### {parent['candidate_id']}",
                        "",
                        f"Section: {parent.get('parent_title') or ''}",
                        "",
                        f"Parent text: {parent['full_parent_text']}",
                        "",
                    ]
                )
                for child in parent["children"]:
                    review = child["codex_review"]
                    lines.extend(
                        [
                            f"- Review key: `{child['review_key']}`",
                            f"  - Code score: {child.get('code_selection_score')}",
                            f"  - Codex role: {review['semantic_role']}",
                            f"  - Citation support: {review['citation_support']}",
                            f"  - Codex rationale: {review['rationale']}",
                            f"  - Child: {child['child_text']}",
                            f"  - Human label: {'' if child.get('human_label') is None else child['human_label']}",
                            f"  - Human rationale: {child.get('human_rationale') or ''}",
                            "",
                        ]
                    )
    return "\n".join(lines).rstrip() + "\n"


def build_slim_annotation_template(
    template: dict[str, Any],
    reviewed_path: Path,
    retained_path: Path,
    demoted_path: Path,
    full_audit_template_path: Path,
) -> dict[str, Any]:
    slim: dict[str, Any] = {
        "schema_version": "slim-human-annotation-v1",
        "annotation_stage": template.get("annotation_stage"),
        "label_semantics": "0 = correct/relevant, 1 = incorrect/irrelevant",
        "caption_policy": (
            "table_caption is preserved as a table anchor field. It is not a "
            "Codex-reviewed child candidate."
        ),
        "source_candidates": template.get("source_candidates"),
        "full_audit_files": {
            "all_reviewed_candidates": str(reviewed_path),
            "retained_children": str(retained_path),
            "demoted_children": str(demoted_path),
            "full_audit_template": str(full_audit_template_path),
        },
        "documents": [],
    }
    total_candidates = 0
    for document in template["documents"]:
        slim_document = {
            "slug": document["slug"],
            "candidate_child_count": document.get("candidate_child_count", 0),
            "tables": [],
        }
        for table in document["tables"]:
            anchor = table.get("table_anchor") or {}
            slim_table = {
                "table_label": table.get("table_label"),
                "table_caption": anchor.get("caption") or "",
                "table_page_index": anchor.get("page_index"),
                "table_bbox": anchor.get("bbox"),
                "table_order_key": anchor.get("order_key"),
                "table_html": anchor.get("table_html"),
                "candidate_child_count": table.get("candidate_child_count", 0),
                "candidates": [],
            }
            for parent in table.get("parents", []):
                for child in parent.get("children", []):
                    review = child.get("codex_review") or {}
                    slim_table["candidates"].append(
                        {
                            "review_key": child.get("review_key"),
                            "candidate_id": parent.get("candidate_id"),
                            "parent_node_id": parent.get("parent_node_id"),
                            "parent_title": parent.get("parent_title"),
                            "parent_page_indices": parent.get(
                                "parent_page_indices"
                            ),
                            "parent_raw_content_indices": parent.get(
                                "parent_raw_content_indices"
                            ),
                            "full_parent_text": parent.get("full_parent_text"),
                            "full_parent_text_sha256": parent.get(
                                "full_parent_text_sha256"
                            ),
                            "child_id": child.get("child_id"),
                            "child_text": child.get("child_text"),
                            "child_text_sha256": child.get("child_text_sha256"),
                            "child_char_start": child.get("char_start"),
                            "child_char_end": child.get("char_end"),
                            "code_selection_score": child.get(
                                "code_selection_score"
                            ),
                            "codex_label": child.get("codex_label"),
                            "codex_semantic_role": review.get("semantic_role"),
                            "codex_citation_support": review.get(
                                "citation_support"
                            ),
                            "codex_rationale": review.get("rationale"),
                            "human_label": child.get("human_label"),
                            "human_rationale": child.get("human_rationale"),
                        }
                    )
                    total_candidates += 1
            slim_document["tables"].append(slim_table)
        slim["documents"].append(slim_document)
    slim["candidate_child_count"] = total_candidates
    slim["document_count"] = len(slim["documents"])
    slim["table_count"] = sum(
        len(document["tables"]) for document in slim["documents"]
    )
    return slim


def render_slim_readable(template: dict[str, Any]) -> str:
    lines = [
        "# Slim Human Annotation Template",
        "",
        "Fill `human_label` with 0 or 1. Captions are table anchors, not reviewed child candidates.",
        "",
    ]
    for document in template["documents"]:
        lines.extend([f"## {document['slug']}", ""])
        for table in document["tables"]:
            lines.extend(
                [
                    f"### {table['table_label']}",
                    "",
                    f"Caption: {table.get('table_caption') or ''}",
                    f"Candidates: {table.get('candidate_child_count', 0)}",
                    "",
                ]
            )
            for candidate in table.get("candidates", []):
                lines.extend(
                    [
                        f"- Review key: `{candidate['review_key']}`",
                        f"  - Role: {candidate.get('codex_semantic_role')}",
                        f"  - Support: {candidate.get('codex_citation_support')}",
                        f"  - Rationale: {candidate.get('codex_rationale')}",
                        f"  - Parent: {candidate.get('full_parent_text')}",
                        f"  - Child: {candidate.get('child_text')}",
                        f"  - Human label: {'' if candidate.get('human_label') is None else candidate['human_label']}",
                        f"  - Human rationale: {candidate.get('human_rationale') or ''}",
                        "",
                    ]
                )
    return "\n".join(lines).rstrip() + "\n"


def single_document_template(
    batch_template: dict[str, Any], document: dict[str, Any]
) -> dict[str, Any]:
    payload = {
        key: copy.deepcopy(value)
        for key, value in batch_template.items()
        if key != "documents"
    }
    payload["documents"] = [copy.deepcopy(document)]
    payload["document_count"] = 1
    payload["table_count"] = len(document["tables"])
    payload["candidate_child_count"] = document["candidate_child_count"]
    payload["tables_with_candidates"] = sum(
        table["candidate_child_count"] > 0 for table in document["tables"]
    )
    return payload


def render_summary(template: dict[str, Any], status: dict[str, Any]) -> str:
    lines = [
        "# Codex Supplementary-Context Review Summary",
        "",
        f"- Reviewed strict candidates: {status['reviewed_children']}",
        f"- Codex label 0 retained: {status['final_label_0_retained']}",
        f"- Codex label 1 demoted: {status['final_label_1_demoted']}",
        f"- Tables with retained candidates: {status['tables_with_candidates']}",
        f"- Existing human gold constraints: {status['human_gold_constraints']}",
        "",
        "## Documents",
        "",
        "| document | retained candidates | tables with candidates |",
        "| --- | ---: | ---: |",
    ]
    for document in template["documents"]:
        tables_with_candidates = sum(
            table["candidate_child_count"] > 0 for table in document["tables"]
        )
        lines.append(
            f"| {document['slug']} | {document['candidate_child_count']} | "
            f"{tables_with_candidates} |"
        )
    lines.extend(
        [
            "",
            "## Policy",
            "",
            "- The code stage preserved all strict candidates.",
            "- Codex applied the supplementary-context and table-scope test.",
            "- Pure table descriptions and off-scope background were demoted.",
            "- Codex labels are provisional and are not human gold.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate and materialize Popo Codex precision-v2 decisions."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--decision-dir", type=Path, default=DEFAULT_DECISION_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--caption-resolution",
        type=Path,
        default=DEFAULT_CAPTION_RESOLUTION,
        help="Optional caption-resolution JSON used to preserve table captions.",
    )
    args = parser.parse_args()

    source = load_json(args.input)
    caption_overrides = load_caption_resolution(args.caption_resolution)
    source = apply_caption_resolution(source, caption_overrides)
    decisions, decision_paths = load_decisions(args.decision_dir)
    decision_source, policy_version = infer_decision_source(decisions)
    expected = {
        child["review_key"]: child for _, _, _, child in iter_children(source)
    }
    missing = sorted(set(expected) - set(decisions))
    unexpected = sorted(set(decisions) - set(expected))
    failures = []
    for key, child in expected.items():
        if key in decisions:
            failures.extend(
                f"{key}: {message}"
                for message in validate_decision(child, decisions[key])
            )
    if missing:
        failures.append(f"Missing {len(missing)} decisions; first: {missing[:5]}")
    if unexpected:
        failures.append(
            f"Unexpected {len(unexpected)} decisions; first: {unexpected[:5]}"
        )
    if failures:
        raise ValueError("\n".join(failures))

    all_reviewed = []
    for document, table, parent, child in iter_children(source):
        decision = decisions[child["review_key"]]
        all_reviewed.append(
            {
                "slug": document["slug"],
                "table_label": table["table_label"],
                "table_anchor": table["table_anchor"],
                "candidate_id": parent["candidate_id"],
                "parent_title": parent.get("parent_title"),
                "full_parent_text": parent["full_parent_text"],
                "child": child,
                "codex_review": decision,
                "final_child_label": decision["codex_label"],
                "final_child_label_source": decision.get(
                    "reaudit_source",
                    "codex_table_scope_precision_review_v2",
                ),
            }
        )

    template = build_retained_template(source, decisions, policy_version)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reviewed_path = (
        args.output_dir / f"all_{len(all_reviewed)}_codex_supplementary_v3_reviewed.json"
    )
    retained_path = args.output_dir / "codex_supplementary_v3_retained_children.json"
    demoted_path = args.output_dir / "codex_supplementary_v3_demoted_children.json"
    template_path = (
        args.output_dir
        / "batch_popo_codex_supplementary_v3_human_child_annotation_template.json"
    )
    full_audit_template_path = (
        args.output_dir
        / "batch_popo_codex_supplementary_v3_full_audit_template.json"
    )
    slim_template_path = (
        args.output_dir
        / "batch_popo_codex_supplementary_v3_human_child_annotation_template.slim.json"
    )
    readable_path = (
        args.output_dir
        / "batch_popo_codex_supplementary_v3_human_child_annotation_template.md"
    )
    slim_readable_path = (
        args.output_dir
        / "batch_popo_codex_supplementary_v3_human_child_annotation_template.slim.md"
    )
    write_json(reviewed_path, all_reviewed)
    write_json(
        retained_path,
        [row for row in all_reviewed if row["final_child_label"] == 0],
    )
    write_json(
        demoted_path,
        [row for row in all_reviewed if row["final_child_label"] == 1],
    )
    write_json(template_path, template)
    write_json(full_audit_template_path, template)
    slim_template = build_slim_annotation_template(
        template,
        reviewed_path,
        retained_path,
        demoted_path,
        full_audit_template_path,
    )
    write_json(slim_template_path, slim_template)
    readable_path.write_text(render_readable(template), encoding="utf-8")
    slim_readable_path.write_text(
        render_slim_readable(slim_template), encoding="utf-8"
    )
    for document in template["documents"]:
        document_template = single_document_template(template, document)
        stem = (
            f"{document['slug']}_popo_codex_supplementary_v3_"
            "human_child_annotation_template"
        )
        write_json(args.output_dir / f"{stem}.json", document_template)
        (args.output_dir / f"{stem}.md").write_text(
            render_readable(document_template),
            encoding="utf-8",
        )
    for document in slim_template["documents"]:
        document_template = {
            key: copy.deepcopy(value)
            for key, value in slim_template.items()
            if key != "documents"
        }
        document_template["documents"] = [copy.deepcopy(document)]
        document_template["document_count"] = 1
        document_template["table_count"] = len(document["tables"])
        document_template["candidate_child_count"] = document[
            "candidate_child_count"
        ]
        stem = (
            f"{document['slug']}_popo_codex_supplementary_v3_"
            "human_child_annotation_template.slim"
        )
        write_json(args.output_dir / f"{stem}.json", document_template)
        (args.output_dir / f"{stem}.md").write_text(
            render_slim_readable(document_template),
            encoding="utf-8",
        )

    counts = Counter(row["final_child_label"] for row in all_reviewed)
    role_counts = Counter(
        row["codex_review"]["semantic_role"] for row in all_reviewed
    )
    status = {
        "schema_version": "2.0",
        "decision_source": decision_source,
        "input_candidates": str(args.input),
        "caption_resolution_input": str(args.caption_resolution),
        "caption_resolution_overrides": len(caption_overrides),
        "decision_files": [str(path) for path in decision_paths],
        "reviewed_output": str(reviewed_path),
        "retained_output": str(retained_path),
        "demoted_output": str(demoted_path),
        "human_template_json": str(template_path),
        "human_template_markdown": str(readable_path),
        "human_template_slim_json": str(slim_template_path),
        "human_template_slim_markdown": str(slim_readable_path),
        "full_audit_template_json": str(full_audit_template_path),
        "reviewed_children": len(all_reviewed),
        "final_label_0_retained": counts[0],
        "final_label_1_demoted": counts[1],
        "tables_with_candidates": template["tables_with_candidates"],
        "semantic_role_counts": dict(sorted(role_counts.items())),
        "human_gold_constraints": sum(
            child.get("human_label") is not None
            for child in expected.values()
        ),
        "reaudited_candidates": sum(
            bool(decision.get("reaudit_source"))
            for decision in decisions.values()
        ),
        "audit_failures": [],
    }
    write_json(args.output_dir / "codex_supplementary_v3_result_status.json", status)
    (
        args.output_dir / "codex_supplementary_v3_result_summary.md"
    ).write_text(render_summary(template, status), encoding="utf-8")
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
