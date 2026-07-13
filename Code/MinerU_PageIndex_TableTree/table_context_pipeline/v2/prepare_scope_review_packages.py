from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_DIR = PROJECT_ROOT / "outputs" / "table_context_pipeline_v2" / "example_run"
DEFAULT_INPUT = DEFAULT_RUN_DIR / "batch_scope_candidate_inventory.json"
DEFAULT_OUTPUT_DIR = DEFAULT_RUN_DIR / "scope_review_packages"

PACKAGE_SCHEMA_VERSION = "scope-review-package-v2"
REVIEW_STAGE = "scope_supplementary_context_review_v2"

ALLOWED_EVIDENCE_ROLES = [
    "experimental_condition",
    "dataset",
    "model",
    "metric",
    "evaluation_protocol",
    "prompt_or_shots",
    "split",
    "baseline",
    "training_setting",
    "limitation",
    "comparability_qualification",
    "other_support",
]

TASK_DEFINITION: dict[str, Any] = {
    "objective": (
        "For each body-text child, retain only table-scoped supplementary "
        "information that is not visible in the table but is needed to "
        "interpret, reproduce, or verify it."
    ),
    "supplementary_context_test": (
        "A child is relevant only when removing it would hide a condition, "
        "assumption, limitation, metric behavior, dataset difference, or "
        "cross-study qualification needed for the named table."
    ),
    "retain": [
        "Datasets, models, metrics, prompts, shots, splits, baselines, and experimental settings scoped to a table.",
        "Necessary limitations, metric behavior, dataset differences, and cross-study comparability qualifications not visible in the table.",
        "Shared conditions may be linked to more than one table when the complete paper context supports every link.",
    ],
    "reject": [
        "Pure table pointers or sentences that only announce, list, show, summarize, or compare a table.",
        "Direct result restatements or trend summaries already visible in the table.",
        "Promotional efficacy or capability claims.",
        "Related method or model innovations outside the table's intended experimental content.",
        "Evidence belonging to another table, figure, dataset, model variant, or experiment.",
    ],
    "caption_policy": (
        "Captions are table-anchor metadata. They provide context but are never child decisions."
    ),
    "label_semantics": {
        "0": "correct/relevant table-child relation",
        "1": "incorrect/irrelevant; represented by omitting the table from relevant_tables",
    },
    "decision_requirements": [
        "Return exactly one decision for every child_id in this package.",
        "relevant_tables may contain multiple table relations or be empty.",
        "Every retained relation has label 0, confidence in [0, 1], an evidence_role, and a concise rationale.",
        "When relevant_tables is empty, provide a rejection_reason.",
        "Do not invent table ids or child ids, and do not return duplicates.",
    ],
}

DECISION_SCHEMA: dict[str, Any] = {
    "schema_version": "scope-review-decisions-v2",
    "review_stage": REVIEW_STAGE,
    "paper_id": "<paper_id>",
    "decisions": [
        {
            "child_id": "<child_id>",
            "parent_id": "<parent_id>",
            "relevant_tables": [
                {
                    "table_id": "<table_id>",
                    "label": 0,
                    "confidence": 0.0,
                    "evidence_role": "<allowed role>",
                    "rationale": "<why this supplements this table>",
                }
            ],
            "rejection_reason": None,
        }
    ],
}

_HISTORY_KEYS = {
    "human_label",
    "human_rationale",
    "gold_label",
    "gold_rationale",
    "historical_label",
    "historical_rationale",
    "existing_human_label",
    "existing_human_rationale",
    "codex_label",
    "codex_review",
    "final_child_label",
    "final_child_label_source",
    "decision_source",
    "decision_source_file",
    "decision",
    "decisions",
    "review_decision",
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _first(mapping: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return default


def _required_id(mapping: dict[str, Any], keys: Iterable[str], context: str) -> str:
    value = _first(mapping, *keys)
    if value is None or not str(value).strip():
        raise ValueError(f"{context}: missing stable id ({', '.join(keys)})")
    return str(value)


def _strip_history(value: Any) -> Any:
    if isinstance(value, dict):
        clean: dict[str, Any] = {}
        for key, item in value.items():
            lowered = str(key).casefold()
            if lowered in _HISTORY_KEYS:
                continue
            if lowered.startswith(("human_", "codex_", "historical_", "gold_")):
                continue
            clean[key] = _strip_history(item)
        return clean
    if isinstance(value, list):
        return [_strip_history(item) for item in value]
    return copy.deepcopy(value)


def paper_id(document: dict[str, Any]) -> str:
    return _required_id(document, ("paper_id", "slug", "document_id"), "document")


def _table_anchor(table: dict[str, Any]) -> dict[str, Any]:
    anchor = _first(table, "table_anchor", "anchor", default={})
    if not isinstance(anchor, dict):
        raise ValueError("table anchor must be an object")
    return _strip_history(anchor)


def normalize_table(table: dict[str, Any], context: str) -> dict[str, Any]:
    anchor = _table_anchor(table)
    table_id = _required_id(
        table,
        ("table_id", "id", "table_label"),
        context,
    )
    label = _first(
        table,
        "table_label",
        "label",
        "canonical_label",
        default=_first(anchor, "canonical_label", "table_label", default=table_id),
    )
    caption = _first(
        table,
        "table_caption",
        "caption",
        default=_first(anchor, "caption", "title", default=""),
    )
    body = _first(
        table,
        "table_body",
        "body",
        "table_html",
        default=_first(anchor, "table_html", "table_code_body", "body", default=""),
    )
    core_keys = {
        "table_id", "id", "table_label", "label", "canonical_label",
        "table_caption", "caption", "table_body", "body", "table_html",
        "table_anchor", "anchor", "page", "page_index", "page_number",
        "page_indices", "bbox", "bboxes", "order_key",
    }
    metadata = _strip_history(
        {key: value for key, value in table.items() if key not in core_keys}
    )
    return {
        "table_id": table_id,
        "table_label": str(label),
        "table_caption": str(caption or ""),
        "table_body": body if body is not None else "",
        "page": _first(table, "page", "page_index", "page_number", "page_indices", default=_first(anchor, "page", "page_index", "page_number")),
        "bbox": _strip_history(_first(table, "bbox", "bboxes", default=anchor.get("bbox"))),
        "order_key": _strip_history(_first(table, "order_key", default=anchor.get("order_key"))),
        "table_anchor": anchor,
        "metadata": metadata,
    }


def _parent_id(parent: dict[str, Any], context: str) -> str:
    return _required_id(
        parent,
        ("parent_id", "id", "parent_node_id", "candidate_id"),
        context,
    )


def _parent_text(parent: dict[str, Any], context: str) -> str:
    text = _first(parent, "full_parent_text", "full_text", "parent_text", "text", "content")
    if text is None and (parent.get("outline_only") or parent.get("content_omitted_reason")):
        return ""
    if not isinstance(text, str):
        raise ValueError(f"{context}: full parent text must be a string")
    return text


def normalize_parent(parent: dict[str, Any], context: str) -> dict[str, Any]:
    parent_id_value = _parent_id(parent, context)
    full_text = _parent_text(parent, context)
    core_keys = {
        "parent_id", "id", "parent_node_id", "candidate_id", "children",
        "full_parent_text", "full_text", "parent_text", "text", "content",
        "full_parent_text_sha256", "full_text_sha256", "title", "parent_title",
        "section_title", "section_id", "scope_id", "page", "page_index",
        "page_number", "page_indices", "parent_page_indices", "bbox", "bboxes",
        "parent_bbox", "source", "source_origin", "source_type",
        "source_block_ids", "mineru_content_indices", "raw_content_indices",
        "parent_raw_content_indices", "block_ids", "popo_node_id",
    }
    metadata = _strip_history(
        {key: value for key, value in parent.items() if key not in core_keys}
    )
    return {
        "parent_id": parent_id_value,
        "full_parent_text": full_text,
        "full_parent_text_sha256": _first(
            parent,
            "full_parent_text_sha256",
            "full_text_sha256",
            default=hashlib.sha256(full_text.encode("utf-8")).hexdigest(),
        ),
        "title": _first(parent, "title", "parent_title", "section_title"),
        "section_id": _first(parent, "section_id", "scope_id"),
        "page": _first(parent, "page", "page_index", "page_number", "page_indices", "parent_page_indices"),
        "bbox": _strip_history(_first(parent, "bbox", "bboxes", "parent_bbox")),
        "source": _strip_history(_first(parent, "source", "source_origin", "source_type")),
        "source_block_ids": _strip_history(
            _first(
                parent,
                "source_block_ids",
                "mineru_content_indices",
                "raw_content_indices",
                "parent_raw_content_indices",
                "block_ids",
                default=[],
            )
        ),
        "popo_node_id": _first(parent, "popo_node_id", "parent_node_id"),
        "metadata": metadata,
    }


def _scope_id(scope: dict[str, Any], context: str) -> str:
    return _required_id(scope, ("scope_id", "id"), context)


def normalize_scope(scope: dict[str, Any], context: str) -> dict[str, Any]:
    normalized = _strip_history(scope)
    normalized.pop("children", None)
    normalized.pop("parents", None)
    normalized["scope_id"] = _scope_id(scope, context)
    return normalized


def _raw_children(document: dict[str, Any], parents: list[dict[str, Any]]) -> list[tuple[dict[str, Any], str | None]]:
    if "children" in document:
        children = document["children"]
        if not isinstance(children, list):
            raise ValueError(f"{paper_id(document)}: children must be a list")
        return [(child, None) for child in children]

    flattened: list[tuple[dict[str, Any], str | None]] = []
    for index, parent in enumerate(parents):
        nested = parent.get("children", [])
        if not isinstance(nested, list):
            raise ValueError(f"{paper_id(document)} parent[{index}]: children must be a list")
        nested_parent_id = _parent_id(parent, f"{paper_id(document)} parent[{index}]")
        flattened.extend((child, nested_parent_id) for child in nested)
    return flattened


def _normalize_suggestions(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        value = [value]
    suggestions: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in value:
        if isinstance(raw, str):
            suggestion: dict[str, Any] = {"table_id": raw}
        elif isinstance(raw, dict):
            suggestion = _strip_history(raw)
            if "table_id" not in suggestion:
                suggested_id = _first(raw, "id", "table", "table_label")
                if suggested_id is not None:
                    suggestion["table_id"] = str(suggested_id)
        else:
            raise ValueError("table suggestion must be a string or object")
        key = json.dumps(suggestion, ensure_ascii=False, sort_keys=True)
        if key not in seen:
            seen.add(key)
            suggestions.append(suggestion)
    return suggestions


def _child_scope_ids(child: dict[str, Any]) -> list[str]:
    values = child.get("scope_ids")
    if values is None:
        single = child.get("scope_id")
        values = [] if single is None else [single]
    elif not isinstance(values, list):
        values = [values]
    return [str(value) for value in values]


def normalize_document(document: dict[str, Any]) -> dict[str, Any]:
    doc_id = paper_id(document)

    raw_tables = document.get("tables")
    raw_parents = document.get("parents")
    raw_scopes = document.get("scopes", [])
    if not isinstance(raw_tables, list) or not isinstance(raw_parents, list):
        raise ValueError(f"{doc_id}: tables and parents must be lists")
    if not isinstance(raw_scopes, list):
        raise ValueError(f"{doc_id}: scopes must be a list")

    tables = [normalize_table(table, f"{doc_id} table[{index}]") for index, table in enumerate(raw_tables)]
    table_ids = [table["table_id"] for table in tables]
    if len(table_ids) != len(set(table_ids)):
        raise ValueError(f"{doc_id}: duplicate table_id")

    parents = [normalize_parent(parent, f"{doc_id} parent[{index}]") for index, parent in enumerate(raw_parents)]
    parent_by_id = {parent["parent_id"]: parent for parent in parents}
    if len(parent_by_id) != len(parents):
        raise ValueError(f"{doc_id}: duplicate parent_id")

    scopes = [normalize_scope(scope, f"{doc_id} scope[{index}]") for index, scope in enumerate(raw_scopes)]
    scope_by_id = {scope["scope_id"]: scope for scope in scopes}
    if len(scope_by_id) != len(scopes):
        raise ValueError(f"{doc_id}: duplicate scope_id")

    review_eligible = document.get("review_eligible_table_ids", table_ids)
    if not isinstance(review_eligible, list):
        raise ValueError(f"{doc_id}: review_eligible_table_ids must be a list")
    review_eligible = [str(value) for value in review_eligible]
    if len(review_eligible) != len(set(review_eligible)) or set(review_eligible) != set(table_ids):
        raise ValueError(
            f"{doc_id}: review_eligible_table_ids must list every table exactly once"
        )

    review_items: list[dict[str, Any]] = []
    child_ids: set[str] = set()
    for index, (child, nested_parent_id) in enumerate(_raw_children(document, raw_parents)):
        context = f"{doc_id} child[{index}]"
        if not isinstance(child, dict):
            raise ValueError(f"{context}: child must be an object")
        child_id = _required_id(child, ("child_id", "id"), context)
        if child_id in child_ids:
            raise ValueError(f"{doc_id}: duplicate child_id {child_id}")
        child_ids.add(child_id)
        parent_id_value = str(_first(child, "parent_id", default=nested_parent_id) or "")
        if parent_id_value not in parent_by_id:
            raise ValueError(f"{context}: unknown parent_id {parent_id_value!r}")
        parent = parent_by_id[parent_id_value]
        child_text = _first(child, "child_text", "text", "content")
        if not isinstance(child_text, str):
            raise ValueError(f"{context}: child text must be a string")
        offsets = child.get("offsets") if isinstance(child.get("offsets"), dict) else {}
        char_start = _first(child, "char_start", "start", default=offsets.get("start"))
        char_end = _first(child, "char_end", "end", default=offsets.get("end"))
        if (
            isinstance(char_start, bool)
            or isinstance(char_end, bool)
            or not isinstance(char_start, int)
            or not isinstance(char_end, int)
            or char_start < 0
            or char_end < char_start
            or char_end > len(parent["full_parent_text"])
        ):
            raise ValueError(f"{context}: invalid child offsets")
        if parent["full_parent_text"][char_start:char_end] != child_text:
            raise ValueError(f"{context}: child text does not match parent offsets")

        scope_ids = _child_scope_ids(child)
        unknown_scopes = sorted(set(scope_ids) - set(scope_by_id))
        if unknown_scopes:
            raise ValueError(f"{context}: unknown scope ids {unknown_scopes}")
        resolved_scopes = [copy.deepcopy(scope_by_id[scope_id]) for scope_id in scope_ids]
        suggestions = _normalize_suggestions(
            _first(child, "table_suggestions", "suggested_tables", "suggested_table_ids", "suggestions")
        )
        if not suggestions:
            for scope in resolved_scopes:
                suggestions.extend(
                    _normalize_suggestions(
                        _first(scope, "table_suggestions", "suggested_tables", "suggested_table_ids", "suggestions")
                    )
                )
            suggestions = _normalize_suggestions(suggestions)

        page = _first(child, "page", "page_index", "page_number", "page_indices", default=parent["page"])
        bbox = _strip_history(_first(child, "bbox", "bboxes", default=parent["bbox"]))
        source = _strip_history(_first(child, "source", "source_origin", "source_type", default=parent["source"]))
        core_child_keys = {
            "child_id", "id", "parent_id", "child_text", "text", "content",
            "child_text_sha256", "text_sha256", "char_start", "start",
            "char_end", "end", "offsets", "scope_id", "scope_ids", "page",
            "page_index", "page_number", "page_indices", "bbox", "bboxes",
            "source", "source_origin", "source_type", "table_suggestions",
            "suggested_tables", "suggested_table_ids", "suggestions",
        }
        metadata = _strip_history(
            {key: value for key, value in child.items() if key not in core_child_keys}
        )
        review_items.append(
            {
                "child_id": child_id,
                "parent_id": parent_id_value,
                "full_parent_text": parent["full_parent_text"],
                "full_parent_text_sha256": parent["full_parent_text_sha256"],
                "child_text": child_text,
                "child_text_sha256": hashlib.sha256(child_text.encode("utf-8")).hexdigest(),
                "char_start": char_start,
                "char_end": char_end,
                "page": page,
                "bbox": bbox,
                "source": source,
                "scope_ids": scope_ids,
                "scopes": resolved_scopes,
                "table_suggestions": suggestions,
                "candidate_metadata": metadata,
            }
        )

    return {
        "paper_id": doc_id,
        "title": _first(document, "title", "paper_title"),
        "tables": tables,
        "review_eligible_table_ids": review_eligible,
        "parents": parents,
        "scopes": scopes,
        "review_items": review_items,
    }


def build_document_package(source: dict[str, Any], document: dict[str, Any]) -> dict[str, Any]:
    anchor_audit = document.get("table_anchor_audit")
    if isinstance(anchor_audit, dict) and anchor_audit.get("status") != "pass":
        raise ValueError(f"{paper_id(document)}: table anchor audit did not pass")
    normalized = normalize_document(document)
    document_core_keys = {
        "paper_id", "slug", "document_id", "title", "paper_title", "tables",
        "review_eligible_table_ids", "scopes", "parents", "children",
    }
    inventory_metadata = _strip_history(
        {key: value for key, value in document.items() if key not in document_core_keys}
    )
    package = {
        "schema_version": PACKAGE_SCHEMA_VERSION,
        "review_stage": REVIEW_STAGE,
        "source_inventory_schema_version": source.get("schema_version"),
        "paper_id": normalized["paper_id"],
        "title": normalized["title"],
        "inventory_metadata": inventory_metadata,
        "task_definition": copy.deepcopy(TASK_DEFINITION),
        "allowed_evidence_roles": list(ALLOWED_EVIDENCE_ROLES),
        "decision_schema": copy.deepcopy(DECISION_SCHEMA),
        "table_count": len(normalized["tables"]),
        "review_item_count": len(normalized["review_items"]),
        "tables": normalized["tables"],
        "review_eligible_table_ids": normalized["review_eligible_table_ids"],
        "scopes": normalized["scopes"],
        "parents": normalized["parents"],
        "review_items": normalized["review_items"],
    }
    package["decision_schema"]["paper_id"] = normalized["paper_id"]
    return package


def _markdown_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def render_markdown(package: dict[str, Any]) -> str:
    task = package["task_definition"]
    lines = [
        f"# Scope Review Package: {package['paper_id']}",
        "",
        f"- Tables: {package['table_count']}",
        f"- Unique child review items: {package['review_item_count']}",
        "- Labels: 0 = relevant/correct; 1 = irrelevant/incorrect",
        "- Captions are anchors and are not decision items.",
        "",
        "## Task",
        "",
        task["objective"],
        "",
        task["supplementary_context_test"],
        "",
        "### Retain",
        "",
    ]
    lines.extend(f"- {item}" for item in task["retain"])
    lines.extend(["", "### Reject", ""])
    lines.extend(f"- {item}" for item in task["reject"])
    lines.extend(["", "## Table Anchors", ""])
    for table in package["tables"]:
        lines.extend(
            [
                f"### {table['table_label']} (`{table['table_id']}`)",
                "",
                f"- Page: {_markdown_value(table.get('page'))}",
                f"- BBox: {_markdown_value(table.get('bbox'))}",
                f"- Caption: {table.get('table_caption') or ''}",
                "",
                "Table body:",
                "",
                _markdown_value(table.get("table_body")),
                "",
            ]
        )

    lines.extend(["## Unique Child Review Items", ""])
    for index, item in enumerate(package["review_items"], 1):
        suggestions = [suggestion.get("table_id") for suggestion in item["table_suggestions"]]
        lines.extend(
            [
                f"### {index}. `{item['child_id']}`",
                "",
                f"- Parent: `{item['parent_id']}`",
                f"- Scope ids: {_markdown_value(item['scope_ids'])}",
                f"- Non-binding table suggestions: {_markdown_value(suggestions)}",
                f"- Page: {_markdown_value(item.get('page'))}",
                f"- BBox: {_markdown_value(item.get('bbox'))}",
                f"- Source: {_markdown_value(item.get('source'))}",
                f"- Offsets: [{item['char_start']}, {item['char_end']})",
                "",
                "Full parent:",
                "",
                item["full_parent_text"],
                "",
                "Child:",
                "",
                item["child_text"],
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def safe_stem(value: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return stem or hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def prepare_review_packages(input_path: Path, output_dir: Path) -> dict[str, Any]:
    source = load_json(input_path)
    documents = source.get("documents") if isinstance(source, dict) else None
    if not isinstance(documents, list):
        raise ValueError("inventory must contain a documents list")
    if source.get("status") not in (None, "pass"):
        raise ValueError("inventory status must be 'pass' before package publication")

    seen_papers: set[str] = set()
    seen_stems: set[str] = set()
    total_items = 0
    prepared: list[tuple[str, str, dict[str, Any]]] = []
    for document in documents:
        package = build_document_package(source, document)
        doc_id = package["paper_id"]
        stem = safe_stem(doc_id)
        if doc_id in seen_papers or stem in seen_stems:
            raise ValueError(f"duplicate paper id or output stem: {doc_id}")
        seen_papers.add(doc_id)
        seen_stems.add(stem)
        prepared.append((doc_id, stem, package))
        total_items += package["review_item_count"]

    expected_total = source.get("candidate_child_count")
    if expected_total is not None and expected_total != total_items:
        raise ValueError(
            f"prepared {total_items} children, inventory declares {expected_total}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    status_documents: list[dict[str, Any]] = []
    for doc_id, stem, package in prepared:
        json_path = output_dir / f"{stem}_scope_review_package.json"
        markdown_path = output_dir / f"{stem}_scope_review_package.md"
        decision_path = output_dir / f"{stem}_scope_review_decisions.json"
        write_json(json_path, package)
        markdown_path.write_text(render_markdown(package), encoding="utf-8")
        status_documents.append(
            {
                "paper_id": doc_id,
                "table_count": package["table_count"],
                "review_item_count": package["review_item_count"],
                "package_json": str(json_path),
                "package_markdown": str(markdown_path),
                "expected_decisions": str(decision_path),
            }
        )
    status = {
        "schema_version": "scope-review-package-status-v2",
        "review_stage": REVIEW_STAGE,
        "source_inventory": str(input_path),
        "output_dir": str(output_dir),
        "document_count": len(status_documents),
        "review_item_count": total_items,
        "documents": status_documents,
    }
    write_json(output_dir / "scope_review_package_status.json", status)
    return status


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create one scope-aware subagent review package JSON and Markdown file per paper."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to batch_scope_candidate_inventory.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for per-paper review packages and package status.",
    )
    args = parser.parse_args()
    status = prepare_review_packages(args.input, args.output_dir)
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
