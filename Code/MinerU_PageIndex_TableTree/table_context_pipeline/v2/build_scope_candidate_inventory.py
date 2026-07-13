#!/usr/bin/env python3
"""Build complete scope-aware candidate inventories from MinerU and Popo."""

from __future__ import annotations

import argparse
import copy
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from table_context_pipeline.v2.common import (  # noqa: E402
    REFERENCE_TITLE_RE,
    apply_anchor_overrides,
    build_table_suggestions,
    extract_popo_nodes,
    is_reference_path,
    lexical_tokens,
    load_json,
    make_children,
    match_table_anchors,
    normalize_text,
    resolve_workspace_path,
    sha256_file,
    sha256_text,
    stable_scope_id,
    workspace_relative,
    write_json,
)


SCHEMA_VERSION = "2.0"
POLICY_VERSION = "scope-candidate-inventory-v2"
BATCH_FILENAME = "batch_scope_candidate_inventory.json"
STATUS_FILENAME = "scope_candidate_inventory_status.json"


class DocumentBuildError(RuntimeError):
    def __init__(self, reason: str, details: Any = None):
        super().__init__(reason)
        self.reason = reason
        self.details = details


def _source_specifications(
    config: dict[str, Any], workspace_root: Path
) -> list[dict[str, Any]]:
    specifications = []
    if config.get("sources"):
        for index, source in enumerate(config["sources"], start=1):
            specifications.append(
                {
                    "name": source.get("name") or f"source_{index}",
                    "anchor_manifest": resolve_workspace_path(
                        source["anchor_manifest"], workspace_root
                    ),
                    "popo_tree_dir": resolve_workspace_path(
                        source["popo_tree_dir"], workspace_root
                    ),
                }
            )
        return specifications

    for key, value in config.items():
        if not key.endswith("_anchor_manifest"):
            continue
        prefix = key[: -len("_anchor_manifest")]
        tree_key = f"{prefix}_popo_tree_dir"
        if tree_key not in config:
            raise ValueError(f"Missing config key paired with {key}: {tree_key}")
        specifications.append(
            {
                "name": prefix or "default",
                "anchor_manifest": resolve_workspace_path(value, workspace_root),
                "popo_tree_dir": resolve_workspace_path(
                    config[tree_key], workspace_root
                ),
            }
        )
    if not specifications:
        raise ValueError("Config contains no anchor manifest / Popo tree source pairs")
    return specifications


def _document_sources(
    config: dict[str, Any], workspace_root: Path
) -> list[dict[str, Any]]:
    rows = []
    seen_slugs: set[str] = set()
    for specification in _source_specifications(config, workspace_root):
        manifest_path = specification["anchor_manifest"]
        manifest = load_json(manifest_path)
        for document in manifest.get("documents") or []:
            slug = document.get("slug")
            if not slug:
                raise ValueError(f"Document without slug in {manifest_path}")
            if slug in seen_slugs:
                raise ValueError(f"Duplicate document slug across manifests: {slug}")
            seen_slugs.add(slug)
            rows.append(
                {
                    "source_name": specification["name"],
                    "manifest_path": manifest_path,
                    "tree_path": specification["popo_tree_dir"] / f"{slug}.json",
                    "document": document,
                }
            )
    return rows


def _anchor_list(document: dict[str, Any]) -> list[dict[str, Any]]:
    anchors = []
    for index, table in enumerate(document.get("tables") or [], start=1):
        anchor = table.get("table_anchor")
        if not isinstance(anchor, dict):
            raise DocumentBuildError(
                "missing_table_anchor", {"table_index": index - 1}
            )
        anchors.append(copy.deepcopy(anchor))
    return anchors


def _relative_optional_path(value: Any, workspace_root: Path) -> Any:
    if not value:
        return value
    try:
        return workspace_relative(resolve_workspace_path(value, workspace_root), workspace_root)
    except (ValueError, OSError):
        return None


def _compact_anchor(anchor: dict[str, Any], workspace_root: Path) -> dict[str, Any]:
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
        "table_footnote": anchor.get("table_footnote") or [],
        "image_path": _relative_optional_path(anchor.get("image_path"), workspace_root),
    }


def _section_path(node: dict[str, Any], include_title: bool = True) -> list[str]:
    result = [str(value) for value in node.get("ancestor_titles") or [] if str(value).strip()]
    title = str(node.get("title") or "").strip()
    if include_title and title and (not result or normalize_text(result[-1]) != normalize_text(title)):
        result.append(title)
    return result


def _locations(node: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    for location in node.get("location") or []:
        if not isinstance(location, dict):
            continue
        result.append(
            {"page": location.get("page"), "bbox": location.get("bbox")}
        )
    return result


def _table_records(
    anchors: list[dict[str, Any]],
    table_nodes: list[dict[str, Any]],
    match_audit: dict[str, Any],
    workspace_root: Path,
) -> list[dict[str, Any]]:
    records = []
    table_ids: set[str] = set()
    for match in match_audit["matches"]:
        anchor_index = match["anchor_index"]
        anchor = anchors[anchor_index]
        node = table_nodes[match["popo_table_index"]]
        table_id = str(anchor.get("table_node_id") or f"table-{anchor_index + 1:04d}")
        if table_id in table_ids:
            raise DocumentBuildError(
                "duplicate_table_id", {"table_id": table_id}
            )
        table_ids.add(table_id)
        section_path = _section_path(node, include_title=True)
        compact_anchor = _compact_anchor(anchor, workspace_root)
        records.append(
            {
                "table_id": table_id,
                "canonical_label": anchor.get("canonical_label"),
                "caption": anchor.get("caption"),
                "table_body": anchor.get("table_html") or anchor.get("table_code_body"),
                "anchor": compact_anchor,
                "popo_node_id": node.get("node_id"),
                "popo_block_ids": node.get("block_ids") or [],
                "page_indices": node.get("page_indices") or [],
                "locations": _locations(node),
                "section_path": section_path,
                "scope_id": stable_scope_id(section_path),
                "anchor_match": match,
                "lexical_tokens": sorted(
                    lexical_tokens(
                        " ".join(
                            str(value or "")
                            for value in (
                                anchor.get("caption"),
                                anchor.get("table_html"),
                                anchor.get("table_code_body"),
                            )
                        )
                    )
                ),
            }
        )
    return records


def _popo_parents(text_nodes: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    parents = []
    children = []
    for node in text_nodes:
        section_path = _section_path(node, include_title=True)
        outline_only = is_reference_path(section_path)
        full_text = str(node.get("content") or "")
        parent = {
            "parent_id": node["node_id"],
            "source_origin": "popo_text",
            "title": node.get("title") or "",
            "section_path": section_path,
            "scope_id": stable_scope_id(section_path),
            "page_indices": node.get("page_indices") or [],
            "page_start": node.get("page_start"),
            "page_end": node.get("page_end"),
            "locations": _locations(node),
            "bboxes": [row["bbox"] for row in _locations(node) if row.get("bbox") is not None],
            "block_ids": node.get("block_ids") or [],
            "mineru_content_indices": [],
            "outline_only": outline_only,
            "full_text": None if outline_only else full_text,
            "full_text_sha256": sha256_text(full_text),
            "content_omitted_reason": "reference_outline_only" if outline_only else None,
            "child_ids": [],
        }
        if not outline_only and full_text.strip():
            parent_children = make_children(parent["parent_id"], full_text)
            parent["child_ids"] = [child["child_id"] for child in parent_children]
            children.extend(parent_children)
        parents.append(parent)
    return parents, children


def _raw_page(block: dict[str, Any]) -> int | None:
    try:
        if block.get("page_idx") is not None:
            return int(block["page_idx"]) + 1
        if block.get("page_index") is not None:
            return int(block["page_index"])
    except (TypeError, ValueError):
        return None
    return None


def _raw_block_ids(block: dict[str, Any]) -> list[int]:
    values: list[Any] = []
    if block.get("block_id") is not None:
        values.append(block["block_id"])
    values.extend(block.get("block_ids") or [])
    if block.get("content_index") is not None:
        values.append(block["content_index"])
    result = []
    for value in values:
        try:
            integer = int(value)
        except (TypeError, ValueError):
            continue
        if integer not in result:
            result.append(integer)
    return result


def _content_text(block: dict[str, Any]) -> str | None:
    """Return reviewable prose from supported MinerU content types."""
    content_type = str(block.get("type") or "")
    if content_type in {"text", "page_footnote"}:
        return str(block.get("text") or "")
    if content_type == "list":
        items = block.get("list_items") or []
        if not isinstance(items, list):
            return ""
        return "\n".join(
            str(item or "").strip()
            for item in items
            if str(item or "").strip()
        )
    return None


def _normalized_bbox(bbox: Any) -> list[float] | None:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        values = [float(value) for value in bbox]
    except (TypeError, ValueError):
        return None
    if max((abs(value) for value in values), default=0) > 1.5:
        values = [value / 1000.0 for value in values]
    return values


def _reference_boundary(text_nodes: list[dict[str, Any]]) -> tuple[int, float, int] | None:
    locations = []
    for node in text_nodes:
        if not is_reference_path(_section_path(node, include_title=True)):
            continue
        for location in node.get("location") or []:
            bbox = _normalized_bbox(location.get("bbox"))
            try:
                page = int(location.get("page"))
            except (TypeError, ValueError):
                continue
            locations.append((page, bbox[1] if bbox else 0.0))
    if not locations:
        return None
    first_page, first_y = min(locations)
    return first_page, first_y, max(page for page, _ in locations)


def _is_in_reference_area(
    block: dict[str, Any], text: str, boundary: tuple[int, float, int] | None
) -> bool:
    if REFERENCE_TITLE_RE.search(text or ""):
        return True
    if boundary is None:
        return False
    page = _raw_page(block)
    if page is None:
        return False
    first_page, first_y, last_page = boundary
    if not first_page <= page <= last_page:
        return False
    if page > first_page:
        return True
    bbox = _normalized_bbox(block.get("bbox"))
    return bbox is None or bbox[1] >= first_y - 0.02


def _is_known_caption(text: str, anchors: list[dict[str, Any]]) -> bool:
    normalized = normalize_text(text)
    if not normalized:
        return False
    for anchor in anchors:
        caption = normalize_text(anchor.get("caption") or "")
        label = normalize_text(anchor.get("canonical_label") or "")
        if caption and (normalized in caption or caption in normalized):
            return True
        if label and normalized == label:
            return True
    return False


def _parent_order_key(parent: dict[str, Any]) -> tuple[float, float, str]:
    page = min(parent.get("page_indices") or [10**9])
    y_values = []
    for bbox in parent.get("bboxes") or []:
        normalized = _normalized_bbox(bbox)
        if normalized:
            y_values.append(normalized[1])
    return float(page), min(y_values or [0.0]), parent["parent_id"]


def _infer_orphan_section(
    page: int | None, bbox: Any, popo_parents: list[dict[str, Any]]
) -> list[str]:
    candidates = [parent for parent in popo_parents if not parent["outline_only"]]
    if not candidates:
        return []
    target = (float(page if page is not None else 10**9), (_normalized_bbox(bbox) or [0, 0, 0, 0])[1])
    preceding = [parent for parent in candidates if _parent_order_key(parent)[:2] <= target]
    chosen = max(preceding, key=_parent_order_key) if preceding else min(candidates, key=_parent_order_key)
    return list(chosen.get("section_path") or [])


def _merge_mineru_orphans(
    content_list: list[dict[str, Any]],
    text_nodes: list[dict[str, Any]],
    anchors: list[dict[str, Any]],
    popo_parents: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    parents = []
    children = []
    audit_rows = []
    popo_block_to_parent: dict[int, list[str]] = defaultdict(list)
    popo_texts = []
    for node in text_nodes:
        combined = "\n".join(
            value for value in (str(node.get("title") or ""), str(node.get("content") or "")) if value
        )
        normalized = normalize_text(combined)
        if normalized:
            popo_texts.append((node["node_id"], normalized))
        for block_id in node.get("block_ids") or []:
            popo_block_to_parent[int(block_id)].append(node["node_id"])
    reference_boundary = _reference_boundary(text_nodes)

    for content_index, block in enumerate(content_list):
        text = _content_text(block)
        if text is None:
            continue
        normalized = normalize_text(text)
        raw_ids = _raw_block_ids(block)
        covered_ids = sorted(
            {
                parent_id
                for block_id in raw_ids
                for parent_id in popo_block_to_parent.get(block_id, [])
            }
        )
        contained_ids = [parent_id for parent_id, popo_text in popo_texts if normalized and normalized in popo_text]
        if not normalized:
            status = "excluded_empty_text"
        elif _is_known_caption(text, anchors):
            status = "excluded_known_table_caption"
        elif _is_in_reference_area(block, text, reference_boundary):
            status = "excluded_reference_area"
        elif covered_ids:
            status = "covered_by_popo_block_id"
        elif contained_ids:
            status = "covered_by_popo_content"
        else:
            status = "mineru_orphan_fallback"

        page = _raw_page(block)
        audit_rows.append(
            {
                "content_index": content_index,
                "content_type": block.get("type"),
                "page_index": page,
                "bbox": block.get("bbox"),
                "block_ids": raw_ids,
                "text_sha256": sha256_text(text),
                "char_count": len(text),
                "status": status,
                "matched_popo_parent_ids": sorted(set(covered_ids + contained_ids)),
            }
        )
        if status != "mineru_orphan_fallback":
            continue

        parent_id = f"mineru-orphan-{content_index:05d}"
        section_path = _infer_orphan_section(page, block.get("bbox"), popo_parents)
        orphan_parent = {
            "parent_id": parent_id,
            "source_origin": "mineru_orphan_fallback",
            "mineru_content_type": block.get("type"),
            "title": "",
            "section_path": section_path,
            "scope_id": stable_scope_id(section_path),
            "page_indices": [page] if page is not None else [],
            "page_start": page,
            "page_end": page,
            "locations": [{"page": page, "bbox": block.get("bbox")}],
            "bboxes": [block.get("bbox")] if block.get("bbox") is not None else [],
            "block_ids": raw_ids,
            "mineru_content_indices": [content_index],
            "outline_only": False,
            "full_text": text,
            "full_text_sha256": sha256_text(text),
            "content_omitted_reason": None,
            "child_ids": [],
        }
        orphan_children = make_children(parent_id, text)
        orphan_parent["child_ids"] = [child["child_id"] for child in orphan_children]
        parents.append(orphan_parent)
        children.extend(orphan_children)

    status_counts = Counter(row["status"] for row in audit_rows)
    body_statuses = {
        "covered_by_popo_block_id",
        "covered_by_popo_content",
        "mineru_orphan_fallback",
    }
    body_rows = [row for row in audit_rows if row["status"] in body_statuses]
    represented = [row for row in body_rows if row["status"] in body_statuses]
    coverage = {
        "mineru_text_block_count": len(audit_rows),
        "eligible_body_text_block_count": len(body_rows),
        "represented_body_text_block_count": len(represented),
        "represented_body_text_ratio": round(len(represented) / len(body_rows), 6) if body_rows else 1.0,
        "popo_text_parent_count": sum(1 for parent in popo_parents if not parent["outline_only"]),
        "popo_reference_outline_count": sum(1 for parent in popo_parents if parent["outline_only"]),
        "mineru_orphan_parent_count": len(parents),
        "status_counts": dict(sorted(status_counts.items())),
        "mineru_text_block_audit": audit_rows,
    }
    return parents, children, coverage


def _scope_records(
    parents: list[dict[str, Any]], tables: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    scopes: dict[str, dict[str, Any]] = {}
    for item_type, rows in (("parent_ids", parents), ("table_ids", tables)):
        id_field = "parent_id" if item_type == "parent_ids" else "table_id"
        for row in rows:
            scope_id = row["scope_id"]
            record = scopes.setdefault(
                scope_id,
                {
                    "scope_id": scope_id,
                    "section_path": row.get("section_path") or [],
                    "parent_ids": [],
                    "table_ids": [],
                },
            )
            record[item_type].append(row[id_field])
    return sorted(scopes.values(), key=lambda row: (row["section_path"], row["scope_id"]))


def _decorate_children(
    children: list[dict[str, Any]],
    parents: list[dict[str, Any]],
    tables: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    parent_lookup = {parent["parent_id"]: parent for parent in parents}
    all_table_ids = [table["table_id"] for table in tables]
    result = []
    for child in children:
        parent = parent_lookup[child["parent_id"]]
        suggestions, mentions, hints = build_table_suggestions(child, parent, tables)
        result.append(
            {
                **child,
                "scope_id": parent["scope_id"],
                "source_origin": parent["source_origin"],
                "review_eligible_table_ids": list(all_table_ids),
                "structured_table_references": mentions,
                "condition_hints": hints,
                "table_suggestions": suggestions,
                "suggestions_are_binding": False,
            }
        )
    return result


def build_document_inventory(
    document: dict[str, Any],
    tree_path: Path,
    mineru_path: Path,
    manifest_path: Path,
    overrides: Iterable[dict[str, Any]],
    overrides_path: Path | None,
    workspace_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    slug = document["slug"]
    for required_path, reason in (
        (tree_path, "missing_popo_tree"),
        (mineru_path, "missing_mineru_content_list"),
        (manifest_path, "missing_anchor_manifest"),
    ):
        if not required_path.exists():
            raise DocumentBuildError(
                reason,
                {"path": workspace_relative(required_path, workspace_root)},
            )

    tree = load_json(tree_path)
    content_list = load_json(mineru_path)
    if not isinstance(content_list, list):
        raise DocumentBuildError("invalid_mineru_content_list", {"expected": "list"})
    text_nodes, table_nodes = extract_popo_nodes(tree, slug)
    anchors, override_audit = apply_anchor_overrides(
        slug, _anchor_list(document), overrides
    )
    if override_audit["status"] != "pass":
        raise DocumentBuildError("anchor_override_failed", override_audit)
    anchor_match_audit = match_table_anchors(anchors, table_nodes)
    if anchor_match_audit["status"] != "pass":
        raise DocumentBuildError(
            "table_anchor_match_failed",
            {
                "override_audit": override_audit,
                "anchor_match_audit": anchor_match_audit,
            },
        )

    tables = _table_records(
        anchors, table_nodes, anchor_match_audit, workspace_root
    )
    popo_parents, popo_children = _popo_parents(text_nodes)
    orphan_parents, orphan_children, source_coverage = _merge_mineru_orphans(
        content_list, text_nodes, anchors, popo_parents
    )
    parents = popo_parents + orphan_parents
    children = _decorate_children(
        popo_children + orphan_children, parents, tables
    )
    parent_lookup = {parent["parent_id"]: parent for parent in parents}
    child_ids = [child["child_id"] for child in children]
    if len(child_ids) != len(set(child_ids)):
        raise DocumentBuildError("duplicate_child_id")
    for child in children:
        parent = parent_lookup[child["parent_id"]]
        if child["text"] != parent["full_text"][child["char_start"] : child["char_end"]]:
            raise DocumentBuildError(
                "child_offset_mismatch", {"child_id": child["child_id"]}
            )

    source_hashes = {
        "anchor_manifest": {
            "path": workspace_relative(manifest_path, workspace_root),
            "sha256": sha256_file(manifest_path),
        },
        "popo_tree": {
            "path": workspace_relative(tree_path, workspace_root),
            "sha256": sha256_file(tree_path),
        },
        "mineru_content_list": {
            "path": workspace_relative(mineru_path, workspace_root),
            "sha256": sha256_file(mineru_path),
        },
    }
    if overrides_path is not None:
        source_hashes["anchor_overrides"] = {
            "path": workspace_relative(overrides_path, workspace_root),
            "sha256": sha256_file(overrides_path),
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "candidate_policy_version": POLICY_VERSION,
        "paper_id": slug,
        "review_eligible_table_ids": [table["table_id"] for table in tables],
        "policy": {
            "source_inventory": "popo_text_plus_mineru_orphan_fallback",
            "references": "outline_only",
            "suggestion_role": "ordering_only_non_binding",
            "score_deletes_candidates": False,
            "pre_review_cap": None,
            "child_storage": "unique_per_paper",
        },
        "source_hashes": source_hashes,
        "source_coverage": source_coverage,
        "table_anchor_audit": {
            "status": "pass",
            "override_audit": override_audit,
            "match_audit": anchor_match_audit,
        },
        "tables": tables,
        "scopes": _scope_records(parents, tables),
        "parents": parents,
        "children": children,
        "counts": {
            "tables": len(tables),
            "scopes": len(_scope_records(parents, tables)),
            "parents": len(parents),
            "popo_parents": len(popo_parents),
            "mineru_orphan_parents": len(orphan_parents),
            "reference_outline_parents": sum(1 for parent in parents if parent["outline_only"]),
            "unique_children": len(children),
            "table_suggestions": sum(len(child["table_suggestions"]) for child in children),
        },
    }


def build_from_config(
    config_path: Path,
    workspace_root: Path = PROJECT_ROOT,
    publish: bool = True,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    workspace_root = workspace_root.resolve()
    config_path = resolve_workspace_path(config_path, workspace_root)
    config = load_json(config_path)
    output_dir = resolve_workspace_path(config["output_dir"], workspace_root)
    inventory_dir = output_dir / "inventory"
    status_path = output_dir / STATUS_FILENAME
    batch_path = output_dir / BATCH_FILENAME

    overrides_path = None
    overrides: list[dict[str, Any]] = []
    if config.get("anchor_overrides"):
        overrides_path = resolve_workspace_path(
            config["anchor_overrides"], workspace_root
        )
        overrides = list((load_json(overrides_path).get("overrides") or []))

    statuses = []
    inventories = []
    document_sources = _document_sources(config, workspace_root)
    for source in document_sources:
        document = source["document"]
        slug = document["slug"]
        try:
            mineru_path = resolve_workspace_path(
                document["mineru_source"], workspace_root
            )
            inventory = build_document_inventory(
                document=document,
                tree_path=source["tree_path"],
                mineru_path=mineru_path,
                manifest_path=source["manifest_path"],
                overrides=overrides,
                overrides_path=overrides_path,
                workspace_root=workspace_root,
            )
            inventories.append(inventory)
            statuses.append(
                {
                    "paper_id": slug,
                    "status": "pass",
                    "inventory_path": workspace_relative(
                        inventory_dir / f"{slug}.json", workspace_root
                    ),
                    "table_count": inventory["counts"]["tables"],
                    "parent_count": inventory["counts"]["parents"],
                    "child_count": inventory["counts"]["unique_children"],
                    "orphan_parent_count": inventory["counts"]["mineru_orphan_parents"],
                    "anchor_audit": inventory["table_anchor_audit"],
                }
            )
        except (DocumentBuildError, FileNotFoundError, KeyError, ValueError) as error:
            reason = error.reason if isinstance(error, DocumentBuildError) else type(error).__name__
            details = (
                error.details
                if isinstance(error, DocumentBuildError)
                else {"message": "document input could not be resolved or validated"}
            )
            statuses.append(
                {
                    "paper_id": slug,
                    "status": "fail",
                    "reason": reason,
                    "details": details,
                    "inventory_path": None,
                }
            )

    all_passed = len(inventories) == len(document_sources) and all(
        row["status"] == "pass" for row in statuses
    )
    status = {
        "schema_version": SCHEMA_VERSION,
        "candidate_policy_version": POLICY_VERSION,
        "status": "pass" if all_passed else "fail",
        "publish_policy": "all_documents_must_pass_before_inventory_publication",
        "config": {
            "path": workspace_relative(config_path, workspace_root),
            "sha256": sha256_file(config_path),
        },
        "document_count": len(document_sources),
        "passed_document_count": sum(row["status"] == "pass" for row in statuses),
        "failed_document_count": sum(row["status"] == "fail" for row in statuses),
        "documents": statuses,
        "batch_inventory_path": workspace_relative(batch_path, workspace_root) if all_passed else None,
    }

    batch = None
    if all_passed:
        batch = {
            "schema_version": SCHEMA_VERSION,
            "candidate_policy_version": POLICY_VERSION,
            "status": "pass",
            "config_sha256": status["config"]["sha256"],
            "document_count": len(inventories),
            "review_eligible_table_ids_by_paper": {
                inventory["paper_id"]: inventory["review_eligible_table_ids"]
                for inventory in inventories
            },
            "documents": inventories,
        }

    if publish:
        output_dir.mkdir(parents=True, exist_ok=True)
        if all_passed:
            for inventory in inventories:
                write_json(
                    inventory_dir / f"{inventory['paper_id']}.json", inventory
                )
            write_json(batch_path, batch)
        else:
            for source in document_sources:
                stale = inventory_dir / f"{source['document']['slug']}.json"
                if stale.exists():
                    stale.unlink()
            if batch_path.exists():
                batch_path.unlink()
        write_json(status_path, status)
    return status, batch


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the complete Popo + MinerU scope candidate inventory."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Workspace-relative path to the inventory configuration JSON.",
    )
    args = parser.parse_args(argv)
    status, _ = build_from_config(Path(args.config), PROJECT_ROOT, publish=True)
    print(
        f"{status['status']}: {status['passed_document_count']}/"
        f"{status['document_count']} documents passed; "
        f"status={STATUS_FILENAME}"
    )
    return 0 if status["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
