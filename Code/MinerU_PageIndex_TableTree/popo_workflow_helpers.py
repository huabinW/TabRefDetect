#!/usr/bin/env python3
"""Build a full comparison between the pre-Popo and Popo TabRef workflows.

This script is intentionally separate from the existing production workflow. It
does not modify the old paragraph-tree builders or child-selection scripts.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from build_table_text_classifier_candidates import (
    CONDITION_RE,
    SECTION_PRIORITY_RE,
    lexical_tokens,
    split_child_spans,
    table_reference_pattern,
)
from select_table_description_child_blocks import (
    DATASET_RE,
    METHOD_RE,
    METRIC_RE,
    MODEL_RE,
    RESULT_RE,
    TRAINING_RE,
)


DEFAULT_BASE = Path("batch_table_text_tree/mineru_popo_comparison")
DEFAULT_MANUAL = Path(
    "batch_table_text_tree/manual_table_body_text_annotations/"
    "batch_manual_table_body_text_annotations.json"
)
DEFAULT_BASELINE = DEFAULT_BASE / "baseline_summary_5papers.json"
DEFAULT_POPO_SUMMARY = DEFAULT_BASE / "remote_reports/popo_summary.json"
DEFAULT_STRICT_STATUS = (
    Path("batch_table_text_tree/human_child_annotations_strict")
    / "strict_human_child_annotation_package_status.json"
)
DEFAULT_TREE_DIR = DEFAULT_BASE / "popo_build_tree"
DEFAULT_OUTPUT_DIR = DEFAULT_BASE / "popo_workflow_full_comparison"

TABLE_ANCHOR_MATCH_POLICY = "exact_page_bbox_unique_bipartite_v1"
TABLE_ANCHOR_MATCH_MIN_IOU = 0.70
TABLE_ANCHOR_MATCH_MAX_COORD_DELTA = 0.03
MINERU_BBOX_SCALE = 1000.0

TABLE_ANCHOR_MATCH_THRESHOLD_BASIS = (
    "provisional conservative geometry guardrail for normalized MinerU/Popo "
    "table boxes; validate with additional human-verified anchors"
)


FAMILY_REGEXES = {
    "dataset": DATASET_RE,
    "model": MODEL_RE,
    "metric": METRIC_RE,
    "training": TRAINING_RE,
    "result": RESULT_RE,
    "method": METHOD_RE,
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip().lower()


def walk_tree(node: dict[str, Any], depth: int = 0, ancestors: tuple[str, ...] = ()):
    title = normalize_space(node.get("title") or "")
    yield node, depth, ancestors
    next_ancestors = ancestors + ((node.get("title") or "").strip(),) if title else ancestors
    for child in node.get("children") or []:
        if isinstance(child, dict):
            yield from walk_tree(child, depth + 1, next_ancestors)


def first_page(node: dict[str, Any]) -> int | None:
    pages = [
        item.get("page")
        for item in node.get("location") or []
        if isinstance(item, dict) and item.get("page") is not None
    ]
    return min(pages) if pages else None


def page_set(node: dict[str, Any]) -> list[int]:
    pages = {
        int(item.get("page"))
        for item in node.get("location") or []
        if isinstance(item, dict) and item.get("page") is not None
    }
    return sorted(pages)


def min_block_distance(left: list[int], right: list[int]) -> int | None:
    if not left or not right:
        return None
    return min(abs(a - b) for a in left for b in right)


def longest_common_prefix(left: tuple[str, ...], right: tuple[str, ...]) -> int:
    count = 0
    for a, b in zip(left, right):
        if normalize_space(a) != normalize_space(b):
            break
        count += 1
    return count


def extract_popo_nodes(tree: dict[str, Any], slug: str) -> tuple[list[dict], list[dict]]:
    text_nodes = []
    table_nodes = []
    text_index = 0
    table_index = 0
    for node, depth, ancestors in walk_tree(tree):
        node_type = node.get("type")
        block_ids = [int(value) for value in (node.get("block_ids") or []) if isinstance(value, int)]
        pages = page_set(node)
        common = {
            "slug": slug,
            "depth": depth,
            "title": node.get("title") or "",
            "content": node.get("content") or "",
            "block_ids": block_ids,
            "page_indices": pages,
            "page_start": min(pages) if pages else None,
            "page_end": max(pages) if pages else None,
            "ancestor_titles": [title for title in ancestors if title],
            "location": node.get("location") or [],
        }
        if node_type == "text":
            text_index += 1
            text_nodes.append(
                {
                    **common,
                    "node_id": f"popo-text-{text_index:04d}",
                    "node_type": "popo_text_node",
                    "content_length": len(common["content"]),
                    "child_spans": [
                        {
                            "child_id": f"popo-text-{text_index:04d}-child-{i:03d}",
                            "char_start": start,
                            "char_end": end,
                            "child_text": common["content"][start:end],
                        }
                        for i, (start, end) in enumerate(split_child_spans(common["content"]), start=1)
                    ],
                }
            )
        elif node_type == "table":
            table_index += 1
            table_nodes.append(
                {
                    **common,
                    "node_id": f"popo-table-{table_index:04d}",
                    "node_type": "popo_table_node",
                    "popo_table_order": table_index,
                }
            )
    table_nodes.sort(key=lambda item: (min(item["block_ids"]) if item["block_ids"] else 10**9, item["node_id"]))
    for i, table in enumerate(table_nodes, start=1):
        table["popo_table_order"] = i
        table["node_id"] = f"popo-table-{i:04d}"
    return text_nodes, table_nodes


def manual_documents(manual: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {doc["slug"]: doc for doc in manual.get("documents", [])}


def table_anchor_by_order(document: dict[str, Any]) -> list[dict[str, Any]]:
    return [table["table_anchor"] for table in document.get("tables", [])]


def normalize_match_bbox(bbox: Any) -> list[float] | None:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        values = [float(value) for value in bbox]
    except (TypeError, ValueError):
        return None
    if max(abs(value) for value in values) > 1.5:
        values = [value / MINERU_BBOX_SCALE for value in values]
    if values[2] <= values[0] or values[3] <= values[1]:
        return None
    return values


def bbox_iou(left: list[float], right: list[float]) -> float:
    x0 = max(left[0], right[0])
    y0 = max(left[1], right[1])
    x1 = min(left[2], right[2])
    y1 = min(left[3], right[3])
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    left_area = (left[2] - left[0]) * (left[3] - left[1])
    right_area = (right[2] - right[0]) * (right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def table_anchor_pair_audit(
    anchor_index: int,
    anchor: dict[str, Any],
    node_index: int,
    table_node: dict[str, Any],
    min_iou: float,
    max_coord_delta: float,
) -> dict[str, Any]:
    anchor_bbox = normalize_match_bbox(anchor.get("bbox"))
    anchor_page = anchor.get("page_index")
    try:
        anchor_page = int(anchor_page) if anchor_page is not None else None
    except (TypeError, ValueError):
        anchor_page = None

    same_page_locations = []
    for location in table_node.get("location") or []:
        if not isinstance(location, dict):
            continue
        try:
            page = int(location.get("page"))
        except (TypeError, ValueError):
            continue
        location_bbox = normalize_match_bbox(location.get("bbox"))
        if page == anchor_page and location_bbox is not None:
            same_page_locations.append((page, location_bbox))

    result = {
        "anchor_index": anchor_index,
        "canonical_label": anchor.get("canonical_label"),
        "anchor_page": anchor_page,
        "anchor_bbox_normalized": anchor_bbox,
        "popo_table_index": node_index,
        "popo_node_id": table_node.get("node_id"),
        "popo_pages": table_node.get("page_indices") or [],
        "eligible": False,
        "reason": None,
        "bbox_iou": None,
        "max_coord_delta": None,
        "popo_bbox_normalized": None,
    }
    if anchor_page is None or anchor_bbox is None:
        result["reason"] = "invalid_anchor_page_or_bbox"
        return result
    if not same_page_locations:
        result["reason"] = "no_same_page_popo_location"
        return result

    geometry = []
    for _, location_bbox in same_page_locations:
        geometry.append(
            (
                bbox_iou(anchor_bbox, location_bbox),
                max(
                    abs(left - right)
                    for left, right in zip(anchor_bbox, location_bbox)
                ),
                location_bbox,
            )
        )
    best_iou, best_delta, best_bbox = max(
        geometry, key=lambda item: (item[0], -item[1])
    )
    result.update(
        {
            "bbox_iou": round(best_iou, 6),
            "max_coord_delta": round(best_delta, 6),
            "popo_bbox_normalized": best_bbox,
        }
    )
    if best_iou < min_iou:
        result["reason"] = "bbox_iou_below_guardrail"
        return result
    if best_delta > max_coord_delta:
        result["reason"] = "bbox_delta_above_guardrail"
        return result
    result["eligible"] = True
    result["reason"] = "exact_page_and_bbox_guardrails_passed"
    return result


def _complete_table_matchings(
    eligible_by_anchor: dict[int, list[int]],
    anchor_count: int,
) -> list[dict[int, int]]:
    solutions: list[dict[int, int]] = []

    def visit(assignments: dict[int, int], used_nodes: set[int]) -> None:
        if len(solutions) >= 2:
            return
        if len(assignments) == anchor_count:
            solutions.append(dict(assignments))
            return
        remaining = [index for index in range(anchor_count) if index not in assignments]
        anchor_index = min(
            remaining,
            key=lambda index: len(
                [node for node in eligible_by_anchor[index] if node not in used_nodes]
            ),
        )
        candidates = [
            node
            for node in eligible_by_anchor[anchor_index]
            if node not in used_nodes
        ]
        for node_index in candidates:
            assignments[anchor_index] = node_index
            used_nodes.add(node_index)
            visit(assignments, used_nodes)
            used_nodes.remove(node_index)
            del assignments[anchor_index]

    visit({}, set())
    return solutions


def match_table_anchors(
    anchors: list[dict[str, Any]],
    table_nodes: list[dict[str, Any]],
    min_iou: float = TABLE_ANCHOR_MATCH_MIN_IOU,
    max_coord_delta: float = TABLE_ANCHOR_MATCH_MAX_COORD_DELTA,
) -> dict[str, Any]:
    audit: dict[str, Any] = {
        "schema_version": "1.0",
        "policy": TABLE_ANCHOR_MATCH_POLICY,
        "threshold_basis": TABLE_ANCHOR_MATCH_THRESHOLD_BASIS,
        "min_iou": min_iou,
        "max_coord_delta": max_coord_delta,
        "manual_table_count": len(anchors),
        "popo_table_count": len(table_nodes),
        "status": "fail",
        "failures": [],
        "pair_audit": [],
        "matches": [],
    }
    if len(anchors) != len(table_nodes):
        audit["failures"].append(
            {
                "reason": "table_count_mismatch",
                "manual_table_count": len(anchors),
                "popo_table_count": len(table_nodes),
            }
        )
        return audit
    if not anchors:
        audit["status"] = "pass"
        return audit

    labels = [anchor.get("canonical_label") for anchor in anchors]
    duplicate_labels = sorted(
        {label for label in labels if label and labels.count(label) > 1}
    )
    if duplicate_labels:
        audit["failures"].append(
            {"reason": "duplicate_canonical_labels", "labels": duplicate_labels}
        )
        return audit

    eligible_by_anchor = {index: [] for index in range(len(anchors))}
    pair_lookup = {}
    for anchor_index, anchor in enumerate(anchors):
        for node_index, table_node in enumerate(table_nodes):
            pair = table_anchor_pair_audit(
                anchor_index,
                anchor,
                node_index,
                table_node,
                min_iou,
                max_coord_delta,
            )
            audit["pair_audit"].append(pair)
            pair_lookup[(anchor_index, node_index)] = pair
            if pair["eligible"]:
                eligible_by_anchor[anchor_index].append(node_index)

    for anchor_index, candidates in eligible_by_anchor.items():
        if not candidates:
            audit["failures"].append(
                {
                    "reason": "anchor_has_no_eligible_popo_table",
                    "anchor_index": anchor_index,
                    "canonical_label": anchors[anchor_index].get(
                        "canonical_label"
                    ),
                }
            )
    if audit["failures"]:
        return audit

    solutions = _complete_table_matchings(eligible_by_anchor, len(anchors))
    if not solutions:
        audit["failures"].append(
            {"reason": "no_complete_one_to_one_matching"}
        )
        return audit
    if len(solutions) > 1:
        audit["failures"].append(
            {"reason": "ambiguous_multiple_complete_matchings"}
        )
        return audit

    for anchor_index, node_index in sorted(solutions[0].items()):
        pair = pair_lookup[(anchor_index, node_index)]
        audit["matches"].append(
            {
                "anchor_index": anchor_index,
                "canonical_label": anchors[anchor_index].get("canonical_label"),
                "popo_table_index": node_index,
                "popo_node_id": table_nodes[node_index].get("node_id"),
                "page": pair["anchor_page"],
                "bbox_iou": pair["bbox_iou"],
                "max_coord_delta": pair["max_coord_delta"],
                "anchor_bbox_normalized": pair["anchor_bbox_normalized"],
                "popo_bbox_normalized": pair["popo_bbox_normalized"],
                "assignment_reason": TABLE_ANCHOR_MATCH_POLICY,
            }
        )
    audit["status"] = "pass"
    return audit


def candidate_signals(table_anchor: dict[str, Any], popo_table: dict[str, Any], text_node: dict[str, Any]) -> dict[str, Any]:
    label = table_anchor["canonical_label"]
    text = f"{text_node.get('title') or ''}\n{text_node.get('content') or ''}"
    exact_reference = bool(table_reference_pattern(label).search(text))
    table_pages = popo_table.get("page_indices") or []
    text_pages = text_node.get("page_indices") or []
    if table_pages and text_pages:
        page_distance = min(abs(a - b) for a in table_pages for b in text_pages)
    else:
        page_distance = None
    block_distance = min_block_distance(popo_table.get("block_ids") or [], text_node.get("block_ids") or [])
    ancestor_overlap = longest_common_prefix(
        tuple(popo_table.get("ancestor_titles") or ()),
        tuple(text_node.get("ancestor_titles") or ()),
    )
    same_or_nested_section = ancestor_overlap >= max(1, min(len(popo_table.get("ancestor_titles") or []), len(text_node.get("ancestor_titles") or [])))
    section_priority = bool(SECTION_PRIORITY_RE.search(" ".join(text_node.get("ancestor_titles") or []) + " " + text_node.get("title", "")))
    condition_rich = bool(CONDITION_RE.search(text))
    table_tokens = lexical_tokens(
        " ".join(
            value or ""
            for value in (
                table_anchor.get("caption"),
                table_anchor.get("table_html"),
                table_anchor.get("table_code_body"),
            )
        )
    )
    text_tokens = lexical_tokens(text)
    overlap = sorted(table_tokens & text_tokens)
    overlap_score = min(30, len(overlap) * 5)
    score = 0
    score += 100 if exact_reference else 0
    score += 35 if same_or_nested_section else min(30, ancestor_overlap * 10)
    if page_distance is not None:
        score += 35 if page_distance == 0 else 25 if page_distance == 1 else 15 if page_distance == 2 else 0
    if block_distance is not None:
        score += 20 if block_distance <= 8 else 10 if block_distance <= 20 else 0
    score += 20 if section_priority else 0
    score += 10 if condition_rich else 0
    score += overlap_score
    if exact_reference:
        tier = "tier_1_explicit_reference"
    elif (page_distance is not None and page_distance <= 1) or (block_distance is not None and block_distance <= 8):
        tier = "tier_2_local_context"
    elif same_or_nested_section or section_priority or condition_rich or overlap:
        tier = "tier_3_section_condition"
    else:
        tier = "tier_4_background"
    return {
        "priority_score": score,
        "priority_tier": tier,
        "exact_table_reference": exact_reference,
        "page_distance": page_distance,
        "block_distance": block_distance,
        "ancestor_overlap": ancestor_overlap,
        "same_or_nested_section": same_or_nested_section,
        "experiment_section": section_priority,
        "condition_rich_text": condition_rich,
        "table_text_lexical_overlap_count": len(overlap),
        "table_text_lexical_overlap_sample": overlap[:20],
    }


def child_score(table_anchor: dict[str, Any], child_text: str) -> dict[str, Any]:
    table_tokens = lexical_tokens(
        " ".join(
            value or ""
            for value in (
                table_anchor.get("caption"),
                table_anchor.get("table_html"),
                table_anchor.get("table_code_body"),
            )
        )
    )
    child_tokens = lexical_tokens(child_text)
    overlap = sorted(table_tokens & child_tokens)
    exact_reference = bool(table_reference_pattern(table_anchor["canonical_label"]).search(child_text))
    family_hits = sorted(name for name, regex in FAMILY_REGEXES.items() if regex.search(child_text))
    score = 0
    score += 100 if exact_reference else 0
    score += min(30, len(overlap) * 5)
    score += 15 if "result" in family_hits else 0
    score += 15 if CONDITION_RE.search(child_text) else 0
    score += min(30, len(family_hits) * 8)
    return {
        "score": score,
        "exact_table_reference": exact_reference,
        "table_text_overlap_count": len(overlap),
        "table_text_overlap_sample": overlap[:20],
        "evidence_family_hits": family_hits,
        "condition_match": bool(CONDITION_RE.search(child_text)),
        "result_match": bool(RESULT_RE.search(child_text)),
    }


def select_projected_children(table_anchor: dict[str, Any], candidate: dict[str, Any], score_threshold: int) -> list[dict[str, Any]]:
    selected = []
    for child in candidate["children"]:
        signals = child_score(table_anchor, child["child_text"])
        keep = bool(signals["exact_table_reference"] or signals["score"] >= score_threshold)
        if keep:
            selected.append({**child, "child_selection_signals": signals})
    if candidate["parent_selected"] and not selected and candidate["children"]:
        fallback = max(candidate["children"], key=lambda item: child_score(table_anchor, item["child_text"])["score"])
        selected.append(
            {
                **fallback,
                "child_selection_signals": child_score(table_anchor, fallback["child_text"]),
                "selection_reason": "selected_parent_fallback_highest_scoring_child",
            }
        )
    return selected


def best_evidence_match(evidence_text: str, text_nodes: list[dict[str, Any]]) -> dict[str, Any]:
    evidence_norm = normalize_space(evidence_text)
    evidence_tokens = lexical_tokens(evidence_text)
    best = {
        "coverage_status": "unmatched",
        "best_node_id": None,
        "best_node_title": None,
        "best_containment": 0.0,
        "best_exact_substring": False,
    }
    if not evidence_tokens:
        return best
    for node in text_nodes:
        node_text = f"{node.get('title') or ''}\n{node.get('content') or ''}"
        node_norm = normalize_space(node_text)
        node_tokens = lexical_tokens(node_text)
        exact = evidence_norm and evidence_norm in node_norm
        containment = len(evidence_tokens & node_tokens) / max(1, len(evidence_tokens))
        if exact:
            containment = 1.0
        if containment > best["best_containment"] or (exact and not best["best_exact_substring"]):
            best = {
                "coverage_status": "covered" if exact or containment >= 0.65 else "weak_match" if containment >= 0.45 else "unmatched",
                "best_node_id": node["node_id"],
                "best_node_title": node.get("title"),
                "best_containment": round(containment, 4),
                "best_exact_substring": exact,
            }
    return best


def build_popo_candidate_dataset(
    manual: dict[str, Any],
    tree_dir: Path,
    parent_score_threshold: int,
    child_score_threshold: int,
) -> dict[str, Any]:
    documents_by_slug = manual_documents(manual)
    output_docs = []
    coverage_rows = []
    audit_failures = []
    table_anchor_match_audits = []
    for slug, document in documents_by_slug.items():
        tree_path = tree_dir / f"{slug}.json"
        if not tree_path.exists():
            audit_failures.append({"slug": slug, "reason": "missing_popo_tree", "path": str(tree_path)})
            continue
        tree = load_json(tree_path)
        text_nodes, table_nodes = extract_popo_nodes(tree, slug)
        anchors = table_anchor_by_order(document)
        anchor_match_audit = match_table_anchors(anchors, table_nodes)
        table_anchor_match_audits.append({"slug": slug, **anchor_match_audit})
        if anchor_match_audit["status"] != "pass":
            audit_failures.append(
                {
                    "slug": slug,
                    "reason": "table_anchor_match_failed",
                    "details": anchor_match_audit["failures"],
                }
            )
            continue
        table_items = []
        evidence_count = 0
        covered_count = 0
        weak_count = 0
        for anchor_match in anchor_match_audit["matches"]:
            table_index = anchor_match["anchor_index"]
            table_anchor = anchors[table_index]
            popo_table = table_nodes[anchor_match["popo_table_index"]]
            candidates = []
            projected_selected_children = []
            selected_parent_count = 0
            for text_node in text_nodes:
                signals = candidate_signals(table_anchor, popo_table, text_node)
                parent_selected = (
                    signals["exact_table_reference"]
                    or signals["priority_score"] >= parent_score_threshold
                    or signals["priority_tier"] in {"tier_1_explicit_reference", "tier_2_local_context"}
                )
                selected_parent_count += 1 if parent_selected else 0
                children = [
                    {
                        "child_id": child["child_id"],
                        "char_start": child["char_start"],
                        "char_end": child["char_end"],
                        "child_text": child["child_text"],
                    }
                    for child in text_node["child_spans"]
                ]
                candidate = {
                    "candidate_id": f"{table_anchor['canonical_label'].lower().replace(' ', '-')}-{text_node['node_id']}",
                    "parent_node_id": text_node["node_id"],
                    "parent_title": text_node.get("title"),
                    "ancestor_titles": text_node.get("ancestor_titles"),
                    "page_indices": text_node.get("page_indices"),
                    "block_ids": text_node.get("block_ids"),
                    "content_length": text_node.get("content_length"),
                    "parent_selected": parent_selected,
                    "candidate_signals": signals,
                    "children": children,
                }
                selected_children = select_projected_children(table_anchor, candidate, child_score_threshold)
                if parent_selected and selected_children:
                    for child in selected_children:
                        projected_selected_children.append(
                            {
                                "candidate_id": candidate["candidate_id"],
                                "parent_node_id": candidate["parent_node_id"],
                                "parent_title": candidate["parent_title"],
                                "child_id": child["child_id"],
                                "char_start": child["char_start"],
                                "char_end": child["char_end"],
                                "child_selection_signals": child["child_selection_signals"],
                                "selection_reason": child.get("selection_reason", "popo_projected_code_filter"),
                            }
                        )
                candidates.append(candidate)
            candidates.sort(
                key=lambda item: (
                    -item["candidate_signals"]["priority_score"],
                    item["page_indices"][0] if item["page_indices"] else 999,
                    min(item["block_ids"]) if item["block_ids"] else 999999,
                )
            )

            table_doc = document["tables"][table_index]
            table_evidence = []
            for evidence in table_doc.get("manual_body_text_evidence", []):
                evidence_count += 1
                match = best_evidence_match(evidence.get("full_parent_text") or "", text_nodes)
                if match["coverage_status"] == "covered":
                    covered_count += 1
                elif match["coverage_status"] == "weak_match":
                    weak_count += 1
                coverage_row = {
                    "slug": slug,
                    "table_label": table_anchor["canonical_label"],
                    "evidence_type": evidence.get("evidence_type"),
                    "old_parent_paragraph_id": evidence.get("parent_paragraph_id"),
                    **match,
                }
                table_evidence.append(coverage_row)
                coverage_rows.append(coverage_row)

            table_items.append(
                {
                    "table_anchor": {
                        "canonical_label": table_anchor["canonical_label"],
                        "caption": table_anchor.get("caption"),
                        "old_page_index": table_anchor.get("page_index"),
                        "old_raw_content_index": table_anchor.get("raw_content_index"),
                    },
                    "popo_table_node": {
                        "node_id": popo_table["node_id"],
                        "block_ids": popo_table.get("block_ids"),
                        "page_indices": popo_table.get("page_indices"),
                        "ancestor_titles": popo_table.get("ancestor_titles"),
                        "anchor_match": anchor_match,
                    },
                    "candidate_count": len(candidates),
                    "selected_parent_count": selected_parent_count,
                    "child_candidate_count": sum(len(item["children"]) for item in candidates),
                    "projected_selected_child_count": len(projected_selected_children),
                    "manual_evidence_coverage": table_evidence,
                    "candidates": candidates,
                    "projected_selected_children": projected_selected_children,
                }
            )
        output_docs.append(
            {
                "slug": slug,
                "popo_tree": str(tree_path),
                "text_node_count": len(text_nodes),
                "table_node_count": len(table_nodes),
                "text_child_span_count": sum(len(node["child_spans"]) for node in text_nodes),
                "manual_evidence_count": evidence_count,
                "manual_evidence_covered_count": covered_count,
                "manual_evidence_weak_match_count": weak_count,
                "tables": table_items,
            }
        )
    return {
        "schema_version": "1.0",
        "source": "popo_tree_candidate_adapter",
        "policy": {
            "parent_score_threshold": parent_score_threshold,
            "child_score_threshold": child_score_threshold,
            "table_mapping": (
                "manual table anchors mapped to Popo table nodes by a unique "
                "exact-page/bbox one-to-one assignment; order is not a match signal"
            ),
            "table_mapping_policy": TABLE_ANCHOR_MATCH_POLICY,
            "table_mapping_threshold_basis": TABLE_ANCHOR_MATCH_THRESHOLD_BASIS,
            "manual_evidence_coverage": "exact substring or lexical-token containment >= 0.65 is covered; >= 0.45 is weak_match",
        },
        "documents": output_docs,
        "manual_evidence_coverage_rows": coverage_rows,
        "table_anchor_match_audits": table_anchor_match_audits,
        "audit_failures": audit_failures,
    }


def summarize_popo_dataset(dataset: dict[str, Any]) -> dict[str, Any]:
    rows = []
    by_doc = {}
    for doc in dataset["documents"]:
        parent_candidates = sum(table["candidate_count"] for table in doc["tables"])
        selected_parents = sum(table["selected_parent_count"] for table in doc["tables"])
        child_candidates = sum(table["child_candidate_count"] for table in doc["tables"])
        projected_children = sum(table["projected_selected_child_count"] for table in doc["tables"])
        evidence_count = doc["manual_evidence_count"]
        covered = doc["manual_evidence_covered_count"]
        weak = doc["manual_evidence_weak_match_count"]
        item = {
            "slug": doc["slug"],
            "tables": doc["table_node_count"],
            "text_nodes": doc["text_node_count"],
            "text_child_spans": doc["text_child_span_count"],
            "parent_candidates": parent_candidates,
            "selected_parent_candidates": selected_parents,
            "child_candidates": child_candidates,
            "projected_selected_children": projected_children,
            "manual_evidence_count": evidence_count,
            "manual_evidence_covered": covered,
            "manual_evidence_weak_match": weak,
            "manual_evidence_coverage_rate": round(covered / evidence_count, 4) if evidence_count else None,
        }
        rows.append(item)
        by_doc[doc["slug"]] = item
    aggregate = {
        "documents": len(rows),
        "tables": sum(row["tables"] for row in rows),
        "text_nodes": sum(row["text_nodes"] for row in rows),
        "text_child_spans": sum(row["text_child_spans"] for row in rows),
        "parent_candidates": sum(row["parent_candidates"] for row in rows),
        "selected_parent_candidates": sum(row["selected_parent_candidates"] for row in rows),
        "child_candidates": sum(row["child_candidates"] for row in rows),
        "projected_selected_children": sum(row["projected_selected_children"] for row in rows),
        "manual_evidence_count": sum(row["manual_evidence_count"] for row in rows),
        "manual_evidence_covered": sum(row["manual_evidence_covered"] for row in rows),
        "manual_evidence_weak_match": sum(row["manual_evidence_weak_match"] for row in rows),
    }
    aggregate["manual_evidence_coverage_rate"] = round(
        aggregate["manual_evidence_covered"] / aggregate["manual_evidence_count"], 4
    ) if aggregate["manual_evidence_count"] else None
    return {"rows": rows, "by_doc": by_doc, "aggregate": aggregate}


def build_comparison(
    baseline: dict[str, Any],
    popo_summary: dict[str, Any],
    strict_status: dict[str, Any],
    popo_dataset_summary: dict[str, Any],
) -> dict[str, Any]:
    base_docs = {doc["slug"]: doc for doc in baseline["documents"]}
    old_review_docs = {
        item["slug"]: item
        for item in baseline["pipeline_counts"]["prepare_codex_child_review_packages"]["documents"]
    }
    strict_docs = {item["slug"]: item for item in strict_status["documents"]}
    popo_docs = popo_dataset_summary["by_doc"]
    rows = []
    for slug in sorted(base_docs):
        base = base_docs[slug]
        popo = popo_docs[slug]
        old_parent = base["table_leaf_count"] * base["paragraph_leaf_count"]
        row = {
            "slug": slug,
            "tables": base["table_leaf_count"],
            "old_paragraphs": base["paragraph_leaf_count"],
            "popo_text_nodes": popo["text_nodes"],
            "old_parent_candidates": old_parent,
            "popo_parent_candidates": popo["parent_candidates"],
            "popo_selected_parent_candidates": popo["selected_parent_candidates"],
            "old_child_candidates": None,
            "popo_child_candidates": popo["child_candidates"],
            "old_codex_review_items": old_review_docs[slug]["review_items"],
            "old_strict_template_children": strict_docs[slug]["candidate_children"],
            "popo_projected_selected_children": popo["projected_selected_children"],
            "manual_evidence_count": popo["manual_evidence_count"],
            "manual_evidence_covered": popo["manual_evidence_covered"],
            "manual_evidence_weak_match": popo["manual_evidence_weak_match"],
            "manual_evidence_coverage_rate": popo["manual_evidence_coverage_rate"],
            "parent_candidate_reduction": round(1 - popo["parent_candidates"] / old_parent, 4) if old_parent else None,
        }
        rows.append(row)
    pipeline = baseline["pipeline_counts"]
    old_child_total = pipeline["build_classifier_candidates"]["child_candidates"]
    old_parent_total = pipeline["build_classifier_candidates"]["parent_candidates"]
    aggregate = {
        "documents": len(rows),
        "tables": sum(row["tables"] for row in rows),
        "old_paragraphs": sum(row["old_paragraphs"] for row in rows),
        "popo_text_nodes": sum(row["popo_text_nodes"] for row in rows),
        "old_parent_candidates": old_parent_total,
        "popo_parent_candidates": sum(row["popo_parent_candidates"] for row in rows),
        "popo_selected_parent_candidates": sum(row["popo_selected_parent_candidates"] for row in rows),
        "old_child_candidates": old_child_total,
        "popo_child_candidates": sum(row["popo_child_candidates"] for row in rows),
        "old_first_stage_review_children": pipeline["select_table_description_children"]["child_label_0_correct"],
        "old_codex_review_items": pipeline["prepare_codex_child_review_packages"]["review_items"],
        "old_strict_template_children": strict_status["candidate_children"],
        "popo_projected_selected_children": sum(row["popo_projected_selected_children"] for row in rows),
        "manual_evidence_count": sum(row["manual_evidence_count"] for row in rows),
        "manual_evidence_covered": sum(row["manual_evidence_covered"] for row in rows),
        "manual_evidence_weak_match": sum(row["manual_evidence_weak_match"] for row in rows),
    }
    aggregate["parent_candidate_reduction"] = round(
        1 - aggregate["popo_parent_candidates"] / aggregate["old_parent_candidates"], 4
    )
    aggregate["child_candidate_change"] = round(
        aggregate["popo_child_candidates"] / aggregate["old_child_candidates"] - 1, 4
    )
    aggregate["manual_evidence_coverage_rate"] = round(
        aggregate["manual_evidence_covered"] / aggregate["manual_evidence_count"], 4
    ) if aggregate["manual_evidence_count"] else None
    return {
        "schema_version": "1.0",
        "comparison_scope": "old local paragraph table-text workflow vs MinerU-Popo tree-adapted workflow",
        "popo_summary_aggregate": popo_summary.get("aggregate"),
        "aggregate": aggregate,
        "documents": rows,
        "interpretation": {
            "main_difference": [
                "Old workflow uses local OCR paragraph leaves as candidate parent blocks.",
                "Popo workflow uses document-tree text nodes with ancestor hierarchy as candidate parent blocks.",
                "Popo therefore reduces parent-block count but makes each parent block longer/coarser.",
            ],
            "important_caveat": [
                "Popo projected selected children are code-only estimates, not Codex second-layer semantic review labels.",
                "Manual evidence coverage is token/substring based and should be treated as a preservation audit, not a final relevance metric.",
            ],
        },
    }


def render_markdown(comparison: dict[str, Any], popo_dataset_summary: dict[str, Any]) -> str:
    agg = comparison["aggregate"]
    lines = [
        "# MinerU-Popo Full Workflow Comparison",
        "",
        "## Executive Summary",
        "",
        f"- Documents: {agg['documents']}",
        f"- Tables: {agg['tables']}",
        f"- Old parent blocks: {agg['old_paragraphs']} OCR paragraphs",
        f"- Popo parent blocks: {agg['popo_text_nodes']} tree text nodes",
        f"- Parent candidates: {agg['old_parent_candidates']} -> {agg['popo_parent_candidates']} ({agg['parent_candidate_reduction']:.1%} fewer)",
        f"- Child spans/candidates: {agg['old_child_candidates']} -> {agg['popo_child_candidates']} ({agg['child_candidate_change']:+.1%})",
        f"- Old first-stage review children: {agg['old_first_stage_review_children']}",
        f"- Old strict human-template children: {agg['old_strict_template_children']}",
        f"- Popo projected code-only selected children: {agg['popo_projected_selected_children']}",
        f"- Manual evidence preserved in Popo text nodes: {agg['manual_evidence_covered']}/{agg['manual_evidence_count']} ({agg['manual_evidence_coverage_rate']:.1%}); weak matches: {agg['manual_evidence_weak_match']}",
        "",
        "## Document-Level Metrics",
        "",
        "| document | tables | old paragraphs | Popo text nodes | old parent cand. | Popo parent cand. | selected Popo parents | old strict children | Popo projected children | manual evidence covered |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in comparison["documents"]:
        lines.append(
            f"| {row['slug']} | {row['tables']} | {row['old_paragraphs']} | {row['popo_text_nodes']} | "
            f"{row['old_parent_candidates']} | {row['popo_parent_candidates']} | {row['popo_selected_parent_candidates']} | "
            f"{row['old_strict_template_children']} | {row['popo_projected_selected_children']} | "
            f"{row['manual_evidence_covered']}/{row['manual_evidence_count']} |"
        )
    lines += [
        "",
        "## What Changed",
        "",
        "- The old workflow's parent unit is the merged OCR paragraph. This gives fine-grained parent blocks but many table-parent combinations.",
        "- The Popo workflow's parent unit is a tree text node with hierarchy. This reduces parent combinations and adds ancestor paths, but each parent can contain many sentences.",
        "- Table counts stayed aligned for all five papers, so the Popo integration did not lose table anchors at the construction-count level.",
        "- Popo preserved most manually selected evidence text inside its tree text nodes, so the main risk is not missing OCR text, but choosing the right child spans from coarser parent nodes.",
        "",
        "## Caveats",
        "",
        "- Popo projected selected children are produced by code-only scoring. They are not equivalent to the old Codex semantic review results.",
        "- The next production step should adapt the child selector to Popo nodes: split Popo text nodes into child spans, keep ancestor paths, then run the same high-recall code filter plus Codex/local-model precision filter.",
        "- Manual evidence coverage uses substring/token containment and does not replace human relevance annotation.",
    ]
    return "\n".join(lines) + "\n"


def write_table_readable(dataset: dict[str, Any], output_path: Path) -> None:
    lines = ["# Popo Projected Candidate Summary", ""]
    for doc in dataset["documents"]:
        lines += [f"## {doc['slug']}", ""]
        for table in doc["tables"]:
            label = table["table_anchor"]["canonical_label"]
            lines.append(
                f"- {label}: parents={table['candidate_count']}, selected_parents={table['selected_parent_count']}, "
                f"child_spans={table['child_candidate_count']}, projected_children={table['projected_selected_child_count']}"
            )
        lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manual", type=Path, default=DEFAULT_MANUAL)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--popo-summary", type=Path, default=DEFAULT_POPO_SUMMARY)
    parser.add_argument("--strict-status", type=Path, default=DEFAULT_STRICT_STATUS)
    parser.add_argument("--tree-dir", type=Path, default=DEFAULT_TREE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--parent-score-threshold", type=int, default=80)
    parser.add_argument("--child-score-threshold", type=int, default=25)
    args = parser.parse_args()

    manual = load_json(args.manual)
    baseline = load_json(args.baseline)
    popo_summary = load_json(args.popo_summary)
    strict_status = load_json(args.strict_status)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = build_popo_candidate_dataset(
        manual,
        args.tree_dir,
        args.parent_score_threshold,
        args.child_score_threshold,
    )
    dataset_summary = summarize_popo_dataset(dataset)
    comparison = build_comparison(baseline, popo_summary, strict_status, dataset_summary)

    dataset_path = args.output_dir / "popo_table_text_candidate_dataset.json"
    summary_path = args.output_dir / "popo_candidate_dataset_summary.json"
    comparison_json_path = args.output_dir / "mineru_popo_full_workflow_comparison.json"
    comparison_md_path = args.output_dir / "mineru_popo_full_workflow_comparison.md"
    readable_path = args.output_dir / "popo_projected_candidate_summary.md"
    dataset_path.write_text(json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(dataset_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    comparison_json_path.write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
    comparison_md_path.write_text(render_markdown(comparison, dataset_summary), encoding="utf-8")
    write_table_readable(dataset, readable_path)
    print(
        json.dumps(
            {
                "dataset": str(dataset_path),
                "summary": str(summary_path),
                "comparison_json": str(comparison_json_path),
                "comparison_md": str(comparison_md_path),
                "readable": str(readable_path),
                "aggregate": comparison["aggregate"],
                "audit_failures": dataset["audit_failures"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if dataset["audit_failures"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
