import argparse
import json
import math
import re
from pathlib import Path

from build_table_text_tree_batch import (
    DEFAULT_MANIFEST,
    find_mineru_content_path,
    load_json,
    page_index,
    text_for_item,
)


TABLE_LABEL_RE = re.compile(
    r"""
    \b(?P<prefix>
        extended\s+data\s+table|
        supplementary\s+table|
        supplemental\s+table|
        table|
        tab\.?
    )\s*
    (?P<number>
        S?\d+[A-Za-z]?|
        [A-Z]\.?\d+[A-Za-z]?|
        [IVXLCDM]+
    )
    (?=\s*[:：.\-\u2013\u2014)]|\s|$)
    |
    (?P<zh_prefix>表)\s*(?P<zh_number>S?\d+[A-Za-z]?|[A-Z]\.?\d+[A-Za-z]?)
    (?=\s*[:：.\-\u2013\u2014)]|\s|$)
    """,
    re.I | re.X,
)

HORIZONTAL_OVERLAP_MIN = 0.35
NEAR_READING_ORDER_DISTANCE = 2
SAME_PAGE_CONTEXT_DISTANCE = 10
CAPTION_LABEL_START_STRICT = 5
CAPTION_LABEL_START_LOOSE = 25
PROSE_REFERENCE_LABEL_START = 60

CAPTION_START_RE = re.compile(
    r"^\s*(extended\s+data\s+table|supplementary\s+table|supplemental\s+table|table|tab\.?|表)\b",
    re.I,
)


def norm_text(value):
    return re.sub(r"\s+", " ", value or "").strip()


def bbox_center(bbox):
    if not bbox or len(bbox) < 4:
        return None
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)


def canonical_label(match):
    number = match.group("number") or match.group("zh_number")
    return f"Table {number.replace('.', '') if len(number) <= 3 and number.count('.') == 1 else number}"


def raw_label(match):
    return norm_text(match.group(0))


def label_matches(text):
    return list(TABLE_LABEL_RE.finditer(text or ""))


def title_segment(text, match, matches):
    start = match.start()
    later = [m.start() for m in matches if m.start() > start]
    end = min(later) if later else len(text)
    return norm_text(text[start:end])


def label_start_for_caption_test(text, match, segment):
    if not segment:
        return match.start()
    segment_match = TABLE_LABEL_RE.search(segment)
    if segment_match:
        return segment_match.start()
    return match.start()


def item_text(item):
    typ = item.get("type")
    if typ == "table":
        caption = " ".join(item.get("table_caption") or [])
        body = item.get("table_body") or ""
        return norm_text(caption or body)
    if typ == "code":
        caption = " ".join(item.get("code_caption") or [])
        return norm_text(caption or item.get("code_body") or "")
    return norm_text(text_for_item(item))


def source_kind(item, table_index):
    if item.get("type") == "table" and item.get("raw_content_index") == table_index:
        return "own_table_caption"
    if item.get("type") == "code" and item.get("raw_content_index") == table_index:
        return "own_code_caption"
    return item.get("type") or "unknown"


def overlap_ratio(a, b):
    if not a or not b or len(a) < 4 or len(b) < 4:
        return 0.0
    left = max(a[0], b[0])
    right = min(a[2], b[2])
    width = max(0, right - left)
    denom = max(1, min(a[2] - a[0], b[2] - b[0]))
    return width / denom


def vertical_relation(candidate_bbox, table_bbox):
    if not candidate_bbox or not table_bbox or len(candidate_bbox) < 4 or len(table_bbox) < 4:
        return "unknown", math.inf
    if candidate_bbox[3] <= table_bbox[1]:
        return "above", table_bbox[1] - candidate_bbox[3]
    if candidate_bbox[1] >= table_bbox[3]:
        return "below", candidate_bbox[1] - table_bbox[3]
    return "overlap", 0


def caption_like(text, match):
    if not text:
        return False
    if match.start() <= CAPTION_LABEL_START_STRICT and CAPTION_START_RE.search(text):
        return True
    if match.start() <= CAPTION_LABEL_START_LOOSE and re.match(r"^\s*(\(?[a-z]\)\s*)?(extended|supplementary|supplemental|table|tab\.?|表)", text, re.I):
        return True
    return False


def same_canonical(left, right):
    return norm_text(left).lower().replace(" ", "") == norm_text(right).lower().replace(" ", "")


def label_number(canonical):
    parts = norm_text(canonical).split(maxsplit=1)
    return parts[1] if len(parts) == 2 else norm_text(canonical)


def label_reference_re(canonical):
    number = re.escape(label_number(canonical))
    return re.compile(rf"\b(table|tab\.?)\s*{number}\b|表\s*{number}\b", re.I)


def score_candidate(table, candidate):
    score = 0.0
    if candidate["source_kind"] in {"own_table_caption", "own_code_caption"}:
        score += 45
    elif candidate["is_caption_like"]:
        score += 25
    else:
        score -= 20

    if candidate["page_index"] == table["page_index"]:
        score += 20
    elif abs((candidate["page_index"] or 0) - (table["page_index"] or 0)) == 1:
        score += 4

    if same_canonical(candidate["canonical_label"], table["current_caption_label"]):
        score += 25

    relation, distance = vertical_relation(candidate["bbox"], table["bbox"])
    candidate["vertical_relation"] = relation
    candidate["vertical_distance"] = None if math.isinf(distance) else distance
    if relation in {"above", "below", "overlap"}:
        score += max(0, 18 - min(distance, 360) / 20)
        score += {"above": 8, "below": 6, "overlap": 4}[relation]

    score += min(8, 8 * overlap_ratio(candidate["bbox"], table["bbox"]))
    index_distance = abs((candidate["content_index"] or 0) - (table["raw_content_index"] or 0))
    score += max(0, 12 - min(index_distance, 12))

    if candidate.get("label_segment_start", candidate["label_match_start"]) > PROSE_REFERENCE_LABEL_START:
        score -= 15
    return round(score, 3)


def confidence_from_score(score):
    if score >= 95:
        return "high"
    if score >= 70:
        return "medium"
    return "low"


def evidence_tags(table, candidate):
    tags = []
    if candidate["source_kind"] == "own_table_caption":
        tags.append("mineru_table_caption")
    elif candidate["source_kind"] == "own_code_caption":
        tags.append("mineru_code_caption")
    elif candidate["is_caption_like"]:
        tags.append("caption_like_text")
    else:
        tags.append("numbered_prose_reference")

    if candidate["page_index"] == table["page_index"]:
        tags.append("same_page")
    elif abs((candidate["page_index"] or 0) - (table["page_index"] or 0)) == 1:
        tags.append("adjacent_page")

    if same_canonical(candidate["canonical_label"], table["current_caption_label"]):
        tags.append("label_matches_current")

    relation = candidate.get("vertical_relation")
    if relation and relation != "unknown":
        tags.append(f"bbox_{relation}")
    if overlap_ratio(candidate["bbox"], table["bbox"]) >= HORIZONTAL_OVERLAP_MIN:
        tags.append("horizontal_overlap")

    index_distance = abs((candidate["content_index"] or 0) - (table["raw_content_index"] or 0))
    if index_distance <= NEAR_READING_ORDER_DISTANCE:
        tags.append("near_reading_order")
    elif index_distance <= SAME_PAGE_CONTEXT_DISTANCE:
        tags.append("same_page_context_window")

    if candidate.get("label_segment_start", candidate["label_match_start"]) > PROSE_REFERENCE_LABEL_START:
        tags.append("label_not_at_text_start")
    return tags


def source_description(candidate):
    if not candidate:
        return ""
    mapping = {
        "own_table_caption": "MinerU table_caption",
        "own_code_caption": "MinerU code_caption",
        "text": "MinerU text",
        "list": "MinerU list",
        "table": "Neighbor MinerU table item",
        "aside_text": "MinerU aside_text",
        "page_footnote": "MinerU page_footnote",
    }
    return mapping.get(candidate.get("source_kind")) or mapping.get(candidate.get("source_type")) or candidate.get("source_type") or ""


def public_candidate(candidate, table, include_full_text=False):
    if not candidate:
        return None
    result = {
        "content_index": candidate.get("content_index"),
        "source_type": candidate.get("source_type"),
        "source": source_description(candidate),
        "page_index": candidate.get("page_index"),
        "bbox": candidate.get("bbox"),
        "raw_label": candidate.get("raw_label"),
        "canonical_label": candidate.get("canonical_label"),
        "title_candidate": candidate.get("title_candidate"),
        "is_caption_like": candidate.get("is_caption_like"),
        "vertical_relation_to_table": candidate.get("vertical_relation"),
        "vertical_distance_to_table": candidate.get("vertical_distance"),
        "evidence_tags": evidence_tags(table, candidate),
    }
    if include_full_text:
        result["source_text"] = candidate.get("full_text")
    return result


def iter_table_nodes(nodes):
    for node in nodes:
        if node.get("node_type") == "table_leaf":
            yield node
        yield from iter_table_nodes(node.get("nodes", []))


def table_records(payload):
    records = []
    for node in iter_table_nodes(payload["structure"]):
        records.append(
            {
                "table_node_id": node.get("node_id"),
                "raw_content_index": node.get("raw_content_index"),
                "page_index": node.get("page_index"),
                "bbox": node.get("bbox"),
                "current_caption_label": node.get("caption_label"),
                "current_caption": node.get("caption"),
                "title": node.get("title"),
                "parent_node_id": node.get("parent_node_id"),
            }
        )
    records.sort(key=lambda item: (item.get("page_index") or 0, (item.get("bbox") or [0, 0, 0, 0])[1]))
    return records


def candidate_items(content, table, index_window=10, page_window=1):
    rows = []
    for idx, item in enumerate(content):
        page = page_index(item)
        if page is None:
            continue
        if abs(page - table["page_index"]) > page_window:
            continue
        if abs(idx - table["raw_content_index"]) > index_window and page != table["page_index"]:
            continue
        text = item_text(item)
        if not text:
            continue
        matches = label_matches(text)
        if not matches:
            continue
        pseudo = dict(item)
        pseudo["raw_content_index"] = idx
        for match in matches:
            segment = title_segment(text, match, matches)
            segment_match = TABLE_LABEL_RE.search(segment)
            segment_caption_like = caption_like(segment, segment_match) if segment_match else caption_like(text, match)
            rows.append(
                {
                    "content_index": idx,
                    "source_type": item.get("type"),
                    "source_kind": source_kind(pseudo, table["raw_content_index"]),
                    "page_index": page,
                    "bbox": item.get("bbox"),
                    "raw_label": raw_label(match),
                    "canonical_label": canonical_label(match),
                    "label_match_start": match.start(),
                    "label_segment_start": label_start_for_caption_test(text, match, segment),
                    "is_caption_like": segment_caption_like,
                    "title_candidate": segment,
                    "full_text": text[:1200],
                }
            )
    return rows


def resolve_for_table(content, table):
    candidates = candidate_items(content, table)
    for candidate in candidates:
        candidate["score"] = score_candidate(table, candidate)
    caption_candidates = [c for c in candidates if c["is_caption_like"] or c["source_kind"].startswith("own_")]
    caption_candidates.sort(key=lambda item: item["score"], reverse=True)
    reference_candidates = [c for c in candidates if not c["is_caption_like"] and not c["source_kind"].startswith("own_")]
    reference_candidates.sort(key=lambda item: item["score"], reverse=True)
    selected = caption_candidates[0] if caption_candidates else None
    reference_label = selected["canonical_label"] if selected else table["current_caption_label"]
    confidence = "none"
    if selected:
        confidence = confidence_from_score(selected["score"])
    flags = flags_for_resolution(table, selected, caption_candidates)
    anchor_status = "resolved"
    if confidence == "none":
        anchor_status = "unresolved"
    elif flags:
        anchor_status = "resolved_with_debug_flags"
    return {
        "table": table,
        "selected_caption": public_candidate(selected, table, include_full_text=True),
        "confidence": confidence,
        "anchor_status": anchor_status,
        "anchor_debug_required": bool(flags) or confidence in {"low", "none"},
        "caption_candidates": [public_candidate(candidate, table) for candidate in caption_candidates[:8]],
        "nearby_reference_candidates": [public_candidate(candidate, table, include_full_text=True) for candidate in reference_candidates[:8]],
        "full_document_references": full_document_references(content, reference_label),
        "flags": flags,
    }


def full_document_references(content, canonical, limit=30):
    if not canonical:
        return []
    pattern = label_reference_re(canonical)
    rows = []
    for idx, item in enumerate(content):
        if item.get("type") in {"table", "code", "page_number"}:
            continue
        text = norm_text(text_for_item(item))
        if not text or not pattern.search(text):
            continue
        match = pattern.search(text)
        rows.append(
            {
                "content_index": idx,
                "type": item.get("type"),
                "page_index": page_index(item),
                "bbox": item.get("bbox"),
                "match": match.group(0),
                "is_caption_like": caption_like(text, match),
                "text": text[:1200],
            }
        )
    return rows[:limit]


def flags_for_resolution(table, selected, candidates):
    flags = []
    if not selected:
        flags.append("no_caption_candidate")
        return flags
    if not same_canonical(selected["canonical_label"], table["current_caption_label"]):
        flags.append("selected_label_differs_from_current_table_label")
    if selected["source_kind"] not in {"own_table_caption", "own_code_caption"}:
        flags.append("caption_recovered_from_surrounding_content")
    if selected["vertical_relation"] == "unknown":
        flags.append("caption_bbox_missing_or_unusable")
    close = [
        c for c in candidates[1:4]
        if c["score"] >= selected["score"] - 8 and c["canonical_label"] != selected["canonical_label"]
    ]
    if close:
        flags.append("nearby_competing_caption_candidate")
    if len({c["canonical_label"] for c in candidates if c["content_index"] == selected["content_index"]}) > 1:
        flags.append("source_text_contains_multiple_table_labels")
    return flags


def resolve_doc(doc, manifest):
    output_root = Path(manifest["output_root"])
    content_path = find_mineru_content_path(output_root, doc)
    tree_path = output_root / "table_text_trees" / f"{doc['slug']}_table_text_tree.json"
    content = load_json(content_path)
    payload = load_json(tree_path)
    resolutions = [resolve_for_table(content, table) for table in table_records(payload)]
    return {
        "slug": doc["slug"],
        "mineru_content": str(content_path),
        "table_tree": str(tree_path),
        "table_count": len(resolutions),
        "resolutions": resolutions,
        "summary": {
            "high": sum(1 for r in resolutions if r["confidence"] == "high"),
            "medium": sum(1 for r in resolutions if r["confidence"] == "medium"),
            "low": sum(1 for r in resolutions if r["confidence"] == "low"),
            "none": sum(1 for r in resolutions if r["confidence"] == "none"),
            "with_anchor_flags": sum(1 for r in resolutions if r["flags"]),
            "unresolved": sum(1 for r in resolutions if r["anchor_status"] == "unresolved"),
        },
    }


def render_doc(report):
    lines = [
        f"## 文档 / Document: {report['slug']}",
        "",
        "**作用 / Purpose:** 自动确定每个表格的编号、标题和位置锚点，并列出全文中显式提到该编号的段落。该文件用于调试表格锚点，不作为主要人工标注入口。",
        "",
        "### 摘要 / Summary",
        "",
        f"- 表格数 / Tables: {report['table_count']}",
        f"- 高置信 / High confidence: {report['summary']['high']}",
        f"- 锚点调试标记 / Anchor debug flags: {report['summary']['with_anchor_flags']}",
        f"- 未解析 / Unresolved: {report['summary']['unresolved']}",
        "",
        "### 自动表格锚点 / Automatic Table Anchors",
        "",
        "| 状态 / Status | 表格 / Table | 页码/BBox | 标题来源 / Source | 识别标题 / Resolved caption | 调试标记 / Debug flags |",
        "|---|---|---|---|---|---|",
    ]
    for item in report["resolutions"]:
        table = item["table"]
        selected = item["selected_caption"] or {}
        status = item["anchor_status"]
        source = (
            f"{selected.get('source', '')} idx={selected.get('content_index')} "
            f"p{selected.get('page_index')} {selected.get('vertical_relation_to_table') or ''}"
        )
        title = selected.get("title_candidate")
        flags = ", ".join(item["flags"]) if item["flags"] else "-"
        lines.append(
            "| "
            f"{status} | {table['current_caption_label']} ({item['confidence']}) | "
            f"p{table['page_index']} `{table['bbox']}` | {escape_md_cell(source)} | "
            f"{escape_md_cell((title or '')[:280])} | {escape_md_cell(flags)} |"
        )
    lines.extend(
        [
            "",
            "### 显式编号段落 / Explicit Numbered Paragraphs",
            "",
            "**作用 / Purpose:** 这些段落来自 MinerU 全文解析结果，显式包含对应表格编号。它们可作为后续补充信息分析的证据来源之一；当前阶段只记录，不进行候选段落筛选。",
            "",
            "| 表格 / Table | 段落位置 / Location | 段落文本 / Paragraph |",
            "|---|---|---|",
        ]
    )
    for item in report["resolutions"]:
        table = item["table"]
        full_refs = [ref for ref in item.get("full_document_references", []) if not ref["is_caption_like"]]
        if not full_refs:
            lines.append(f"| {table['current_caption_label']} | - | 未找到非标题段落 / No non-caption paragraph found |")
            continue
        for ref in full_refs[:3]:
            loc = f"p{ref['page_index']} idx={ref['content_index']} match={ref['match']}"
            lines.append(f"| {table['current_caption_label']} | {loc} | {escape_md_cell(ref['text'][:360])} |")
    lines.append("")
    return "\n".join(lines)


def escape_md_cell(value):
    return norm_text(str(value or "")).replace("|", "\\|")


def main():
    parser = argparse.ArgumentParser(description="Resolve table captions and labels from MinerU content around table bboxes.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()

    manifest = load_json(args.manifest)
    out_dir = Path(manifest["output_root"]) / "table_caption_resolution"
    out_dir.mkdir(parents=True, exist_ok=True)
    reports = [resolve_doc(doc, manifest) for doc in manifest["documents"]]

    out_json = out_dir / "batch_table_caption_resolution.json"
    out_md = out_dir / "batch_table_caption_resolution.md"
    out_json.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Batch Table Caption Resolution", ""]
    for report in reports:
        lines.append(render_doc(report))
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps([{k: report[k] for k in ["slug", "table_count", "summary"]} for report in reports], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
