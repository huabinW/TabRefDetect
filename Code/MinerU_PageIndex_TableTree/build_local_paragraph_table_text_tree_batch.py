import argparse
import copy
import json
import re
from pathlib import Path

from build_local_table_text_tree_batch import find_reviewed_text_tree_path
from build_table_text_tree_batch import (
    DEFAULT_MANIFEST,
    MEDIA_TYPES,
    OUTLINE_ONLY_TITLE_PATTERNS,
    best_page_node,
    caption_for_table,
    count_desc,
    find_mineru_content_path,
    find_node,
    is_code_table,
    is_explicit_figure_table,
    is_outline_only_node,
    load_json,
    make_context,
    make_table_leaf,
    order_key,
    page_index,
    text_for_item,
)


PARAGRAPH_TYPES = {"text", "list"}
STANDALONE_TEXT_TYPES = {"aside_text", "page_footnote", "equation", "ref_text", "header"}
SKIP_TYPES = {"page_number"}
SENTENCE_END_RE = re.compile(r"""[.!?]["')\]]*\s*$""")
LOWERCASE_START_RE = re.compile(r"""^\s*["'(\[]*[a-z]""")
CAPTION_START_RE = re.compile(r"^\s*(?:figure|fig\.|table|tab\.)\s*\w+", re.I)
DANGLING_END_RE = re.compile(
    r"\b(?:a|an|the|and|or|but|of|to|in|on|for|with|by|from|as|than|that|which|who|whose|where|when)\s*$",
    re.I,
)


def full_text_for_item(item):
    text = text_for_item(item)
    if text:
        return text
    if item.get("type") in {"equation", "ref_text", "header"}:
        return item.get("text", "")
    return ""


def bbox_overlap_ratio(left, right):
    if not left or not right or len(left) < 4 or len(right) < 4:
        return 0.0
    overlap = max(0, min(left[2], right[2]) - max(left[0], right[0]))
    denominator = max(1, min(left[2] - left[0], right[2] - right[0]))
    return overlap / denominator


def same_column(left, right):
    return bbox_overlap_ratio(left.get("bbox"), right.get("bbox")) >= 0.45


def ends_sentence(text):
    return bool(SENTENCE_END_RE.search((text or "").strip()))


def starts_lowercase(text):
    return bool(LOWERCASE_START_RE.search(text or ""))


def vertical_gap(left, right):
    left_bbox = left.get("bbox")
    right_bbox = right.get("bbox")
    if not left_bbox or not right_bbox or len(left_bbox) < 4 or len(right_bbox) < 4:
        return None
    return right_bbox[1] - left_bbox[3]


def should_merge_blocks(previous, current):
    if previous["type"] != "text" or current["type"] != "text":
        return False, None
    if previous.get("text_level") is not None or current.get("text_level") is not None:
        return False, None

    previous_text = previous["text"]
    current_text = current["text"]
    previous_page = previous["page_index"]
    current_page = current["page_index"]
    if not previous_text.strip() or not current_text.strip():
        return False, None
    if CAPTION_START_RE.match(previous_text) or CAPTION_START_RE.match(current_text):
        return False, None
    if previous_text.rstrip().endswith(":"):
        return False, None
    if current_page == previous_page:
        if previous["target_node_id"] != current["target_node_id"]:
            return False, None
        gap = vertical_gap(previous, current)
        continuation = not ends_sentence(previous_text) or starts_lowercase(current_text)
        if same_column(previous, current) and gap is not None and -4 <= gap <= 28 and continuation:
            return True, "same_page_same_column_continuation"
        return False, None

    if current_page == previous_page + 1:
        previous_bbox = previous.get("bbox") or []
        current_bbox = current.get("bbox") or []
        previous_near_bottom = len(previous_bbox) >= 4 and previous_bbox[3] >= 700
        current_near_top = len(current_bbox) >= 4 and current_bbox[1] <= 350
        continuation = not ends_sentence(previous_text) or starts_lowercase(current_text)
        if previous_near_bottom and current_near_top and continuation:
            if previous["target_node_id"] != current["target_node_id"]:
                strong_continuation = starts_lowercase(current_text) or bool(
                    DANGLING_END_RE.search(previous_text.strip())
                )
                if strong_continuation:
                    return True, "cross_page_sentence_continuation_across_coarse_section_boundary"
                return False, None
            return True, "cross_page_sentence_continuation"
    return False, None


def atomic_text_record(item_index, item, target):
    return {
        "content_index": item_index,
        "type": item.get("type"),
        "text": full_text_for_item(item),
        "text_level": item.get("text_level"),
        "page_index": page_index(item),
        "page_idx_zero_based": item.get("page_idx"),
        "bbox": item.get("bbox"),
        "order_key": order_key(item),
        "target_node_id": target.get("node_id"),
        "target_title": target.get("title"),
    }


def make_paragraph_leaf(paragraph_index, records, merge_reasons):
    full_text = "\n".join(record["text"] for record in records)
    pages = [record["page_index"] for record in records if record["page_index"] is not None]
    return {
        "title": f"OCR paragraph {paragraph_index:04d}",
        "node_id": f"paragraph-{paragraph_index:04d}",
        "node_type": "paragraph_leaf",
        "source": "mineru",
        "content_policy": "verbatim_ocr_blocks_joined_with_newline_no_summary_no_truncation",
        "text": full_text,
        "page_index": pages[0] if pages else None,
        "page_start": min(pages) if pages else None,
        "page_end": max(pages) if pages else None,
        "page_indices": sorted(set(pages)),
        "bbox": records[0].get("bbox"),
        "bboxes": [record.get("bbox") for record in records],
        "order_key": records[0].get("order_key"),
        "raw_content_index": records[0]["content_index"],
        "raw_content_indices": [record["content_index"] for record in records],
        "raw_blocks": [
            {
                "content_index": record["content_index"],
                "type": record["type"],
                "page_index": record["page_index"],
                "page_idx_zero_based": record["page_idx_zero_based"],
                "bbox": record["bbox"],
                "text_level": record["text_level"],
                "text": record["text"],
            }
            for record in records
        ],
        "merge_reasons": list(merge_reasons),
        "parent_node_id": records[0]["target_node_id"],
        "parent_title": records[0]["target_title"],
    }


def make_standalone_text_leaf(item_index, item, target):
    text = full_text_for_item(item)
    typ = item.get("type")
    if typ == "text" and item.get("text_level") is not None:
        node_type = "heading_evidence_leaf"
    elif typ == "equation":
        node_type = "equation_leaf"
    elif typ == "page_footnote":
        node_type = "footnote_leaf"
    elif typ == "header":
        node_type = "header_leaf"
    elif typ == "ref_text":
        node_type = "reference_text_leaf"
    else:
        node_type = "standalone_text_leaf"
    return {
        "title": f"OCR {typ} {item_index:04d}",
        "node_id": f"{node_type}-{item_index:04d}",
        "node_type": node_type,
        "source": "mineru",
        "content_policy": "verbatim_ocr_no_summary_no_truncation",
        "mineru_type": typ,
        "text": text,
        "text_level": item.get("text_level"),
        "page_index": page_index(item),
        "page_idx_zero_based": item.get("page_idx"),
        "bbox": item.get("bbox"),
        "order_key": order_key(item),
        "raw_content_index": item_index,
        "raw_content_indices": [item_index],
        "raw_blocks": [
            {
                "content_index": item_index,
                "type": typ,
                "page_index": page_index(item),
                "page_idx_zero_based": item.get("page_idx"),
                "bbox": item.get("bbox"),
                "text_level": item.get("text_level"),
                "text": text,
            }
        ],
        "parent_node_id": target.get("node_id"),
        "parent_title": target.get("title"),
    }


def make_media_leaf(item_index, item, mineru_dir, target, classification_note=None):
    text = full_text_for_item(item)
    leaf = {
        "title": f"OCR media {item_index:04d}",
        "node_id": f"media-{item_index:04d}",
        "node_type": "media_leaf",
        "source": "mineru",
        "content_policy": "verbatim_ocr_no_summary_no_truncation",
        "mineru_type": item.get("type"),
        "text": text,
        "page_index": page_index(item),
        "page_idx_zero_based": item.get("page_idx"),
        "bbox": item.get("bbox"),
        "order_key": order_key(item),
        "image_path": str(mineru_dir / item["img_path"]) if item.get("img_path") else None,
        "raw_content_index": item_index,
        "raw_mineru_item": item,
        "parent_node_id": target.get("node_id"),
        "parent_title": target.get("title"),
    }
    if classification_note:
        leaf["classification_note"] = classification_note
    return leaf


def flush_paragraph(buffer, merge_reasons, paragraph_counter, paragraph_leaves):
    if not buffer:
        return paragraph_counter
    paragraph_counter += 1
    paragraph_leaves.append(make_paragraph_leaf(paragraph_counter, buffer, merge_reasons))
    buffer.clear()
    merge_reasons.clear()
    return paragraph_counter


def attach_full_ocr_leaves(tree, content, mineru_dir, doc):
    assignments = []
    table_count = 0
    paragraph_counter = 0
    skipped_outline_only = 0
    used_table_labels = set()
    table_parent_overrides = doc.get("table_parent_overrides", {})
    paragraph_leaves = []
    standalone_leaves = []
    media_leaves = []
    table_leaves = []
    buffer = []
    merge_reasons = []

    def flush():
        nonlocal paragraph_counter
        paragraph_counter = flush_paragraph(buffer, merge_reasons, paragraph_counter, paragraph_leaves)

    for item_index, item in enumerate(content):
        typ = item.get("type")
        if typ in SKIP_TYPES:
            continue
        page = page_index(item)
        target = best_page_node(tree, page)
        if target is None:
            flush()
            continue

        if is_explicit_figure_table(item):
            media_leaves.append(
                make_media_leaf(
                    item_index,
                    item,
                    mineru_dir,
                    target,
                    "mineru_table_reclassified_as_media_leaf_because_caption_is_explicit_figure",
                )
            )
            continue

        if typ == "table" or is_code_table(item):
            table_count += 1
            leaf = make_table_leaf(item_index, table_count, item, mineru_dir, used_table_labels)
            override_node_id = table_parent_overrides.get(leaf["caption_label"])
            if override_node_id:
                target = find_node(tree, override_node_id) or target
                reason = f"manifest_override:{leaf['caption_label']}->{override_node_id}"
            else:
                reason = "page_range_deepest_match"
            leaf["parent_node_id"] = target.get("node_id")
            leaf["parent_title"] = target.get("title")
            leaf["assignment_reason"] = reason
            leaf["context_window"] = make_context(content, item_index)
            table_leaves.append(leaf)
            assignments.append(
                {
                    "table_node_id": leaf["node_id"],
                    "caption_label": leaf["caption_label"],
                    "page_index": leaf["page_index"],
                    "bbox": leaf["bbox"],
                    "raw_content_index": item_index,
                    "parent_node_id": target.get("node_id"),
                    "parent_title": target.get("title"),
                    "assignment_reason": reason,
                }
            )
            continue

        if is_outline_only_node(target):
            flush()
            skipped_outline_only += 1
            target["outline_only"] = True
            target["outline_only_reason"] = "reference_or_back_matter_section"
            continue

        if typ in MEDIA_TYPES:
            media_leaves.append(make_media_leaf(item_index, item, mineru_dir, target))
            continue

        text = full_text_for_item(item)
        if not text.strip():
            flush()
            continue

        if typ == "text" and item.get("text_level") is not None:
            flush()
            standalone_leaves.append(make_standalone_text_leaf(item_index, item, target))
            continue

        if typ in {"header", "page_footnote"}:
            standalone_leaves.append(make_standalone_text_leaf(item_index, item, target))
            continue

        if typ in STANDALONE_TEXT_TYPES:
            flush()
            standalone_leaves.append(make_standalone_text_leaf(item_index, item, target))
            continue

        if typ not in PARAGRAPH_TYPES:
            flush()
            standalone_leaves.append(make_standalone_text_leaf(item_index, item, target))
            continue

        record = atomic_text_record(item_index, item, target)
        if buffer:
            merge, reason = should_merge_blocks(buffer[-1], record)
            if merge:
                buffer.append(record)
                merge_reasons.append(
                    {
                        "left_content_index": buffer[-2]["content_index"],
                        "right_content_index": record["content_index"],
                        "reason": reason,
                    }
                )
                continue
            flush()
        buffer.append(record)

    flush()

    all_leaves = paragraph_leaves + standalone_leaves + media_leaves + table_leaves
    for leaf in all_leaves:
        target = find_node(tree, leaf.get("parent_node_id"))
        if target is not None:
            target.setdefault("nodes", []).append(leaf)

    for node, _depth in iter_structure_nodes(tree):
        if "nodes" in node:
            node["nodes"].sort(
                key=lambda child: (
                    child.get("start_index", child.get("page_index", 10**9)),
                    (
                        child.get("order_key", [10**9, 10**9, 10**9])[1]
                        if child.get("node_type") and len(child.get("order_key") or []) > 1
                        else -1
                    ),
                    child.get("raw_content_index", -1),
                    child.get("node_type", ""),
                    child.get("node_id", ""),
                )
            )

    return assignments, {
        "table_leaf_count": len(table_leaves),
        "paragraph_leaf_count": len(paragraph_leaves),
        "standalone_text_leaf_count": len(standalone_leaves),
        "media_leaf_count": len(media_leaves),
        "raw_text_block_count": sum(len(leaf["raw_content_indices"]) for leaf in paragraph_leaves),
        "merged_paragraph_count": sum(1 for leaf in paragraph_leaves if len(leaf["raw_content_indices"]) > 1),
        "mineru_content_items": len(content),
        "skipped_outline_only_items": skipped_outline_only,
    }, paragraph_leaves


def iter_structure_nodes(nodes, depth=0):
    for node in nodes:
        if not node.get("node_type"):
            yield node, depth
            yield from iter_structure_nodes(node.get("nodes", []) or [], depth + 1)


def render_full_markdown(payload):
    lines = [
        "# Local Paragraph Table-Text Tree",
        "",
        f"- Document slug: `{payload['slug']}`",
        f"- Text tree source: `{payload['text_tree_source']}`",
        f"- MinerU source: `{payload['mineru_source']}`",
        "- Paragraph content: verbatim MinerU OCR blocks joined with newlines",
        "- Summaries/truncation: none",
        "",
        "## Table Assignments",
        "",
    ]
    for item in payload["table_assignments"]:
        lines.append(
            f"- {item['caption_label']} | page {item['page_index']} | "
            f"{item['parent_node_id']} {item['parent_title']} | {item['assignment_reason']}"
        )

    lines.extend(["", "## Full OCR Tree", ""])

    def render(nodes, level=0):
        prefix = "#" * min(6, level + 2)
        for node in nodes:
            typ = node.get("node_type")
            if typ == "paragraph_leaf":
                lines.extend(
                    [
                        f"{prefix} {node['node_id']} | pages {node['page_start']}-{node['page_end']} "
                        f"| raw indices {node['raw_content_indices']}",
                        "",
                        node["text"],
                        "",
                    ]
                )
            elif typ == "table_leaf":
                lines.extend(
                    [
                        f"{prefix} {node['caption_label']} | page {node['page_index']}",
                        "",
                        node.get("caption") or "",
                        "",
                    ]
                )
            elif typ:
                lines.extend(
                    [
                        f"{prefix} {node['node_id']} | {typ} | page {node.get('page_index')}",
                        "",
                        node.get("text") or "",
                        "",
                    ]
                )
            else:
                lines.append(
                    f"{prefix} {node.get('node_id')} {node.get('title')} "
                    f"(pages {node.get('start_index')}-{node.get('end_index')}, "
                    f"paragraphs={count_desc(node, 'paragraph_leaf')}, "
                    f"tables={count_desc(node, 'table_leaf')})"
                )
                lines.append("")
                if not node.get("outline_only"):
                    render(node.get("nodes", []) or [], level + 1)

    render(payload["structure"])
    return "\n".join(lines).rstrip() + "\n"


def build_doc(doc, manifest, allow_draft=False):
    output_root = Path(manifest["output_root"])
    text_tree_path, source_kind = find_reviewed_text_tree_path(output_root, doc, allow_draft=allow_draft)
    if not text_tree_path.exists():
        raise FileNotFoundError(f"Local text tree not found for {doc['slug']}: {text_tree_path}")

    content_path = find_mineru_content_path(output_root, doc)
    mineru_dir = content_path.parent
    text_tree_payload = load_json(text_tree_path)
    content = load_json(content_path)
    tree = copy.deepcopy(text_tree_payload["structure"])
    assignments, counts, paragraph_corpus = attach_full_ocr_leaves(tree, content, mineru_dir, doc)

    payload = {
        "slug": doc["slug"],
        "source_pdf": doc["pdf_path"],
        "page_count": doc.get("page_count") or text_tree_payload.get("page_count"),
        "text_tree_source": str(text_tree_path),
        "text_tree_source_kind": source_kind,
        "mineru_source": str(content_path),
        "content_policy": {
            "paragraph_text": "Verbatim MinerU OCR block text joined with newline separators.",
            "summary": "omitted",
            "truncation": "none",
            "traceability": "Every paragraph stores all raw content indices, block texts, pages, and bboxes.",
            "merge_boundary": "Only conservative same-column or cross-page sentence continuations are merged.",
            "outline_only_sections": [pattern.pattern for pattern in OUTLINE_ONLY_TITLE_PATTERNS],
        },
        "evidence_counts": counts,
        "table_assignments": assignments,
        "paragraph_corpus": paragraph_corpus,
        "structure": tree,
    }

    out_dir = output_root / "local_paragraph_table_text_trees"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f"{doc['slug']}_local_paragraph_table_text_tree.json"
    out_md = out_dir / f"{doc['slug']}_local_paragraph_table_text_tree.md"
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.write_text(render_full_markdown(payload), encoding="utf-8")
    return {"slug": doc["slug"], "json": str(out_json), "md": str(out_md), "counts": counts}


def main():
    parser = argparse.ArgumentParser(
        description="Build local table-text trees with full, untruncated MinerU OCR paragraph leaves."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--slug", action="append", help="Build only selected slug(s). Can be repeated.")
    parser.add_argument("--allow-draft", action="store_true", help="Use local draft trees when reviewed trees are missing.")
    args = parser.parse_args()

    manifest = load_json(args.manifest)
    docs = manifest["documents"]
    if args.slug:
        selected = set(args.slug)
        docs = [doc for doc in docs if doc["slug"] in selected]

    results = [build_doc(doc, manifest, allow_draft=args.allow_draft) for doc in docs]
    status_path = Path(manifest["output_root"]) / "local_paragraph_table_text_tree_status.json"
    status_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
