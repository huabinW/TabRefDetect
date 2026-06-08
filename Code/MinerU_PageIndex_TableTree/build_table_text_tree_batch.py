import argparse
import copy
import json
import re
from pathlib import Path


BASE = Path(__file__).resolve().parent
DEFAULT_MANIFEST = BASE / "manifest.json"

TEXTUAL_TYPES = {"text", "list", "page_footnote", "aside_text"}
MEDIA_TYPES = {"chart", "image"}
SKIP_TYPES = {"page_number"}
TABLE_LABEL_RE = re.compile(r"\b(Table|Tab\.)\s*([A-Za-z]?\d+[A-Za-z]?)", re.IGNORECASE)
FIGURE_LABEL_RE = re.compile(r"\b(Figure|Fig\.)\s*([A-Za-z]?\d+[A-Za-z]?)", re.IGNORECASE)
OUTLINE_ONLY_TITLE_PATTERNS = [
    re.compile(r"\breferences?\b", re.I),
    re.compile(r"\bbibliograph(?:y|ies)\b", re.I),
    re.compile(r"\bauthor list\b", re.I),
    re.compile(r"\bcontributions?\b", re.I),
    re.compile(r"\backnowledg(?:e)?ments?\b", re.I),
    re.compile(r"\bethics statement\b", re.I),
    re.compile(r"\blicense\b", re.I),
]


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def norm_text(value):
    return re.sub(r"\s+", " ", value or "").strip()


def page_index(item):
    page_idx = item.get("page_idx")
    return int(page_idx) + 1 if page_idx is not None else None


def iter_nodes(nodes, depth=0):
    for node in nodes:
        yield node, depth
        yield from iter_nodes(node.get("nodes", []), depth + 1)


def find_node(nodes, node_id):
    for node, _depth in iter_nodes(nodes):
        if node.get("node_id") == node_id:
            return node
    return None


def best_page_node(nodes, page):
    candidates = []
    for node, depth in iter_nodes(nodes):
        if node.get("node_type"):
            continue
        start = node.get("start_index")
        end = node.get("end_index")
        if isinstance(start, int) and isinstance(end, int) and start <= page <= end:
            candidates.append((end - start, -depth, node))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[0][2]


def is_outline_only_node(node):
    title = (node.get("title") or "").strip().lower()
    return any(pattern.search(title) for pattern in OUTLINE_ONLY_TITLE_PATTERNS)


def page_in_outline_only_node(nodes, page):
    for node, _depth in iter_nodes(nodes):
        if not is_outline_only_node(node):
            continue
        start = node.get("start_index")
        end = node.get("end_index")
        if isinstance(start, int) and isinstance(end, int) and start <= page <= end:
            return node
    return None


def text_for_item(item):
    typ = item.get("type")
    if typ == "text":
        return item.get("text", "")
    if typ == "list":
        return "\n".join(item.get("list_items") or [])
    if typ in {"page_footnote", "aside_text"}:
        return item.get("text", "")
    if typ == "chart":
        caption = " ".join(item.get("chart_caption") or [])
        content = item.get("content") or ""
        return (caption + "\n" + content).strip()
    if typ == "image":
        caption = " ".join(item.get("image_caption") or [])
        content = item.get("content") or ""
        return (caption + "\n" + content).strip()
    if typ == "table":
        return " ".join(item.get("table_caption") or []).strip()
    if typ == "code":
        caption = " ".join(item.get("code_caption") or [])
        content = item.get("code_body") or ""
        return (caption + "\n" + content).strip()
    return ""


def caption_for_table(item):
    if item.get("type") == "code":
        return norm_text(" ".join(item.get("code_caption") or []))
    return norm_text(" ".join(item.get("table_caption") or []))


def has_table_label(caption):
    return bool(TABLE_LABEL_RE.search(caption or ""))


def has_figure_label(caption):
    return bool(FIGURE_LABEL_RE.search(caption or ""))


def is_code_table(item):
    return item.get("type") == "code" and has_table_label(caption_for_table(item))


def is_explicit_figure_table(item):
    caption = caption_for_table(item)
    return item.get("type") == "table" and has_figure_label(caption) and not has_table_label(caption)


def table_label(caption, fallback_index, used_labels=None):
    used_labels = used_labels or set()
    matches = list(re.finditer(r"\b(Table|Tab\.|Figure|Fig\.)\s*([A-Za-z]?\d+[A-Za-z]?)", caption, flags=re.IGNORECASE))
    if not matches:
        return f"Table {fallback_index}"
    table_matches = [m for m in matches if m.group(1).lower().startswith(("tab", "table"))]
    preferred = table_matches or matches
    labels = []
    for match in preferred:
        kind = "Table" if match.group(1).lower().startswith(("tab", "table")) else "Figure"
        labels.append(f"{kind} {match.group(2)}")
    for label in labels:
        if label not in used_labels:
            return label
    return labels[-1]


def table_title(caption, label):
    if not caption:
        return label
    kind, number = label.split(" ", 1)
    aliases = [kind]
    if kind == "Table":
        aliases.append("Tab.")
    elif kind == "Figure":
        aliases.append("Fig.")
    for alias in aliases:
        match = re.search(rf"\b{re.escape(alias)}\s*{re.escape(number)}\b[:.]?", caption, flags=re.IGNORECASE)
        if match:
            return caption[match.start():]
    return caption


def order_key(item):
    bbox = item.get("bbox") or [0, 0, 0, 0]
    page = page_index(item) or -1
    return [page, bbox[1], bbox[0]]


def make_table_leaf(item_index, table_number, item, mineru_dir, used_labels):
    caption = caption_for_table(item)
    label = table_label(caption, table_number, used_labels)
    used_labels.add(label)
    title = table_title(caption, label)
    return {
        "title": title or label,
        "node_id": f"table-{table_number:04d}",
        "node_type": "table_leaf",
        "source": "mineru",
        "mineru_type": item.get("type"),
        "caption_label": label,
        "page_index": page_index(item),
        "page_idx_zero_based": item.get("page_idx"),
        "bbox": item.get("bbox"),
        "order_key": order_key(item),
        "caption": caption,
        "raw_caption": caption,
        "html": item.get("table_body"),
        "code_body": item.get("code_body"),
        "image_path": str(mineru_dir / item["img_path"]) if item.get("img_path") else None,
        "footnote": item.get("table_footnote") or item.get("code_footnote") or [],
        "raw_content_index": item_index,
        "raw_mineru_item": item,
    }


def make_content_leaf(item_index, item, mineru_dir):
    typ = item.get("type")
    text = text_for_item(item)
    leaf_type = "text_leaf" if typ in TEXTUAL_TYPES else "media_leaf"
    return {
        "title": (text[:90] + "...") if len(text) > 90 else text,
        "node_id": f"{leaf_type}-{item_index:04d}",
        "node_type": leaf_type,
        "source": "mineru",
        "mineru_type": typ,
        "page_index": page_index(item),
        "page_idx_zero_based": item.get("page_idx"),
        "bbox": item.get("bbox"),
        "order_key": order_key(item),
        "text_level": item.get("text_level"),
        "text": text,
        "image_path": str(mineru_dir / item["img_path"]) if item.get("img_path") else None,
        "raw_content_index": item_index,
    }


def make_context(content, table_pos, window=4):
    contexts = []
    start = max(0, table_pos - window)
    end = min(len(content), table_pos + window + 1)
    for idx in range(start, end):
        item = content[idx]
        if item.get("type") == "page_number":
            continue
        contexts.append({
            "relative_index": idx - table_pos,
            "content_index": idx,
            "type": item.get("type"),
            "page_index": page_index(item),
            "bbox": item.get("bbox"),
            "text": text_for_item(item),
        })
    return contexts


def attach_leaves(tree, content, mineru_dir, doc):
    assignments = []
    text_count = 0
    media_count = 0
    table_count = 0
    skipped_outline_only = 0
    used_table_labels = set()
    table_parent_overrides = doc.get("table_parent_overrides", {})

    for idx, item in enumerate(content):
        typ = item.get("type")
        if typ in SKIP_TYPES:
            continue
        if is_explicit_figure_table(item):
            page = page_index(item)
            target = best_page_node(tree, page)
            if target is None:
                continue
            leaf = make_content_leaf(idx, item, mineru_dir)
            leaf["classification_note"] = "mineru_table_reclassified_as_media_leaf_because_caption_is_explicit_figure"
            target.setdefault("nodes", []).append(leaf)
            media_count += 1
            continue

        if typ == "table" or is_code_table(item):
            table_count += 1
            leaf = make_table_leaf(idx, table_count, item, mineru_dir, used_table_labels)
            override_node_id = table_parent_overrides.get(leaf["caption_label"])
            if override_node_id:
                target = find_node(tree, override_node_id)
                reason = f"manifest_override:{leaf['caption_label']}->{override_node_id}"
            else:
                target = best_page_node(tree, leaf["page_index"])
                reason = "page_range_deepest_match"
            if target is None:
                target = best_page_node(tree, leaf["page_index"])
                reason = "page_range_deepest_match_after_missing_override"
            if target is None:
                target = tree[-1] if tree else None
                reason = "fallback_last_node"
            if target is None:
                continue
            leaf["parent_node_id"] = target.get("node_id")
            leaf["assignment_reason"] = reason
            leaf["context_window"] = make_context(content, idx)
            target.setdefault("nodes", []).append(leaf)
            assignments.append({
                "table_node_id": leaf["node_id"],
                "caption_label": leaf["caption_label"],
                "page_index": leaf["page_index"],
                "bbox": leaf["bbox"],
                "raw_content_index": idx,
                "parent_node_id": target.get("node_id"),
                "parent_title": target.get("title"),
                "assignment_reason": reason,
            })
            continue

        if typ not in TEXTUAL_TYPES and typ not in MEDIA_TYPES:
            continue
        page = page_index(item)
        target = best_page_node(tree, page)
        if target is None:
            continue
        outline_node = page_in_outline_only_node(tree, page)
        if outline_node is not None:
            skipped_outline_only += 1
            outline_node["outline_only"] = True
            outline_node["outline_only_reason"] = "reference_or_back_matter_section"
            continue
        leaf = make_content_leaf(idx, item, mineru_dir)
        target.setdefault("nodes", []).append(leaf)
        if leaf["node_type"] == "text_leaf":
            text_count += 1
        else:
            media_count += 1

    for node, _depth in iter_nodes(tree):
        if "nodes" in node:
            node["nodes"].sort(key=lambda child: (
                child.get("start_index", child.get("page_index", 10**9)),
                child.get("order_key", [10**9, 10**9, 10**9])[1] if child.get("node_type") else -1,
                child.get("node_type", ""),
                child.get("node_id", ""),
            ))
    return assignments, {
        "table_leaf_count": table_count,
        "text_leaf_count": text_count,
        "media_leaf_count": media_count,
        "mineru_content_items": len(content),
        "skipped_outline_only_items": skipped_outline_only,
    }


def count_desc(node, kind):
    total = 1 if node.get("node_type") == kind else 0
    for child in node.get("nodes", []):
        total += count_desc(child, kind)
    return total


def render_md(payload):
    lines = [
        "# PageIndex + MinerU Table-Text Tree",
        "",
        f"- Document slug: {payload['slug']}",
        f"- PDF: `{payload['source_pdf']}`",
        f"- PageIndex doc: `{payload['pageindex_doc_name']}`",
        f"- Tables: {payload['evidence_counts']['table_leaf_count']}",
        f"- Text leaves: {payload['evidence_counts']['text_leaf_count']}",
        f"- Media leaves: {payload['evidence_counts']['media_leaf_count']}",
        f"- Skipped outline-only items: {payload['evidence_counts']['skipped_outline_only_items']}",
        "",
        "## Table Assignments",
    ]
    for item in payload["table_assignments"]:
        lines.append(
            f"- {item['caption_label']} | page {item['page_index']} | "
            f"{item['parent_node_id']} {item['parent_title']} | {item['assignment_reason']}"
        )

    lines.extend(["", "## Tree With Evidence Counts"])

    def render(nodes, level=0):
        indent = "  " * level
        for node in nodes:
            typ = node.get("node_type")
            if typ == "table_leaf":
                lines.append(f"{indent}- [TABLE] {node['caption_label']} {node['title']} (page {node['page_index']})")
            elif typ == "text_leaf":
                preview = node.get("text", "").replace("\n", " ")[:120]
                lines.append(f"{indent}- [TEXT] p{node['page_index']} {preview}")
            elif typ == "media_leaf":
                preview = node.get("text", "").replace("\n", " ")[:120]
                lines.append(f"{indent}- [MEDIA] p{node['page_index']} {preview}")
            else:
                outline_note = ", outline-only" if node.get("outline_only") else ""
                lines.append(
                    f"{indent}- {node.get('node_id')} {node.get('title')} "
                    f"(pages {node.get('start_index')}-{node.get('end_index')}, "
                    f"text={count_desc(node, 'text_leaf')}, media={count_desc(node, 'media_leaf')}, "
                    f"tables={count_desc(node, 'table_leaf')}{outline_note})"
                )
                if not node.get("outline_only"):
                    render(node.get("nodes", []), level + 1)

    render(payload["structure"])
    return "\n".join(lines) + "\n"


def find_mineru_content_path(output_root, doc):
    explicit = doc.get("mineru_content_list")
    if explicit:
        return Path(explicit)
    stems = [doc["slug"], Path(doc["pdf_path"]).stem]
    modes = ["hybrid_auto", "auto", "ocr", "txt"]
    for stem in stems:
        for mode in modes:
            candidate = output_root / "mineru_output" / stem / mode / f"{stem}_content_list.json"
            if candidate.exists():
                return candidate
        root = output_root / "mineru_output" / stem
        matches = list(root.glob("*/*_content_list.json")) if root.exists() else []
        if matches:
            return matches[0]
    stem = Path(doc["pdf_path"]).stem
    return output_root / "mineru_output" / stem / "hybrid_auto" / f"{stem}_content_list.json"


def build_doc(doc, manifest):
    output_root = Path(manifest["output_root"])
    structure_path = output_root / "pageindex_structures" / f"{doc['slug']}.json"
    content_path = find_mineru_content_path(output_root, doc)
    mineru_dir = content_path.parent
    out_dir = output_root / "table_text_trees"
    out_dir.mkdir(parents=True, exist_ok=True)

    pageindex_payload = load_json(structure_path)
    content = load_json(content_path)
    tree = copy.deepcopy(pageindex_payload["structure"])
    assignments, counts = attach_leaves(tree, content, mineru_dir, doc)

    payload = {
        "slug": doc["slug"],
        "source_pdf": doc["pdf_path"],
        "pageindex_doc_name": doc.get("pageindex_doc_name"),
        "page_count": doc.get("page_count"),
        "pageindex_source": str(structure_path),
        "mineru_source": str(content_path),
        "merge_policy": {
            "page_numbering": "MinerU page_idx is zero-based; PageIndex start_index/end_index are treated as one-based PDF pages.",
            "assignment": "Attach each MinerU table/text/media leaf to the deepest PageIndex node whose page range contains the leaf page.",
            "traceability": "Every table leaf keeps page_index, page_idx_zero_based, bbox, raw_content_index, raw_mineru_item, caption, html, image_path, and context_window.",
            "outline_only_sections": [pattern.pattern for pattern in OUTLINE_ONLY_TITLE_PATTERNS],
            "table_parent_overrides": doc.get("table_parent_overrides", {}),
        },
        "evidence_counts": counts,
        "table_assignments": assignments,
        "structure": tree,
    }
    out_json = out_dir / f"{doc['slug']}_table_text_tree.json"
    out_md = out_dir / f"{doc['slug']}_table_text_tree.md"
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.write_text(render_md(payload), encoding="utf-8")
    return {"slug": doc["slug"], "json": str(out_json), "md": str(out_md), "counts": counts}


def main():
    parser = argparse.ArgumentParser(description="Build table-text evidence trees from PageIndex snapshots and MinerU outputs.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--slug", action="append", help="Build only selected slug(s). Can be repeated.")
    args = parser.parse_args()

    manifest = load_json(args.manifest)
    docs = manifest["documents"]
    if args.slug:
        selected = set(args.slug)
        docs = [doc for doc in docs if doc["slug"] in selected]

    results = [build_doc(doc, manifest) for doc in docs]
    status_path = Path(manifest["output_root"]) / "table_text_tree_status.json"
    status_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
