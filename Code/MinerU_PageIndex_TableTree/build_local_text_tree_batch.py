import argparse
import json
import re
from pathlib import Path

from build_table_text_tree_batch import DEFAULT_MANIFEST, find_mineru_content_path, load_json, page_index, text_for_item


TOC_ENTRY_RE = re.compile(
    r"^\s*(?P<label>(?:\d+(?:\.\d+)*\.?|[A-Z]\.?))\s+"
    r"(?P<title>.+?)\s*(?:\.{2,}\s*)?(?P<page>\d+)\s*$"
)
HEADING_START_RE = re.compile(r"^\s*(?P<label>(?:\d+(?:\.\d+)*\.?|[A-Z]\.?))\s+(?P<title>.+?)\s*$")
BACK_MATTER_RE = re.compile(
    r"\b(references?|bibliograph(?:y|ies)|author list|contributions?|acknowledg(?:e)?ments?|ethics statement|license)\b",
    re.I,
)


def norm_text(value):
    return re.sub(r"\s+", " ", value or "").strip()


def page_count_from_content(content):
    pages = [page_index(item) for item in content if page_index(item) is not None]
    return max(pages) if pages else 0


def content_lines(item):
    typ = item.get("type")
    if typ == "list":
        return [norm_text(line) for line in item.get("list_items") or [] if norm_text(line)]
    text = norm_text(text_for_item(item))
    return [text] if text else []


def parse_toc_entries(content, max_scan_pages=4):
    entries = []
    seen = set()
    in_contents = False
    for item_index, item in enumerate(content):
        page = page_index(item)
        if page is None or page > max_scan_pages:
            continue
        for line in content_lines(item):
            if line.lower() in {"contents", "table of contents"}:
                in_contents = True
                continue
            match = TOC_ENTRY_RE.match(line)
            if not match:
                continue
            if not in_contents and item.get("text_level") != 1:
                continue
            label = clean_label(match.group("label"))
            title = clean_heading_title(label, match.group("title"))
            key = (label, title.lower())
            if key in seen:
                continue
            seen.add(key)
            entries.append(
                {
                    "label": label,
                    "title": title,
                    "toc_page_index": page,
                    "start_index": int(match.group("page")),
                    "source_content_index": item_index,
                    "source_text": line,
                }
            )
    return entries


def heading_depth(label):
    label = (label or "").rstrip(".")
    if re.match(r"^\d+(?:\.\d+)*$", label):
        return label.count(".") + 1
    return 1


def clean_heading_title(label, title):
    title = norm_text(title)
    title = re.sub(r"(?:\s*\.\s*){2,}", " ", title).strip()
    title = re.sub(r"\s+\.$", "", title).strip()
    title = re.sub(r"\s+\d+$", "", title).strip()
    return title or label


def clean_label(label):
    return (label or "").strip().rstrip(".")


def heading_candidates(content):
    candidates = []
    for item_index, item in enumerate(content):
        if item.get("type") != "text":
            continue
        text = norm_text(item.get("text"))
        if not text:
            continue
        match = HEADING_START_RE.match(text)
        if not match and item.get("text_level") is None:
            continue
        if match:
            label = clean_label(match.group("label"))
            title = clean_heading_title(label, match.group("title"))
        else:
            label = None
            title = text
        if len(title) > 120:
            continue
        candidates.append(
            {
                "label": label,
                "title": title,
                "page_index": page_index(item),
                "page_idx_zero_based": item.get("page_idx"),
                "bbox": item.get("bbox"),
                "text_level": item.get("text_level"),
                "content_index": item_index,
                "source_text": text,
            }
        )
    return candidates


def titles_match(left, right):
    left = norm_text(left).lower()
    right = norm_text(right).lower()
    return left == right or left.startswith(right) or right.startswith(left)


def attach_heading_evidence(entries, candidates):
    for entry in entries:
        best = None
        for cand in candidates:
            if cand["page_index"] is None:
                continue
            if cand.get("label") and cand["label"] != entry["label"]:
                continue
            if not titles_match(cand["title"], entry["title"]):
                continue
            distance = abs(cand["page_index"] - entry["start_index"])
            if best is None or distance < best[0]:
                best = (distance, cand)
        if best is not None and best[0] <= 1:
            cand = best[1]
            entry["start_index"] = cand["page_index"]
            entry["heading_content_index"] = cand["content_index"]
            entry["heading_bbox"] = cand["bbox"]
            entry["heading_source_text"] = cand["source_text"]
            entry["start_source"] = "mineru_heading_match"
        else:
            entry["start_source"] = "toc_page_number"
    return entries


def fallback_entries_from_headings(candidates):
    entries = []
    for cand in candidates:
        label = cand.get("label")
        if not label:
            continue
        if not re.match(r"^(?:\d+(?:\.\d+)*|[A-Z])$", label):
            continue
        entries.append(
            {
                "label": label,
                "title": cand["title"],
                "toc_page_index": None,
                "start_index": cand["page_index"],
                "source_content_index": cand["content_index"],
                "source_text": cand["source_text"],
                "heading_content_index": cand["content_index"],
                "heading_bbox": cand["bbox"],
                "heading_source_text": cand["source_text"],
                "start_source": "mineru_heading_fallback",
            }
        )
    return entries


def node_id(index):
    return f"{index:04d}"


def first_text_preview(content, start_page, end_page, start_content_index=None, end_content_index=None, max_chars=260):
    parts = []
    for idx, item in enumerate(content):
        if start_content_index is not None and idx < start_content_index:
            continue
        if end_content_index is not None and idx >= end_content_index:
            continue
        page = page_index(item)
        if page is None or page < start_page or page > end_page:
            continue
        if item.get("type") not in {"text", "list", "page_footnote", "aside_text"}:
            continue
        text = norm_text(text_for_item(item))
        if text:
            parts.append(text)
        if sum(len(part) for part in parts) > max_chars:
            break
    preview = norm_text(" ".join(parts))[:max_chars].strip()
    return preview


def build_nested_tree(entries, content, page_count):
    linear = []
    for idx, entry in enumerate(entries):
        linear.append(
            {
                "title": entry["title"],
                "node_id": node_id(idx),
                "label": entry["label"],
                "depth": heading_depth(entry["label"]),
                "start_index": entry["start_index"],
                "start_source": entry["start_source"],
                "heading_content_index": entry.get("heading_content_index"),
                "heading_bbox": entry.get("heading_bbox"),
                "toc_source_text": entry.get("source_text"),
                "needs_codex_review": True,
            }
        )

    for index, node in enumerate(linear):
        end_index = page_count
        end_content_index = None
        for next_node in linear[index + 1 :]:
            if next_node["depth"] <= node["depth"]:
                end_index = next_node["start_index"]
                end_content_index = next_node.get("heading_content_index")
                break
        node["end_index"] = max(node["start_index"], end_index)
        preview = first_text_preview(
            content,
            node["start_index"],
            node["end_index"],
            start_content_index=node.get("heading_content_index"),
            end_content_index=end_content_index,
        )
        node["summary"] = preview or f"Draft node for {node['title']}."
        node["summary_source"] = "local_mineru_preview_needs_codex_review"
        if BACK_MATTER_RE.search(node["title"]):
            node["outline_only"] = True
            node["outline_only_reason"] = "reference_or_back_matter_section"

    roots = []
    stack = []
    for raw in linear:
        node = {key: value for key, value in raw.items() if key != "depth"}
        while stack and stack[-1][0] >= raw["depth"]:
            stack.pop()
        if stack:
            stack[-1][1].setdefault("nodes", []).append(node)
        else:
            roots.append(node)
        stack.append((raw["depth"], node))
    return roots


def table_anchors(content):
    anchors = []
    table_number = 0
    for item_index, item in enumerate(content):
        typ = item.get("type")
        if typ not in {"table", "code"}:
            continue
        caption = ""
        if typ == "table":
            caption = norm_text(" ".join(item.get("table_caption") or []))
        elif typ == "code":
            caption = norm_text(" ".join(item.get("code_caption") or []))
        if not caption and typ != "table":
            continue
        table_number += 1
        anchors.append(
            {
                "table_index": table_number,
                "content_index": item_index,
                "type": typ,
                "page_index": page_index(item),
                "bbox": item.get("bbox"),
                "caption": caption,
            }
        )
    return anchors


def page_summaries(content, max_chars_per_page=1800):
    pages = {}
    for item in content:
        page = page_index(item)
        if page is None:
            continue
        text = norm_text(text_for_item(item))
        if not text:
            continue
        pages.setdefault(page, [])
        if sum(len(part) for part in pages[page]) < max_chars_per_page:
            pages[page].append(text)
    return [
        {"page_index": page, "text_preview": norm_text(" ".join(parts))[:max_chars_per_page]}
        for page, parts in sorted(pages.items())
    ]


def render_review_markdown(package):
    lines = [
        "# Local Text Tree Review Package",
        "",
        f"- Slug: `{package['slug']}`",
        f"- PDF: `{package['source_pdf']}`",
        f"- MinerU content: `{package['mineru_source']}`",
        f"- Page count: {package['page_count']}",
        "",
        "## Codex Task",
        "",
        "Review the draft tree using only the MinerU evidence below. Produce a JSON file with the same top-level shape as",
        "`reviewed_text_trees/{slug}.json`: `doc_name`, `source`, `mineru_source`, `structure`, and optional `review_notes`.",
        "Keep page indices one-based. Preserve uncertainty in `review_notes` or node-level `assignment_note` fields.",
        "",
        "## Draft Tree",
        "",
        "```json",
        json.dumps(package["draft_tree"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## TOC Entries",
    ]
    for entry in package["toc_entries"]:
        lines.append(
            f"- {entry['label']} {entry['title']} | start p{entry['start_index']} | {entry['start_source']}"
        )
    lines.extend(["", "## Heading Candidates"])
    for cand in package["heading_candidates"]:
        label = f"{cand['label']} " if cand.get("label") else ""
        lines.append(
            f"- p{cand['page_index']} idx={cand['content_index']} level={cand.get('text_level')} "
            f"{label}{cand['title']} bbox={cand.get('bbox')}"
        )
    lines.extend(["", "## Table Anchors"])
    for table in package["table_anchors"]:
        lines.append(
            f"- table-{table['table_index']:04d} p{table['page_index']} idx={table['content_index']} "
            f"bbox={table.get('bbox')} {table.get('caption')}"
        )
    lines.extend(["", "## Page Previews"])
    for page in package["pages"]:
        lines.append(f"### Page {page['page_index']}")
        lines.append(page["text_preview"])
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def build_package(doc, manifest):
    output_root = Path(manifest["output_root"])
    content_path = find_mineru_content_path(output_root, doc)
    content = load_json(content_path)
    candidates = heading_candidates(content)
    entries = parse_toc_entries(content)
    if entries:
        entries = attach_heading_evidence(entries, candidates)
        source_method = "mineru_toc_plus_heading_match"
    else:
        entries = fallback_entries_from_headings(candidates)
        entries.sort(key=lambda item: item.get("source_content_index") or 0)
        source_method = "mineru_heading_fallback_no_toc"
    entries = [entry for entry in entries if isinstance(entry.get("start_index"), int)]
    page_count = int(doc.get("page_count") or page_count_from_content(content))
    draft_tree = build_nested_tree(entries, content, page_count)
    package = {
        "slug": doc["slug"],
        "doc_name": doc.get("pageindex_doc_name") or Path(doc["pdf_path"]).name,
        "source_pdf": doc["pdf_path"],
        "mineru_source": str(content_path),
        "page_count": page_count,
        "local_tree_method": source_method,
        "toc_entries": entries,
        "heading_candidates": candidates,
        "table_anchors": table_anchors(content),
        "pages": page_summaries(content),
        "draft_tree": {
            "doc_name": doc.get("pageindex_doc_name") or Path(doc["pdf_path"]).name,
            "source": "local_mineru_draft_needs_codex_review",
            "mineru_source": str(content_path),
            "page_count": page_count,
            "structure": draft_tree,
            "review_notes": [
                "Draft generated from MinerU TOC/headings. Codex should review section boundaries, summaries, and back-matter handling before use."
            ],
        },
    }
    return package


def write_outputs(package, manifest, write_draft_reviewed=False):
    output_root = Path(manifest["output_root"])
    package_dir = output_root / "local_text_tree_review_packages"
    draft_dir = output_root / "local_text_tree_drafts"
    reviewed_dir = output_root / "reviewed_text_trees"
    package_dir.mkdir(parents=True, exist_ok=True)
    draft_dir.mkdir(parents=True, exist_ok=True)

    slug = package["slug"]
    package_json = package_dir / f"{slug}_review_package.json"
    package_md = package_dir / f"{slug}_review_package.md"
    draft_json = draft_dir / f"{slug}.json"
    package_json.write_text(json.dumps(package, ensure_ascii=False, indent=2), encoding="utf-8")
    package_md.write_text(render_review_markdown(package), encoding="utf-8")
    draft_json.write_text(json.dumps(package["draft_tree"], ensure_ascii=False, indent=2), encoding="utf-8")

    result = {
        "slug": slug,
        "review_package_json": str(package_json),
        "review_package_md": str(package_md),
        "draft_tree_json": str(draft_json),
        "toc_entry_count": len(package["toc_entries"]),
        "heading_candidate_count": len(package["heading_candidates"]),
        "table_anchor_count": len(package["table_anchors"]),
    }
    if write_draft_reviewed:
        reviewed_dir.mkdir(parents=True, exist_ok=True)
        reviewed_json = reviewed_dir / f"{slug}.json"
        reviewed = dict(package["draft_tree"])
        reviewed["source"] = "local_mineru_draft_used_as_reviewed_for_bootstrap"
        reviewed["review_notes"] = [
            "This file is a bootstrap copy of the local draft. It has not received semantic Codex review."
        ]
        reviewed_json.write_text(json.dumps(reviewed, ensure_ascii=False, indent=2), encoding="utf-8")
        result["bootstrap_reviewed_tree_json"] = str(reviewed_json)
    return result


def main():
    parser = argparse.ArgumentParser(description="Prepare local MinerU text-tree packages for Codex review.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--slug", action="append", help="Build only selected slug(s). Can be repeated.")
    parser.add_argument(
        "--write-draft-as-reviewed",
        action="store_true",
        help="Bootstrap reviewed_text_trees from the draft. Use only for smoke tests or temporary local runs.",
    )
    args = parser.parse_args()

    manifest = load_json(args.manifest)
    docs = manifest["documents"]
    if args.slug:
        selected = set(args.slug)
        docs = [doc for doc in docs if doc["slug"] in selected]

    results = [write_outputs(build_package(doc, manifest), manifest, args.write_draft_as_reviewed) for doc in docs]
    status_path = Path(manifest["output_root"]) / "local_text_tree_status.json"
    status_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
