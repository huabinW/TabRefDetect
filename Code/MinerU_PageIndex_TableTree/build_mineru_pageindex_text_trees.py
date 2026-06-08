import argparse
import json
from pathlib import Path

from build_table_text_tree_batch import DEFAULT_MANIFEST, find_mineru_content_path, load_json, page_index, text_for_item
from build_local_text_tree_batch import (
    attach_heading_evidence,
    clean_label,
    fallback_entries_from_headings,
    heading_candidates,
    heading_depth,
    norm_text,
    page_count_from_content,
    page_summaries,
    parse_toc_entries,
    table_anchors,
)


PAGEINDEX_SOURCE = "local_pageindex_style_mineru_text_tree_no_llm_summary"
SIGNIFICANT_TYPES = {"text", "list", "table", "code", "page_footnote", "aside_text"}


def first_significant_content_index_by_page(content):
    first = {}
    for index, item in enumerate(content):
        page = page_index(item)
        if page is None or page in first:
            continue
        if item.get("type") not in SIGNIFICANT_TYPES:
            continue
        if norm_text(text_for_item(item)) or item.get("type") in {"table", "code"}:
            first[page] = index
    return first


def text_by_page(content):
    pages = {}
    for item in content:
        page = page_index(item)
        if page is None:
            continue
        text = norm_text(text_for_item(item))
        if text:
            pages.setdefault(page, []).append(text)
    return {page: norm_text(" ".join(parts)) for page, parts in pages.items()}


def pageindex_check_toc_local(content):
    entries = parse_toc_entries(content)
    if entries:
        return {
            "toc_content": entries,
            "toc_page_list": sorted({entry.get("toc_page_index") for entry in entries if entry.get("toc_page_index")}),
            "page_index_given_in_toc": "yes",
        }
    return {"toc_content": None, "toc_page_list": [], "page_index_given_in_toc": "no"}


def process_toc_with_page_numbers_local(toc_entries, candidates):
    entries = attach_heading_evidence([dict(entry) for entry in toc_entries], candidates)
    return entries


def process_no_toc_local(candidates):
    entries = fallback_entries_from_headings(candidates)
    entries.sort(key=lambda item: (item.get("source_content_index") if item.get("source_content_index") is not None else 10**9))
    return entries


def validate_and_truncate_physical_indices_local(entries, page_count):
    valid = []
    removed = []
    seen = set()
    for entry in entries:
        physical_index = entry.get("start_index")
        if not isinstance(physical_index, int) or physical_index < 1 or physical_index > page_count:
            removed.append(
                {
                    "title": entry.get("title"),
                    "structure": entry.get("label"),
                    "physical_index": physical_index,
                    "reason": "outside_document_page_range",
                }
            )
            continue
        key = (entry.get("label"), norm_text(entry.get("title")).lower(), physical_index)
        if key in seen:
            removed.append(
                {
                    "title": entry.get("title"),
                    "structure": entry.get("label"),
                    "physical_index": physical_index,
                    "reason": "duplicate_section_entry",
                }
            )
            continue
        seen.add(key)
        valid.append(entry)
    return valid, removed


def title_appears_near_page_start(entry, page_text):
    probe = norm_text(page_text)[:700].lower()
    if not probe:
        return "no"
    title = norm_text(entry.get("title")).lower()
    label = clean_label(entry.get("label"))
    variants = [title]
    if label:
        variants.append(norm_text(f"{label} {title}").lower())
        variants.append(norm_text(f"{label}. {title}").lower())
    return "yes" if any(variant and variant in probe for variant in variants) else "no"


def add_local_appearance_flags(entries, content):
    first_by_page = first_significant_content_index_by_page(content)
    pages = text_by_page(content)
    for entry in entries:
        physical_index = entry.get("start_index")
        heading_index = entry.get("heading_content_index")
        first_index = first_by_page.get(physical_index)
        if heading_index is not None and first_index is not None:
            entry["appear_start"] = "yes" if heading_index <= first_index + 1 else "no"
            entry["appearance_source"] = "mineru_reading_order_first_item"
        else:
            entry["appear_start"] = title_appears_near_page_start(entry, pages.get(physical_index, ""))
            entry["appearance_source"] = "page_text_prefix_match"
    return entries


def add_preface_if_needed_local(entries):
    if not entries:
        return entries
    first = entries[0].get("start_index")
    if isinstance(first, int) and first > 1:
        preface = {
            "label": "0",
            "title": "Preface",
            "start_index": 1,
            "start_source": "pageindex_style_preface_insert",
            "appear_start": "yes",
            "appearance_source": "synthetic_front_matter",
        }
        return [preface] + entries
    return entries


def back_matter_entries_from_headings(candidates, existing_titles, page_count):
    additions = []
    back_matter_terms = (
        ("references", "References"),
        ("bibliography", "Bibliography"),
        ("acknowledgement", "Acknowledgement"),
        ("acknowledgments", "Acknowledgments"),
        ("ethics statement", "Ethics Statement"),
        ("license", "License"),
        ("limitations", "Limitations"),
    )
    next_index = 0
    for cand in candidates:
        title = norm_text(cand.get("title"))
        lower = title.lower()
        canonical = None
        for term, display in back_matter_terms:
            if term in lower:
                canonical = display
                break
        if not canonical or canonical.lower() in existing_titles:
            continue
        page = cand.get("page_index")
        if not isinstance(page, int) or page < 1 or page > page_count:
            continue
        additions.append(
            {
                "label": f"BM{next_index}",
                "title": canonical,
                "start_index": page,
                "start_source": "mineru_back_matter_heading",
                "heading_content_index": cand.get("content_index"),
                "heading_bbox": cand.get("bbox"),
                "heading_source_text": cand.get("source_text"),
                "outline_only": True,
                "outline_only_reason": "reference_or_back_matter_section",
            }
        )
        existing_titles.add(canonical.lower())
        next_index += 1
    return additions


def pageindex_list_to_tree_local(flat_nodes):
    roots = []
    stack = []
    for raw in flat_nodes:
        depth = raw.pop("_depth")
        node = dict(raw)
        while stack and stack[-1][0] >= depth:
            stack.pop()
        if stack:
            stack[-1][1].setdefault("nodes", []).append(node)
        else:
            roots.append(node)
        stack.append((depth, node))
    return roots


def assign_node_ids(nodes):
    counter = 0

    def visit(node_list):
        nonlocal counter
        for node in node_list:
            node["node_id"] = f"{counter:04d}"
            counter += 1
            if node.get("nodes"):
                visit(node["nodes"])

    visit(nodes)
    return nodes


def pageindex_post_processing_local(entries, page_count):
    flat = []
    for index, entry in enumerate(entries):
        start = entry["start_index"]
        if index < len(entries) - 1:
            next_start = entries[index + 1]["start_index"]
            end = next_start - 1 if entries[index + 1].get("appear_start") == "yes" else next_start
        else:
            end = page_count
        node = {
            "title": entry["title"],
            "label": entry.get("label"),
            "start_index": start,
            "end_index": max(start, end),
            "start_source": entry.get("start_source"),
            "appearance_source": entry.get("appearance_source"),
            "appear_start": entry.get("appear_start"),
            "heading_content_index": entry.get("heading_content_index"),
            "heading_bbox": entry.get("heading_bbox"),
            "toc_source_text": entry.get("source_text"),
            "outline_only": entry.get("outline_only"),
            "outline_only_reason": entry.get("outline_only_reason"),
            "_depth": heading_depth(entry.get("label")),
        }
        flat.append({key: value for key, value in node.items() if value is not None})
    return assign_node_ids(pageindex_list_to_tree_local(flat))


def count_nodes(nodes):
    total = 0
    for node in nodes:
        total += 1
        total += count_nodes(node.get("nodes", []) or [])
    return total


def render_pageindex_style_markdown(package):
    lines = [
        "# Local PageIndex-Style MinerU Text Tree Package",
        "",
        f"- Slug: `{package['slug']}`",
        f"- PDF: `{package['source_pdf']}`",
        f"- MinerU content: `{package['mineru_source']}`",
        f"- Page count: {package['page_count']}",
        f"- PageIndex-style mode: `{package['pageindex_style_pipeline']['meta_processor_mode']}`",
        "- LLM/API summary generation: omitted",
        "",
        "## Pipeline",
        "",
        "- `check_toc`: detect TOC entries from MinerU early-page text/list blocks.",
        "- `meta_processor`: choose `process_toc_with_page_numbers` when TOC page numbers exist, otherwise `process_no_toc` from MinerU headings.",
        "- `validate_and_truncate_physical_indices`: remove section entries outside the document page range.",
        "- `add_preface_if_needed`: insert a front-matter node when the first detected section starts after page 1.",
        "- `post_processing/list_to_tree`: convert flat section entries to a nested page-range tree.",
        "",
        "## Structure Tree",
        "",
        "```json",
        json.dumps(package["draft_tree"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## TOC / Heading Entries",
    ]
    for entry in package["toc_entries"]:
        lines.append(
            f"- {entry.get('label')} {entry.get('title')} | start p{entry.get('start_index')} | "
            f"{entry.get('start_source')} | appear_start={entry.get('appear_start')}"
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


def build_pageindex_style_tree(doc, manifest):
    output_root = Path(manifest["output_root"])
    content_path = find_mineru_content_path(output_root, doc)
    content = load_json(content_path)
    page_count = int(doc.get("page_count") or page_count_from_content(content))
    candidates = heading_candidates(content)

    check_toc_result = pageindex_check_toc_local(content)
    if check_toc_result["page_index_given_in_toc"] == "yes":
        entries = process_toc_with_page_numbers_local(check_toc_result["toc_content"], candidates)
        mode = "process_toc_with_page_numbers"
    else:
        entries = process_no_toc_local(candidates)
        mode = "process_no_toc"

    entries, removed_entries = validate_and_truncate_physical_indices_local(entries, page_count)
    existing_titles = {norm_text(entry.get("title")).lower() for entry in entries}
    entries.extend(back_matter_entries_from_headings(candidates, existing_titles, page_count))
    entries.sort(
        key=lambda entry: (
            entry.get("start_index") if entry.get("start_index") is not None else 10**9,
            entry.get("heading_content_index") if entry.get("heading_content_index") is not None else 10**9,
            entry.get("source_content_index") if entry.get("source_content_index") is not None else 10**9,
        )
    )
    entries = add_preface_if_needed_local(entries)
    entries = add_local_appearance_flags(entries, content)
    structure = pageindex_post_processing_local(entries, page_count)

    package = {
        "slug": doc["slug"],
        "doc_name": doc.get("pageindex_doc_name") or Path(doc["pdf_path"]).name,
        "source_pdf": doc["pdf_path"],
        "mineru_source": str(content_path),
        "page_count": page_count,
        "local_tree_method": "pageindex_style_mineru",
        "pageindex_style_pipeline": {
            "check_toc": check_toc_result["page_index_given_in_toc"],
            "meta_processor_mode": mode,
            "validate_and_truncate_physical_indices": {
                "removed_count": len(removed_entries),
                "removed_entries": removed_entries,
            },
            "add_preface_if_needed": bool(structure and structure[0].get("title") == "Preface"),
            "post_processing": "flat MinerU TOC/heading entries converted to nested start_index/end_index tree",
            "summary_generation": "omitted",
        },
        "toc_entries": entries,
        "heading_candidates": candidates,
        "table_anchors": table_anchors(content),
        "pages": page_summaries(content),
        "draft_tree": {
            "doc_name": doc.get("pageindex_doc_name") or Path(doc["pdf_path"]).name,
            "source": PAGEINDEX_SOURCE,
            "mineru_source": str(content_path),
            "page_count": page_count,
            "structure_policy": {
                "summary": "omitted",
                "tree_source": "PageIndex-style local pipeline over MinerU TOC/heading evidence",
                "content_source": "Detailed section content is attached later from MinerU text/media/table leaves.",
            },
            "pageindex_style_pipeline": {
                "check_toc": check_toc_result["page_index_given_in_toc"],
                "meta_processor_mode": mode,
                "validate_and_truncate_physical_indices_removed": len(removed_entries),
                "add_preface_if_needed": bool(structure and structure[0].get("title") == "Preface"),
                "post_processing": "local equivalent of PageIndex post_processing/list_to_tree",
                "summary_generation": "omitted",
            },
            "structure": structure,
            "review_notes": [
                "No PageIndex service, PDF upload, LLM, or external API was used.",
                "This tree mirrors PageIndex's open-source processing stages, replacing LLM extraction with MinerU TOC and heading evidence.",
            ],
        },
    }
    return package


def write_outputs(package, manifest):
    output_root = Path(manifest["output_root"])
    package_dir = output_root / "local_text_tree_review_packages"
    draft_dir = output_root / "local_text_tree_drafts"
    reviewed_dir = output_root / "reviewed_text_trees"
    package_dir.mkdir(parents=True, exist_ok=True)
    draft_dir.mkdir(parents=True, exist_ok=True)
    reviewed_dir.mkdir(parents=True, exist_ok=True)

    slug = package["slug"]
    package_json = package_dir / f"{slug}_review_package.json"
    package_md = package_dir / f"{slug}_review_package.md"
    draft_json = draft_dir / f"{slug}.json"
    reviewed_json = reviewed_dir / f"{slug}.json"
    package_json.write_text(json.dumps(package, ensure_ascii=False, indent=2), encoding="utf-8")
    package_md.write_text(render_pageindex_style_markdown(package), encoding="utf-8")
    draft_json.write_text(json.dumps(package["draft_tree"], ensure_ascii=False, indent=2), encoding="utf-8")
    reviewed_json.write_text(json.dumps(package["draft_tree"], ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "slug": slug,
        "review_package_json": str(package_json),
        "review_package_md": str(package_md),
        "draft_tree_json": str(draft_json),
        "reviewed_tree_json": str(reviewed_json),
        "source": package["draft_tree"]["source"],
        "top_level_nodes": len(package["draft_tree"]["structure"]),
        "total_nodes": count_nodes(package["draft_tree"]["structure"]),
        "toc_entry_count": len(package["toc_entries"]),
        "heading_candidate_count": len(package["heading_candidates"]),
        "table_anchor_count": len(package["table_anchors"]),
        "meta_processor_mode": package["pageindex_style_pipeline"]["meta_processor_mode"],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Build PageIndex-style local text trees from MinerU TOC/heading evidence without LLM summaries."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--slug", action="append", help="Build only selected slug(s). Can be repeated.")
    args = parser.parse_args()

    manifest = load_json(args.manifest)
    docs = manifest["documents"]
    if args.slug:
        selected = set(args.slug)
        docs = [doc for doc in docs if doc["slug"] in selected]

    results = [write_outputs(build_pageindex_style_tree(doc, manifest), manifest) for doc in docs]
    status_path = Path(manifest["output_root"]) / "reviewed_text_tree_status.json"
    status_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
