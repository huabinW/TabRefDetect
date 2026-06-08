import argparse
import copy
import json
from pathlib import Path

from build_table_text_tree_batch import (
    DEFAULT_MANIFEST,
    OUTLINE_ONLY_TITLE_PATTERNS,
    attach_leaves,
    count_desc,
    find_mineru_content_path,
    load_json,
)


def find_reviewed_text_tree_path(output_root, doc, allow_draft=False):
    explicit = doc.get("local_text_tree_path")
    if explicit:
        return Path(explicit), "manifest_local_text_tree_path"

    reviewed = output_root / "reviewed_text_trees" / f"{doc['slug']}.json"
    if reviewed.exists():
        return reviewed, "reviewed_or_generated_text_tree"

    if allow_draft:
        draft = output_root / "local_text_tree_drafts" / f"{doc['slug']}.json"
        return draft, "local_draft_tree"

    return reviewed, "missing_codex_reviewed_text_tree"


def render_local_md(payload):
    lines = [
        "# Local Text Tree + MinerU Table-Text Tree",
        "",
        f"- Document slug: {payload['slug']}",
        f"- PDF: `{payload['source_pdf']}`",
        f"- Text tree source: `{payload['text_tree_source']}`",
        f"- Text tree source kind: `{payload['text_tree_source_kind']}`",
        f"- MinerU source: `{payload['mineru_source']}`",
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


def build_doc(doc, manifest, allow_draft=False):
    output_root = Path(manifest["output_root"])
    text_tree_path, source_kind = find_reviewed_text_tree_path(output_root, doc, allow_draft=allow_draft)
    if not text_tree_path.exists():
        raise FileNotFoundError(
            f"Local text tree not found for {doc['slug']}: {text_tree_path}. "
            "Run build_mineru_pageindex_text_trees.py or provide local_text_tree_path in the manifest. "
            "Use --allow-draft only for smoke tests."
        )

    content_path = find_mineru_content_path(output_root, doc)
    mineru_dir = content_path.parent
    out_dir = output_root / "local_table_text_trees"
    out_dir.mkdir(parents=True, exist_ok=True)

    text_tree_payload = load_json(text_tree_path)
    if text_tree_payload.get("source") == "local_pageindex_style_mineru_text_tree_no_llm_summary":
        source_kind = "local_pageindex_style_mineru_text_tree"
    elif text_tree_payload.get("source") == "local_mineru_structure_tree_no_llm_summary":
        source_kind = "local_mineru_structure_tree"
    content = load_json(content_path)
    tree = copy.deepcopy(text_tree_payload["structure"])
    assignments, counts = attach_leaves(tree, content, mineru_dir, doc)

    payload = {
        "slug": doc["slug"],
        "source_pdf": doc["pdf_path"],
        "page_count": doc.get("page_count") or text_tree_payload.get("page_count"),
        "text_tree_source": str(text_tree_path),
        "text_tree_source_kind": source_kind,
        "text_tree_doc_source": text_tree_payload.get("source"),
        "mineru_source": str(content_path),
        "merge_policy": {
            "page_numbering": "MinerU page_idx is zero-based; local text-tree start_index/end_index are one-based PDF pages.",
            "assignment": "Attach each MinerU table/text/media leaf to the deepest local text-tree node whose page range contains the leaf page.",
            "traceability": "Every table leaf keeps page_index, page_idx_zero_based, bbox, raw_content_index, raw_mineru_item, caption, html, image_path, and context_window.",
            "outline_only_sections": [pattern.pattern for pattern in OUTLINE_ONLY_TITLE_PATTERNS],
            "table_parent_overrides": doc.get("table_parent_overrides", {}),
        },
        "evidence_counts": counts,
        "table_assignments": assignments,
        "structure": tree,
    }
    out_json = out_dir / f"{doc['slug']}_local_table_text_tree.json"
    out_md = out_dir / f"{doc['slug']}_local_table_text_tree.md"
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.write_text(render_local_md(payload), encoding="utf-8")
    return {"slug": doc["slug"], "json": str(out_json), "md": str(out_md), "counts": counts}


def main():
    parser = argparse.ArgumentParser(description="Build local table-text trees from structure trees and MinerU outputs.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--slug", action="append", help="Build only selected slug(s). Can be repeated.")
    parser.add_argument("--allow-draft", action="store_true", help="Use local_text_tree_drafts when reviewed trees are missing.")
    args = parser.parse_args()

    manifest = load_json(args.manifest)
    docs = manifest["documents"]
    if args.slug:
        selected = set(args.slug)
        docs = [doc for doc in docs if doc["slug"] in selected]

    results = [build_doc(doc, manifest, allow_draft=args.allow_draft) for doc in docs]
    status_path = Path(manifest["output_root"]) / "local_table_text_tree_status.json"
    status_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
