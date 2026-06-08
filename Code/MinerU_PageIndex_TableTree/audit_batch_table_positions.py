import argparse
import json
from pathlib import Path

from build_table_text_tree_batch import (
    DEFAULT_MANIFEST,
    caption_for_table,
    find_mineru_content_path,
    is_code_table,
    is_explicit_figure_table,
    load_json,
    page_index,
)


def iter_nodes(nodes):
    for node in nodes:
        yield node
        yield from iter_nodes(node.get("nodes", []))


def table_nodes(payload):
    return [node for node in iter_nodes(payload["structure"]) if node.get("node_type") == "table_leaf"]


def table_like_items(content):
    accepted = []
    excluded_figure_like = []
    raw_tables = []
    code_tables = []
    for idx, item in enumerate(content):
        if item.get("type") == "table":
            raw_tables.append(idx)
        if is_explicit_figure_table(item):
            excluded_figure_like.append(idx)
            continue
        if item.get("type") == "table" or is_code_table(item):
            accepted.append(idx)
            if item.get("type") == "code":
                code_tables.append(idx)
    return {
        "accepted": accepted,
        "raw_tables": raw_tables,
        "code_tables": code_tables,
        "excluded_figure_like": excluded_figure_like,
    }


def bbox_equal(left, right):
    return list(left or []) == list(right or [])


def status_from_flags(flags):
    hard = {
        "raw_content_index_missing",
        "raw_content_index_out_of_range",
        "source_item_not_table_like",
        "page_mismatch",
        "bbox_mismatch",
        "override_parent_mismatch",
    }
    if hard.intersection(flags):
        return "fail"
    if flags:
        return "warn"
    return "pass"


def audit_doc(doc, manifest):
    output_root = Path(manifest["output_root"])
    content_path = find_mineru_content_path(output_root, doc)
    tree_path = output_root / "table_text_trees" / f"{doc['slug']}_table_text_tree.json"
    payload = load_json(tree_path)
    content = load_json(content_path)
    table_sets = table_like_items(content)
    accepted_indices = set(table_sets["accepted"])
    overrides = doc.get("table_parent_overrides", {})

    rows = []
    for node in table_nodes(payload):
        flags = []
        raw_idx = node.get("raw_content_index")
        source_item = None
        source_caption = ""
        if not isinstance(raw_idx, int):
            flags.append("raw_content_index_missing")
        elif raw_idx < 0 or raw_idx >= len(content):
            flags.append("raw_content_index_out_of_range")
        else:
            source_item = content[raw_idx]
            source_caption = caption_for_table(source_item)
            if raw_idx not in accepted_indices:
                flags.append("source_item_not_table_like")
            if page_index(source_item) != node.get("page_index"):
                flags.append("page_mismatch")
            if not bbox_equal(source_item.get("bbox"), node.get("bbox")):
                flags.append("bbox_mismatch")
            if not source_caption:
                flags.append("source_caption_empty")
            if source_item.get("type") == "code":
                flags.append("source_code_promoted_to_table")

        expected_parent = overrides.get(node.get("caption_label"))
        if expected_parent and node.get("parent_node_id") != expected_parent:
            flags.append("override_parent_mismatch")

        if node.get("caption_label", "").startswith("Table ") and not node.get("caption"):
            flags.append("caption_label_fallback")

        rows.append(
            {
                "status": status_from_flags(flags),
                "flags": flags,
                "table_node_id": node.get("node_id"),
                "caption_label": node.get("caption_label"),
                "title": node.get("title"),
                "page_index": node.get("page_index"),
                "bbox": node.get("bbox"),
                "raw_content_index": raw_idx,
                "source_type": source_item.get("type") if source_item else None,
                "source_caption": source_caption,
                "parent_node_id": node.get("parent_node_id"),
                "parent_title": next(
                    (item.get("parent_title") for item in payload["table_assignments"] if item.get("table_node_id") == node.get("node_id")),
                    None,
                ),
                "assignment_reason": node.get("assignment_reason"),
            }
        )

    summary = {
        "slug": doc["slug"],
        "mineru_content": str(content_path),
        "table_tree": str(tree_path),
        "raw_mineru_table_items": len(table_sets["raw_tables"]),
        "accepted_table_like_items": len(table_sets["accepted"]),
        "code_items_promoted_to_tables": len(table_sets["code_tables"]),
        "figure_like_tables_excluded": len(table_sets["excluded_figure_like"]),
        "table_tree_nodes": len(rows),
        "count_matches": len(rows) == len(table_sets["accepted"]),
        "status_counts": {
            "pass": sum(1 for row in rows if row["status"] == "pass"),
            "warn": sum(1 for row in rows if row["status"] == "warn"),
            "fail": sum(1 for row in rows if row["status"] == "fail"),
        },
    }
    return {"summary": summary, "tables": rows, "table_like_indices": table_sets}


def render_doc(report):
    s = report["summary"]
    lines = [
        f"## {s['slug']}",
        "",
        f"- raw MinerU table items: {s['raw_mineru_table_items']}",
        f"- accepted table-like items: {s['accepted_table_like_items']}",
        f"- table tree nodes: {s['table_tree_nodes']}",
        f"- count matches: {s['count_matches']}",
        f"- promoted code tables: {s['code_items_promoted_to_tables']}",
        f"- excluded figure-like tables: {s['figure_like_tables_excluded']}",
        f"- status counts: {s['status_counts']}",
        "",
    ]
    for row in report["tables"]:
        lines.append(
            f"- {row['status'].upper()} {row['caption_label']} | p{row['page_index']} "
            f"bbox={row['bbox']} | idx={row['raw_content_index']} {row['source_type']} | "
            f"{row['parent_node_id']} {row['parent_title']} | {row['assignment_reason']}"
        )
        if row["flags"]:
            lines.append(f"  - flags: {', '.join(row['flags'])}")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Audit table counts and positions for batch table-text trees.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()

    manifest = load_json(args.manifest)
    out_dir = Path(manifest["output_root"]) / "table_position_audits"
    out_dir.mkdir(parents=True, exist_ok=True)
    reports = [audit_doc(doc, manifest) for doc in manifest["documents"]]

    out_json = out_dir / "batch_table_position_audit.json"
    out_md = out_dir / "batch_table_position_audit.md"
    out_json.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Batch Table Count and Position Audit", ""]
    for report in reports:
        lines.append(render_doc(report))
    out_md.write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps([report["summary"] for report in reports], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
