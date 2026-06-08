import argparse
import json
import shutil
from pathlib import Path

from build_table_text_tree_batch import DEFAULT_MANIFEST, find_mineru_content_path, load_json
from resolve_table_captions import render_doc, resolve_for_table, table_records


def render_local_doc(report):
    lines = [
        render_doc(report),
        "",
        "### Local Provenance",
        "",
        f"- Local table-text tree: `{report.get('table_tree')}`",
        f"- Reviewed text tree: `{report.get('table_tree_source')}`",
        f"- Text tree source kind: `{report.get('table_tree_source_kind')}`",
        "",
    ]
    return "\n".join(lines)


def resolve_local_doc(doc, manifest):
    output_root = Path(manifest["output_root"])
    content_path = find_mineru_content_path(output_root, doc)
    tree_path = output_root / "local_table_text_trees" / f"{doc['slug']}_local_table_text_tree.json"
    if not tree_path.exists():
        raise FileNotFoundError(
            f"Local table-text tree not found for {doc['slug']}: {tree_path}. "
            "Run build_local_table_text_tree_batch.py first."
        )

    content = load_json(content_path)
    payload = load_json(tree_path)
    resolutions = [resolve_for_table(content, table) for table in table_records(payload)]
    return {
        "slug": doc["slug"],
        "mineru_content": str(content_path),
        "table_tree": str(tree_path),
        "table_tree_source": payload.get("text_tree_source"),
        "table_tree_source_kind": payload.get("text_tree_source_kind"),
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


def copy_methodology(output_root, out_dir):
    source = output_root / "table_caption_resolution" / "caption_resolution_methodology.md"
    target = out_dir / "caption_resolution_methodology.md"
    if source.exists():
        shutil.copy2(source, target)


def main():
    parser = argparse.ArgumentParser(
        description="Resolve table captions from local table-text trees and MinerU content."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--slug", action="append", help="Resolve only selected slug(s). Can be repeated.")
    args = parser.parse_args()

    manifest = load_json(args.manifest)
    docs = manifest["documents"]
    if args.slug:
        selected = set(args.slug)
        docs = [doc for doc in docs if doc["slug"] in selected]

    output_root = Path(manifest["output_root"])
    out_dir = output_root / "local_table_caption_resolution"
    out_dir.mkdir(parents=True, exist_ok=True)
    copy_methodology(output_root, out_dir)

    reports = [resolve_local_doc(doc, manifest) for doc in docs]
    out_json = out_dir / "batch_local_table_caption_resolution.json"
    out_md = out_dir / "batch_local_table_caption_resolution.md"
    out_json.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# Batch Local Table Caption Resolution", ""]
    for report in reports:
        lines.append(render_local_doc(report))
    out_md.write_text("\n".join(lines), encoding="utf-8")

    print(
        json.dumps(
            [{k: report[k] for k in ["slug", "table_count", "summary"]} for report in reports],
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
