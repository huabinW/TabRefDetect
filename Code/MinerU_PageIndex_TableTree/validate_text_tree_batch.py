import argparse
import json
from pathlib import Path

from build_table_text_tree_batch import DEFAULT_MANIFEST, load_json


REQUIRED_NODE_FIELDS = {"title", "node_id", "start_index", "end_index"}


def iter_nodes(nodes, path="structure"):
    for index, node in enumerate(nodes):
        node_path = f"{path}[{index}]"
        yield node, node_path
        yield from iter_nodes(node.get("nodes", []), f"{node_path}.nodes")


def validate_payload(path):
    payload = load_json(path)
    errors = []
    warnings = []
    structure = payload.get("structure")
    if not isinstance(structure, list):
        return {"path": str(path), "ok": False, "errors": ["top-level structure must be a list"], "warnings": []}

    page_count = payload.get("page_count")
    if page_count is not None and not isinstance(page_count, int):
        errors.append("top-level page_count must be an integer when present")

    seen_ids = {}
    last_root_start = 0
    for node, node_path in iter_nodes(structure):
        missing = sorted(REQUIRED_NODE_FIELDS - set(node.keys()))
        if missing:
            errors.append(f"{node_path} missing fields: {', '.join(missing)}")

        node_id = node.get("node_id")
        if node_id in seen_ids:
            errors.append(f"{node_path} duplicate node_id {node_id!r}; first seen at {seen_ids[node_id]}")
        elif node_id is not None:
            seen_ids[node_id] = node_path

        start = node.get("start_index")
        end = node.get("end_index")
        if not isinstance(start, int) or not isinstance(end, int):
            errors.append(f"{node_path} start_index/end_index must be integers")
            continue
        if start < 1 or end < 1:
            errors.append(f"{node_path} page indices must be one-based positive integers")
        if end < start:
            errors.append(f"{node_path} end_index {end} is before start_index {start}")
        if isinstance(page_count, int) and (start > page_count or end > page_count):
            errors.append(f"{node_path} page range {start}-{end} exceeds page_count {page_count}")

        if ".nodes" not in node_path:
            if start < last_root_start:
                warnings.append(f"{node_path} root start_index decreases from previous root")
            last_root_start = start

        if node.get("outline_only") and node.get("nodes"):
            warnings.append(f"{node_path} is outline_only but still has child nodes")

    return {"path": str(path), "ok": not errors, "errors": errors, "warnings": warnings}


def reviewed_paths(manifest, slugs):
    output_root = Path(manifest["output_root"])
    docs = manifest["documents"]
    if slugs:
        selected = set(slugs)
        docs = [doc for doc in docs if doc["slug"] in selected]
    paths = []
    for doc in docs:
        explicit = doc.get("text_tree_path")
        if explicit:
            paths.append(Path(explicit))
        else:
            paths.append(output_root / "reviewed_text_trees" / f"{doc['slug']}.json")
    return paths


def main():
    parser = argparse.ArgumentParser(description="Validate local or externally reviewed text trees.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--slug", action="append", help="Validate only selected slug(s). Can be repeated.")
    parser.add_argument("--path", action="append", type=Path, help="Validate explicit text tree JSON path(s).")
    args = parser.parse_args()

    manifest = load_json(args.manifest)
    paths = args.path or reviewed_paths(manifest, args.slug)
    results = []
    for path in paths:
        if not path.exists():
            results.append({"path": str(path), "ok": False, "errors": ["file does not exist"], "warnings": []})
        else:
            results.append(validate_payload(path))
    print(json.dumps(results, ensure_ascii=False, indent=2))
    if not all(item["ok"] for item in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
