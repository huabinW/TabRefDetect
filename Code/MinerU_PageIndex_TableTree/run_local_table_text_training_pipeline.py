import json
import subprocess
import sys
import argparse
from datetime import datetime, timezone
from pathlib import Path


BASE = Path(__file__).resolve().parent
STEPS = [
    ("build_paragraph_trees", "build_local_paragraph_table_text_tree_batch.py"),
    ("audit_paragraph_trees", "audit_local_paragraph_table_text_trees.py"),
    ("materialize_manual_annotations", "materialize_manual_table_body_text_annotations.py"),
    ("build_classifier_candidates", "build_table_text_classifier_candidates.py"),
    ("select_table_description_children", "select_table_description_child_blocks.py"),
    ("prepare_codex_child_review_packages", "prepare_codex_child_review_packages.py"),
]


def load_manifest_output_root(manifest_path):
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    return manifest["output_root"]


def main():
    parser = argparse.ArgumentParser(description="Run the local table-text training-data pipeline.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=BASE / "manifest.json",
        help="Path to the local manifest.",
    )
    parser.add_argument(
        "--selections",
        type=Path,
        required=True,
        help="Private manual parent-paragraph selections JSON.",
    )
    parser.add_argument(
        "--allow-draft",
        action="store_true",
        help="Allow locally generated draft structure trees when reviewed trees are unavailable.",
    )
    args = parser.parse_args()

    results = []
    for name, script_name in STEPS:
        script_path = BASE / script_name
        command = [sys.executable, str(script_path), "--manifest", str(args.manifest)]
        if name == "build_paragraph_trees" and args.allow_draft:
            command.append("--allow-draft")
        if name == "materialize_manual_annotations":
            command.extend(["--selections", str(args.selections)])
        completed = subprocess.run(
            command,
            cwd=BASE,
            check=False,
            text=True,
            capture_output=True,
        )
        results.append(
            {
                "step": name,
                "script": str(script_path),
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            }
        )
        if completed.returncode != 0:
            break

    payload = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if len(results) == len(STEPS) and all(
            result["returncode"] == 0 for result in results
        ) else "fail",
        "steps": results,
    }
    status_path = (
        Path(load_manifest_output_root(args.manifest))
        / "local_table_text_training_pipeline_status.json"
    )
    status_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if payload["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
