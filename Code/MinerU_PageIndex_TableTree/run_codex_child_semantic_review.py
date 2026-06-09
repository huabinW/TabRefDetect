import argparse
import json
import subprocess
import sys
from pathlib import Path

from build_table_text_tree_batch import DEFAULT_MANIFEST, load_json
from prepare_codex_child_review_packages import DEFAULT_OUTPUT_DIR


BASE = Path(__file__).resolve().parent
STATUS_PATH = None
MATERIALIZER = BASE / "materialize_codex_child_review_results.py"


def review_prompt(document):
    package_path = Path(document["package_json"])
    decision_path = Path(document["expected_decisions"])
    return f"""Use the installed $tabref-table-text-child-selector skill.

Perform the Codex semantic precision-review stage for exactly one paper.

Input package:
{package_path}

Write decisions to:
{decision_path}

Read the complete package. Judge every review item from its table caption/body,
complete parent paragraph, and exact child text. The code_recall_score is only a
ranking aid and must not determine the decision.

Use 0 for correct/relevant and 1 for incorrect/irrelevant. Retain table
introductions, table-scoped experimental conditions, datasets, models, metrics,
settings, comparisons, limitations, and result interpretations that can support
later citation-reference judgment. Demote generic background, headings,
transitions, redundancy, unrelated text, and insufficiently table-scoped text.

Write valid UTF-8 JSON with this exact top-level shape:
{{
  "schema_version": "1.0",
  "review_stage": "codex_semantic_precision_review",
  "slug": "{document['slug']}",
  "reviewer": "codex",
  "decisions": [
    {{
      "review_key": "...",
      "child_id": "...",
      "child_text_sha256": "...",
      "codex_label": 0,
      "semantic_role": "table_introduction",
      "citation_support": "direct",
      "rationale": "..."
    }}
  ]
}}

Copy review_key, child_id, and child_text_sha256 exactly from the package.
Allowed semantic_role values:
table_introduction, experimental_condition, dataset, model, metric,
training_setting, method, result_interpretation, comparison, limitation,
other_support, irrelevant.
Allowed citation_support values: direct, indirect, none.
Every package item must have exactly one decision. Label 1 must use
citation_support=none. Do not modify any other project file.
"""


def main():
    parser = argparse.ArgumentParser(description="Run Codex semantic review over all paper packages.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--status", type=Path, default=STATUS_PATH)
    parser.add_argument("--codex-command", default="codex")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    manifest = load_json(args.manifest)
    status_path = args.status or (
        Path(manifest["output_root"])
        / "codex_child_review_packages"
        / "codex_child_review_package_status.json"
    )
    status = load_json(status_path)
    runs = []
    for document in status["documents"]:
        decision_path = Path(document["expected_decisions"])
        if decision_path.exists() and not args.force:
            runs.append(
                {
                    "slug": document["slug"],
                    "status": "skipped_existing",
                    "decision_path": str(decision_path),
                }
            )
            continue

        command = [
            args.codex_command,
            "exec",
            "--sandbox",
            "workspace-write",
            "--skip-git-repo-check",
            review_prompt(document),
        ]
        if args.dry_run:
            runs.append(
                {
                    "slug": document["slug"],
                    "status": "dry_run",
                    "decision_path": str(decision_path),
                    "command_prefix": command[:4],
                    "prompt_characters": len(command[-1]),
                }
            )
            continue

        completed = subprocess.run(
            command,
            cwd=BASE,
            check=False,
            text=True,
            capture_output=True,
        )
        runs.append(
            {
                "slug": document["slug"],
                "status": "pass" if completed.returncode == 0 else "fail",
                "returncode": completed.returncode,
                "decision_path": str(decision_path),
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            }
        )
        if completed.returncode != 0:
            break

    all_ready = all(Path(document["expected_decisions"]).exists() for document in status["documents"])
    materialize = None
    if not args.dry_run and all_ready:
        completed = subprocess.run(
            [sys.executable, str(MATERIALIZER), "--manifest", str(args.manifest)],
            cwd=BASE,
            check=False,
            text=True,
            capture_output=True,
        )
        materialize = {
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
        if completed.returncode != 0:
            raise SystemExit(completed.returncode)

    payload = {
        "dry_run": args.dry_run,
        "runs": runs,
        "all_decisions_ready": all_ready,
        "materialize": materialize,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
