from __future__ import annotations

import argparse
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


BASE = Path(__file__).resolve().parent
DEFAULT_REVIEW_DIR = (
    BASE
    / "batch_table_text_tree"
    / "mineru_popo_comparison"
    / "popo_codex_precision_v2"
    / "review_packages"
)
DEFAULT_STATUS = DEFAULT_REVIEW_DIR / "codex_precision_v2_package_status.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def review_prompt(document: dict[str, Any]) -> str:
    package_path = Path(document["package_json"])
    decision_path = Path(document["expected_decisions"])
    return f"""Use the installed $tabref-table-text-child-selector skill.

Task:
Perform the second-stage semantic precision review for exactly one review
package. The deterministic code stage has preserved a high-recall candidate
pool. Do not apply or invent a code precision threshold.

Input package:
{package_path}

Write decisions to:
{decision_path}

Scope:
- Use only the package, the installed Skill, and the fields listed in this
  prompt as review inputs.
- Previous decisions or historical outputs are not inputs unless this prompt
  explicitly says the run is in comparison/audit mode.
- Table captions and table bodies are table anchors. Use them to understand
  the table, but do not create a decision for a caption unless the package
  lists it as a child candidate.
- Existing human labels are binding gold constraints. Copy them exactly.

Decision test:
Assign label 0 only when the exact child span adds information not already
visible in the table that is needed to interpret, reproduce, or verify the
table. Assign label 1 otherwise.

Retain label 0 for table-scoped:
- datasets, splits, prompts, shots, metrics, and evaluation protocols;
- baseline definitions, model/configuration identities, and training settings;
- experimental constraints, limitations, metric behavior, dataset differences,
  or cross-study comparability qualifications.

Assign label 1 for:
- pure table pointers or sentences that only say the table shows/reports/lists content;
- direct restatements of visible gains, gaps, rankings, or trends;
- promotional efficacy, capability, or generalization claims;
- method/model background outside the table's intended content;
- evidence for another table, figure, experiment, dataset, or model variant;
- generic background, transition text, heading fragments, or redundant text.

Write valid UTF-8 JSON with exactly this shape:
{{
  "schema_version": "2.0",
  "review_stage": "codex_table_scope_precision_review_v2",
  "slug": "{document['slug']}",
  "reviewer": "codex",
  "decisions": [
    {{
      "review_key": "...",
      "child_id": "...",
      "child_text_sha256": "...",
      "codex_label": 0,
      "semantic_role": "experimental_condition",
      "citation_support": "direct",
      "rationale": "..."
    }}
  ]
}}

Copy review_key, child_id, and child_text_sha256 exactly from the package.
Allowed semantic_role values:
experimental_condition, dataset, model, metric, training_setting, method,
result_interpretation, comparison, limitation, other_support, irrelevant.
Allowed citation_support values: direct, indirect, none.
Every item must have exactly one decision. Label 1 must use semantic_role
irrelevant and citation_support none. Do not modify any other project file.
"""


def run_document(
    document: dict[str, Any],
    codex_command: str,
    force: bool,
    dry_run: bool,
) -> dict[str, Any]:
    decision_path = Path(document["expected_decisions"])
    if decision_path.exists() and not force:
        return {
            "slug": document["slug"],
            "status": "skipped_existing",
            "decision_path": str(decision_path),
        }
    command = [
        codex_command,
        "exec",
        "--sandbox",
        "workspace-write",
        "--skip-git-repo-check",
        review_prompt(document),
    ]
    if dry_run:
        return {
            "slug": document["slug"],
            "status": "dry_run",
            "decision_path": str(decision_path),
            "review_items": document["review_items"],
            "prompt_characters": len(command[-1]),
        }
    completed = subprocess.run(
        command,
        cwd=BASE,
        check=False,
        text=True,
        capture_output=True,
    )
    return {
        "slug": document["slug"],
        "status": "pass" if completed.returncode == 0 else "fail",
        "returncode": completed.returncode,
        "decision_path": str(decision_path),
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Codex table-scope precision review over Popo strict packages."
    )
    parser.add_argument("--status", type=Path, default=DEFAULT_STATUS)
    parser.add_argument("--codex-command", default="codex")
    parser.add_argument("--parallel", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    status = load_json(args.status)
    runs = []
    with ThreadPoolExecutor(max_workers=max(1, args.parallel)) as executor:
        futures = {
            executor.submit(
                run_document,
                document,
                args.codex_command,
                args.force,
                args.dry_run,
            ): document["slug"]
            for document in status["documents"]
        }
        for future in as_completed(futures):
            runs.append(future.result())
    runs.sort(key=lambda item: item["slug"])
    all_ready = all(
        Path(document["expected_decisions"]).exists()
        for document in status["documents"]
    )
    payload = {
        "dry_run": args.dry_run,
        "parallel": args.parallel,
        "runs": runs,
        "all_decisions_ready": all_ready,
    }
    run_status_path = args.status.parent / "codex_precision_v2_run_status.json"
    run_status_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if any(run["status"] == "fail" for run in runs):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
