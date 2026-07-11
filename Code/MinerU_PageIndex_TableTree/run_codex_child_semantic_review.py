import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from build_table_text_tree_batch import DEFAULT_MANIFEST, load_json


BASE = Path(__file__).resolve().parent
STATUS_PATH = None
MATERIALIZER = BASE / "materialize_codex_child_review_results.py"
SKILL_NAME = "tabref-table-text-child-selector"
PROMPT_REFERENCE = Path("references/subagent-review-prompt.md")


def candidate_skill_dirs(skill_dir=None):
    if skill_dir is not None:
        yield Path(skill_dir).expanduser().resolve()

    configured = os.environ.get("TABREF_SELECTOR_SKILL_DIR")
    if configured:
        yield Path(configured).expanduser().resolve()

    if BASE.name == "scripts" and (BASE.parent / "SKILL.md").exists():
        yield BASE.parent

    for ancestor in (BASE, *BASE.parents):
        yield ancestor / "skill" / SKILL_NAME

    yield Path.home() / ".codex" / "skills" / SKILL_NAME


def resolve_review_prompt_path(skill_dir=None):
    checked = []
    seen = set()
    for candidate in candidate_skill_dirs(skill_dir):
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        prompt_path = candidate / PROMPT_REFERENCE
        checked.append(prompt_path)
        if prompt_path.is_file():
            return prompt_path
    locations = "\n".join(f"- {path}" for path in checked)
    raise FileNotFoundError(
        f"Cannot locate {PROMPT_REFERENCE} for Skill {SKILL_NAME}. Checked:\n"
        f"{locations}"
    )


def review_prompt(document, skill_dir=None):
    prompt_path = resolve_review_prompt_path(skill_dir)
    template = prompt_path.read_text(encoding="utf-8")
    replacements = {
        "{PACKAGE_JSON_OR_MD}": str(Path(document["package_json"])),
        "{DECISION_JSON}": str(Path(document["expected_decisions"])),
        "{SLUG}": str(document["slug"]),
    }
    for placeholder, value in replacements.items():
        template = template.replace(placeholder, value)
    unresolved = [key for key in replacements if key in template]
    if unresolved:
        raise ValueError(f"Unresolved review prompt placeholders: {unresolved}")
    return template


def main():
    parser = argparse.ArgumentParser(description="Run Codex semantic review over all paper packages.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--status", type=Path, default=STATUS_PATH)
    parser.add_argument("--codex-command", default="codex")
    parser.add_argument(
        "--skill-dir",
        type=Path,
        help="Path to the installed tabref-table-text-child-selector Skill.",
    )
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
            review_prompt(document, args.skill_dir),
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
