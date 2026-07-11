import sys
from pathlib import Path

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
TABLE_TREE_CODE = REPOSITORY_ROOT / "Code/MinerU_PageIndex_TableTree"
if str(TABLE_TREE_CODE) not in sys.path:
    sys.path.insert(0, str(TABLE_TREE_CODE))

from run_codex_child_semantic_review import (
    resolve_review_prompt_path,
    review_prompt,
)


def write_prompt(skill_dir: Path) -> Path:
    prompt_path = skill_dir / "references/subagent-review-prompt.md"
    prompt_path.parent.mkdir(parents=True)
    prompt_path.write_text(
        "package={PACKAGE_JSON_OR_MD}\n"
        "decision={DECISION_JSON}\n"
        "slug={SLUG}\n",
        encoding="utf-8",
    )
    return prompt_path


def test_review_prompt_uses_configured_skill_reference(tmp_path):
    skill_dir = tmp_path / "selector-skill"
    prompt_path = write_prompt(skill_dir)
    document = {
        "package_json": tmp_path / "package.json",
        "expected_decisions": tmp_path / "decision.json",
        "slug": "paper-a",
    }

    assert resolve_review_prompt_path(skill_dir) == prompt_path
    rendered = review_prompt(document, skill_dir)

    assert str(document["package_json"]) in rendered
    assert str(document["expected_decisions"]) in rendered
    assert "slug=paper-a" in rendered
    assert "{PACKAGE_JSON_OR_MD}" not in rendered
    assert "{DECISION_JSON}" not in rendered
    assert "{SLUG}" not in rendered


def test_missing_configured_skill_reports_checked_path(tmp_path, monkeypatch):
    missing = tmp_path / "missing-skill"
    monkeypatch.setenv("TABREF_SELECTOR_SKILL_DIR", str(missing))
    monkeypatch.setattr(
        "run_codex_child_semantic_review.candidate_skill_dirs",
        lambda skill_dir=None: iter([missing]),
    )

    with pytest.raises(FileNotFoundError, match="subagent-review-prompt.md"):
        resolve_review_prompt_path(missing)
