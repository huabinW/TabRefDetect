import json
from pathlib import Path

from tabref_agent.learning import (
    DEFAULT_POLICY,
    approve_skill_proposal,
    build_active_memory_pack,
    evaluate_policy,
    list_skill_proposals,
    stage_skill_proposal,
    upsert_memory_item,
)


def example(candidate_id, child_id, score_signal, gold_label):
    candidate = {
        "slug": "paper",
        "table": {"canonical_label": "Table 1"},
        "candidate_id": candidate_id,
        "child": {
            "child_id": child_id,
            "child_selection_signals": {
                "exact_table_reference": score_signal == "exact",
                "table_text_overlap": ["model"] if score_signal == "overlap" else [],
                "annotation_rationale_overlap": [],
                "evidence_family": "method",
                "evidence_family_match": score_signal == "family",
                "condition_match": False,
                "result_match": False,
            },
        },
    }
    feedback = {
        "slug": "paper",
        "table_label": "Table 1",
        "candidate_id": candidate_id,
        "child_id": child_id,
        "gold_label": gold_label,
    }
    return candidate, feedback


def test_policy_metrics_use_zero_as_relevant():
    metrics = evaluate_policy(
        [
            example("parent-a", "child-a", "exact", 0),
            example("parent-a", "child-b", "none", 1),
        ],
        DEFAULT_POLICY,
    )
    assert metrics["true_positive"] == 1
    assert metrics["false_positive"] == 1
    assert metrics["recall"] == 1.0


def test_skill_proposal_requires_explicit_approval_and_preserves_history(
    tmp_path: Path,
):
    skill_dir = tmp_path / "skill"
    (skill_dir / "references").mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "# Selector\n\nCurrent version: `0.3.0`.\n\n"
        "## Updating With User Gold\n",
        encoding="utf-8",
    )
    (skill_dir / "references/version.json").write_text(
        json.dumps({"version": "0.3.0"}),
        encoding="utf-8",
    )
    learning_root = tmp_path / "learning"
    proposal = stage_skill_proposal(
        skill_dir,
        learning_root,
        {
            "reflection_path": "reflection.json",
            "lessons": [
                {
                    "rule": "Keep implicit condition evidence.",
                    "support": 4,
                }
            ],
        },
    )
    assert proposal["status"] == "pending_human_approval"
    assert json.loads(
        (skill_dir / "references/version.json").read_text(encoding="utf-8")
    )["version"] == "0.3.0"
    assert len(list_skill_proposals(learning_root)) == 1

    approved = approve_skill_proposal(
        learning_root,
        proposal["proposal_id"],
        "human-reviewer",
    )
    assert approved["status"] == "approved"
    assert json.loads(
        (skill_dir / "references/version.json").read_text(encoding="utf-8")
    )["version"] == "0.3.1"
    assert Path(approved["history_snapshot"]).exists()


def test_active_memory_pack_uses_budgeted_retrieval(tmp_path: Path):
    learning_root = tmp_path / "learning"
    db_path = learning_root / "memory" / "memory.sqlite"
    for index in range(5):
        upsert_memory_item(
            db_path,
            {
                "memory_type": "case_feedback",
                "scope": f"paper-{index}::Table 1",
                "subject": f"candidate-{index}",
                "content": f"Human feedback about model dataset evidence {index}",
                "tags": ["human_gold", "table_context"],
                "evidence": {"index": index},
                "status": "active",
                "confidence": 1.0,
                "source": "test",
                "task_key": "table_context_selection",
            },
        )

    pack = build_active_memory_pack(
        learning_root,
        db_path=db_path,
        query="model dataset evidence",
        max_items=2,
    )

    assert pack["budget"]["available_items"] == 5
    assert pack["budget"]["selected_items"] == 2
    assert len(pack["core_memory"]) > 0
    assert len(pack["retrieved_memory"]) == 2
    assert pack["active_memory_pack_path"].endswith("active_memory_pack.json")
