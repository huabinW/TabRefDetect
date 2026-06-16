from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import uuid
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from .config import AgentConfig
from .graph import build_graph, mermaid_diagram
from .learning import (
    approve_skill_proposal,
    list_skill_proposals,
    reject_skill_proposal,
)


def json_default(value: Any) -> Any:
    if hasattr(value, "_asdict"):
        return value._asdict()
    if hasattr(value, "__dict__"):
        return value.__dict__
    return str(value)


def print_json(payload: Any) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=json_default))


def load_config(
    path: Path,
    mode: str | None = None,
    create_runtime_dirs: bool = False,
) -> AgentConfig:
    config = AgentConfig.from_json(path)
    if mode:
        config = config.model_copy(update={"review_mode": mode})
    if create_runtime_dirs:
        config.resolved_output_dir.mkdir(parents=True, exist_ok=True)
        config.resolved_checkpoint_db.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(config.resolved_checkpoint_db):
            pass
    return config


def initial_state(config: AgentConfig, thread_id: str, job_id: str) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "job_id": job_id,
        "thread_id": thread_id,
        "review_mode": config.review_mode,
        "learning_mode": config.learning_mode,
        "stage": "created",
        "status": "running",
        "workspace_root": str(config.workspace_root),
        "manifest_path": str(config.resolved_manifest_path),
        "output_dir": str(config.resolved_output_dir),
        "metrics": {},
        "artifacts": {},
        "events": [],
        "errors": [],
        "parallel_review_results": [],
        "learning": {},
    }


def run_command(args: argparse.Namespace) -> None:
    config = load_config(args.config, args.mode, create_runtime_dirs=True)
    if args.force_codex_review:
        config = config.model_copy(update={"force_codex_review": True})
    thread_id = args.thread_id or f"tabref-{uuid.uuid4().hex[:12]}"
    job_id = args.job_id or thread_id
    with SqliteSaver.from_conn_string(str(config.resolved_checkpoint_db)) as saver:
        graph = build_graph(config, checkpointer=saver)
        result = graph.invoke(
            initial_state(config, thread_id, job_id),
            {
                "configurable": {"thread_id": thread_id},
                "max_concurrency": config.max_parallel_reviews,
            },
        )
    print_json(result)


def resume_command(args: argparse.Namespace) -> None:
    config = load_config(args.config, create_runtime_dirs=True)
    resume_value = {"approved": args.approve}
    with SqliteSaver.from_conn_string(str(config.resolved_checkpoint_db)) as saver:
        graph = build_graph(config, checkpointer=saver)
        result = graph.invoke(
            Command(resume=resume_value),
            {
                "configurable": {"thread_id": args.thread_id},
                "max_concurrency": config.max_parallel_reviews,
            },
        )
    print_json(result)


def status_command(args: argparse.Namespace) -> None:
    config = load_config(args.config, create_runtime_dirs=True)
    with SqliteSaver.from_conn_string(str(config.resolved_checkpoint_db)) as saver:
        graph = build_graph(config, checkpointer=saver)
        snapshot = graph.get_state(
            {"configurable": {"thread_id": args.thread_id}}
        )
    print_json(
        {
            "values": snapshot.values,
            "next": snapshot.next,
            "created_at": snapshot.created_at,
            "metadata": snapshot.metadata,
        }
    )


def plan_command(args: argparse.Namespace) -> None:
    config = load_config(args.config, args.mode)
    print_json(
        {
            "agent": "TabRef Context Agent",
            "version": "0.3.0",
            "review_mode": config.review_mode,
            "workspace_root": str(config.workspace_root),
            "manifest_path": str(config.resolved_manifest_path),
            "output_dir": str(config.resolved_output_dir),
            "checkpoint_db": str(config.resolved_checkpoint_db),
            "max_parallel_reviews": config.max_parallel_reviews,
            "learning_mode": config.learning_mode,
            "human_feedback_path": str(config.resolved_learning_feedback_path),
            "candidate_policy_path": str(config.resolved_candidate_policy_path),
            "skill_write_approval": "always_required",
            "nodes": [
                "validate_workspace",
                "build_evidence_trees",
                "audit_evidence_trees",
                "materialize_annotations",
                "build_recall_candidates",
                "prepare_review_packages",
                "review_gate",
                "review_one_paper",
                "aggregate_codex_reviews",
                "materialize_results",
                "optimize_candidate_policy",
                "background_reflection",
                "stage_skill_update",
                "finalize",
            ],
        }
    )


def diagram_command(args: argparse.Namespace) -> None:
    output = args.output
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(mermaid_diagram(), encoding="utf-8")
        print(output)
    else:
        print(mermaid_diagram())


def learning_status_command(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    print_json(
        {
            "learning_mode": config.learning_mode,
            "feedback_path": str(config.resolved_learning_feedback_path),
            "feedback_exists": config.resolved_learning_feedback_path.exists(),
            "active_policy_path": str(config.resolved_candidate_policy_path),
            "active_policy_exists": config.resolved_candidate_policy_path.exists(),
            "pending_skill_proposals": list_skill_proposals(
                config.resolved_learning_output_dir
            ),
            "skill_write_approval": "always_required",
        }
    )


def approve_skill_command(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    print_json(
        approve_skill_proposal(
            config.resolved_learning_output_dir,
            args.proposal_id,
            args.approver,
        )
    )


def reject_skill_command(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    print_json(
        reject_skill_proposal(
            config.resolved_learning_output_dir,
            args.proposal_id,
            args.approver,
            args.reason,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tabref-agent",
        description="Run the TabRef table-context LangGraph agent.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Start a new agent run.")
    run_parser.add_argument("--config", type=Path, required=True)
    run_parser.add_argument(
        "--mode",
        choices=["prepare", "codex", "existing", "manual"],
    )
    run_parser.add_argument("--thread-id")
    run_parser.add_argument("--job-id")
    run_parser.add_argument(
        "--force-codex-review",
        action="store_true",
        help="Regenerate Codex decisions even when decision files already exist.",
    )
    run_parser.set_defaults(func=run_command)

    resume_parser = subparsers.add_parser(
        "resume",
        help="Resume a manual-review interrupt.",
    )
    resume_parser.add_argument("--config", type=Path, required=True)
    resume_parser.add_argument("--thread-id", required=True)
    resume_parser.add_argument(
        "--approve",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    resume_parser.set_defaults(func=resume_command)

    status_parser = subparsers.add_parser("status", help="Inspect a saved thread.")
    status_parser.add_argument("--config", type=Path, required=True)
    status_parser.add_argument("--thread-id", required=True)
    status_parser.set_defaults(func=status_command)

    plan_parser = subparsers.add_parser("plan", help="Show the resolved run plan.")
    plan_parser.add_argument("--config", type=Path, required=True)
    plan_parser.add_argument(
        "--mode",
        choices=["prepare", "codex", "existing", "manual"],
    )
    plan_parser.set_defaults(func=plan_command)

    diagram_parser = subparsers.add_parser(
        "diagram",
        help="Print or save the Mermaid graph.",
    )
    diagram_parser.add_argument("--output", type=Path)
    diagram_parser.set_defaults(func=diagram_command)

    learning_parser = subparsers.add_parser(
        "learning-status",
        help="Inspect human feedback, active policy, and pending Skill proposals.",
    )
    learning_parser.add_argument("--config", type=Path, required=True)
    learning_parser.set_defaults(func=learning_status_command)

    approve_parser = subparsers.add_parser(
        "approve-skill",
        help="Approve one pending Skill proposal after reviewing its diff.",
    )
    approve_parser.add_argument("--config", type=Path, required=True)
    approve_parser.add_argument("--proposal-id", required=True)
    approve_parser.add_argument("--approver", required=True)
    approve_parser.set_defaults(func=approve_skill_command)

    reject_parser = subparsers.add_parser(
        "reject-skill",
        help="Reject one pending Skill proposal without modifying the live Skill.",
    )
    reject_parser.add_argument("--config", type=Path, required=True)
    reject_parser.add_argument("--proposal-id", required=True)
    reject_parser.add_argument("--approver", required=True)
    reject_parser.add_argument("--reason", required=True)
    reject_parser.set_defaults(func=reject_skill_command)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.func(args)
    except Exception as exc:
        print_json(
            {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
