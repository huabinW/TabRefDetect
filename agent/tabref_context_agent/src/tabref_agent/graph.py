from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from .config import AgentConfig
from .nodes import AgentNodes
from .state import AgentState
from .tools import ProjectToolRunner


def route_after_review_gate(state: AgentState):
    action = state.get("next_action", "stop")
    if action == "codex":
        return [
            Send(
                "review_one_paper",
                {
                    "job_id": state["job_id"],
                    "review_document": document,
                },
            )
            for document in state.get("review_documents", [])
        ]
    if action == "materialize":
        return "materialize_results"
    return "finalize"


def route_after_materialization(state: AgentState):
    return (
        "optimize_candidate_policy"
        if state.get("learning_mode", "off") != "off"
        else "finalize"
    )


def route_after_reflection(state: AgentState):
    return (
        "stage_skill_update"
        if state.get("learning_mode") == "propose"
        else "finalize"
    )


def build_graph(
    config: AgentConfig,
    checkpointer: Any = None,
    runner: ProjectToolRunner | None = None,
):
    tool_runner = runner or ProjectToolRunner(
        workspace_root=config.workspace_root,
        output_dir=config.resolved_output_dir,
        python_executable=config.python_executable,
    )
    nodes = AgentNodes(config, tool_runner)

    builder = StateGraph(AgentState)
    builder.add_node("validate_workspace", nodes.validate_workspace)
    builder.add_node("build_evidence_trees", nodes.build_evidence_trees)
    builder.add_node("audit_evidence_trees", nodes.audit_evidence_trees)
    builder.add_node("materialize_annotations", nodes.materialize_annotations)
    builder.add_node("build_recall_candidates", nodes.build_recall_candidates)
    builder.add_node("prepare_review_packages", nodes.prepare_review_packages)
    builder.add_node("review_gate", nodes.review_gate)
    builder.add_node("review_one_paper", nodes.review_one_paper)
    builder.add_node("aggregate_codex_reviews", nodes.aggregate_codex_reviews)
    builder.add_node("materialize_results", nodes.materialize_results)
    builder.add_node("optimize_candidate_policy", nodes.optimize_candidate_policy)
    builder.add_node("background_reflection", nodes.background_reflection)
    builder.add_node("stage_skill_update", nodes.stage_skill_update)
    builder.add_node("finalize", nodes.finalize)

    builder.add_edge(START, "validate_workspace")
    builder.add_edge("validate_workspace", "build_evidence_trees")
    builder.add_edge("build_evidence_trees", "audit_evidence_trees")
    builder.add_edge("audit_evidence_trees", "materialize_annotations")
    builder.add_edge("materialize_annotations", "build_recall_candidates")
    builder.add_edge("build_recall_candidates", "prepare_review_packages")
    builder.add_edge("prepare_review_packages", "review_gate")
    builder.add_conditional_edges("review_gate", route_after_review_gate)
    builder.add_edge("review_one_paper", "aggregate_codex_reviews")
    builder.add_edge("aggregate_codex_reviews", "materialize_results")
    builder.add_conditional_edges(
        "materialize_results",
        route_after_materialization,
    )
    builder.add_edge("optimize_candidate_policy", "background_reflection")
    builder.add_conditional_edges(
        "background_reflection",
        route_after_reflection,
    )
    builder.add_edge("stage_skill_update", "finalize")
    builder.add_edge("finalize", END)
    return builder.compile(checkpointer=checkpointer, name="tabref-context-agent-v0.3.1")


def mermaid_diagram() -> str:
    return """flowchart TD
    A[validate_workspace] --> B[build_evidence_trees]
    B --> C[audit_evidence_trees]
    C --> D[materialize_annotations]
    D --> E[build_recall_candidates]
    E --> F[prepare_review_packages]
    F --> G{review_gate}
    G -->|codex| H[Send: review_one_paper x N]
    G -->|existing decisions| I[materialize_results]
    G -->|prepare only| J[finalize awaiting review]
    G -->|manual| K[LangGraph interrupt]
    K --> I
    H --> M[aggregate_codex_reviews]
    M --> I
    I --> N{learning_mode}
    N -->|off| L[finalize completed]
    N -->|analyze/propose| O[optimize_candidate_policy]
    O --> P[background_reflection]
    P -->|analyze| L
    P -->|propose| Q[stage_skill_update pending approval]
    Q --> L
"""
