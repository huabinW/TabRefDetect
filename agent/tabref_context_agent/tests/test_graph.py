from pathlib import Path
import operator
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from tabref_agent.config import AgentConfig
from tabref_agent.graph import build_graph, mermaid_diagram, route_after_review_gate
from tabref_agent.nodes import AgentNodes


def test_graph_compiles(tmp_path: Path):
    config = AgentConfig(workspace_root=tmp_path)
    graph = build_graph(config)
    assert graph.name == "tabref-context-agent-v0.3.1"


def test_diagram_contains_review_gate():
    diagram = mermaid_diagram()
    assert "review_gate" in diagram
    assert "review_one_paper" in diagram
    assert "optimize_candidate_policy" in diagram


def test_log_names_are_scoped_by_job():
    state = {"job_id": "paper batch/001"}
    assert (
        AgentNodes._log_name(state, "04_build_candidates")
        == "paper_batch_001_04_build_candidates"
    )


def test_codex_route_fans_out_one_send_per_paper():
    state = {
        "job_id": "job-1",
        "next_action": "codex",
        "review_documents": [{"slug": "paper-a"}, {"slug": "paper-b"}],
    }
    sends = route_after_review_gate(state)
    assert len(sends) == 2
    assert all(isinstance(item, Send) for item in sends)
    assert [item.arg["review_document"]["slug"] for item in sends] == [
        "paper-a",
        "paper-b",
    ]


def test_send_fanout_reduces_all_paper_results():
    class ParallelState(TypedDict, total=False):
        job_id: str
        review_documents: list[dict]
        review_document: dict
        parallel_review_results: Annotated[list[dict], operator.add]
        completed: int

    def worker(state):
        return {
            "parallel_review_results": [
                {"slug": state["review_document"]["slug"], "status": "pass"}
            ]
        }

    builder = StateGraph(ParallelState)
    builder.add_node("gate", lambda state: {})
    builder.add_node("worker", worker)
    builder.add_node(
        "aggregate",
        lambda state: {"completed": len(state["parallel_review_results"])},
    )
    builder.add_edge(START, "gate")
    builder.add_conditional_edges(
        "gate",
        lambda state: [
            Send("worker", {"review_document": document})
            for document in state["review_documents"]
        ],
    )
    builder.add_edge("worker", "aggregate")
    builder.add_edge("aggregate", END)

    result = builder.compile().invoke(
        {
            "job_id": "parallel-test",
            "review_documents": [
                {"slug": "paper-a"},
                {"slug": "paper-b"},
                {"slug": "paper-c"},
            ],
            "parallel_review_results": [],
        }
    )
    assert result["completed"] == 3
