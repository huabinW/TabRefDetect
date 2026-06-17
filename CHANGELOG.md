# Changelog

All notable public-facing changes to this repository are summarized here.

## Unreleased

- Continue improving the MinerU + MinerU-Popo table-context workflow.
- Calibrate table-context candidate thresholds after human child-level annotations are available.
- Prepare local classifier or reranker replacements for Codex semantic review.

## 2026-06-17

- Added the `agent/tabref_context_agent` LangGraph workflow agent.
- Added structured local memory for the agent: core memory, SQLite long-term memory, and active memory packs.
- Added Codex Skills for table-tree auditing, table-caption resolution, and table-text child selection.
- Added reusable MinerU + PageIndex table-text tree code under `Code/MinerU_PageIndex_TableTree/`.
- Added README sections describing context-aware table reference analysis and agent engineering highlights.
