# Changelog

All notable public-facing changes to this repository are summarized here.

## Unreleased

- Added the scope-aware MinerU + MinerU-Popo v2 table-context workflow.
- Added fail-closed one-to-one table-anchor matching and MinerU fallback evidence.
- Removed parent/child thresholds and per-table caps from the active pre-review
  candidate stage; scores now provide ordering metadata only.
- Added independent per-paper review packages, a standard fresh-run prompt,
  full-audit materialization, compact human templates, and anonymous config schemas.
- Kept the earlier threshold workflow as a historical compatibility path.
- Defer stable threshold or local-model calibration until enough human
  child-level annotations are available.
- Prepare local classifier or reranker replacements for Codex semantic review.

## 2026-06-17

- Added the `agent/tabref_context_agent` LangGraph workflow agent.
- Added structured local memory for the agent: core memory, SQLite long-term memory, and active memory packs.
- Added Codex Skills for table-tree auditing, table-caption resolution, and table-text child selection.
- Added reusable MinerU + PageIndex table-text tree code under `Code/MinerU_PageIndex_TableTree/`.
- Added README sections describing context-aware table reference analysis and agent engineering highlights.
