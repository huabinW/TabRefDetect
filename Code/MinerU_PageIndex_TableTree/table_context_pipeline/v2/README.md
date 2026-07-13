# Scope-aware table-context workflow v2

This is the active high-recall workflow. It consumes existing MinerU content lists, Popo trees,
and audited table anchors; it does not rerun OCR.

1. `build_scope_candidate_inventory.py` performs fail-closed one-to-one table anchoring, merges
   Popo text parents with uncovered MinerU text/list/footnote evidence, and creates unique child
   spans with exact offsets.
2. `prepare_scope_review_packages.py` creates one self-contained review package per paper. Every
   child can be linked to every table; scores and suggestions are non-binding.
3. Independent subagents apply the supplementary-context test and write one decision per child.
4. `materialize_scope_reviews.py` validates complete coverage and produces a full audit, selected
   relations, readable Markdown, and a compact human annotation template.
5. `audit_reference_coverage.py` is an explicit regression-only check. It never supplies historical
   decisions to fresh semantic reviewers.

Use `STANDARD_SCOPE_REVIEW_PROMPT.md` for every fresh per-paper subagent run.

From `Code/MinerU_PageIndex_TableTree`, copy the files in
`table_context_pipeline/config/` to their `.local.json` counterparts, update
only local paths, and run:

```powershell
python table_context_pipeline\v2\build_scope_candidate_inventory.py `
  --config table_context_pipeline\config\config.local.json
python table_context_pipeline\v2\prepare_scope_review_packages.py `
  --input outputs\table_context_pipeline_v2\example_run\batch_scope_candidate_inventory.json `
  --output-dir outputs\table_context_pipeline_v2\example_run\scope_review_packages
```

After one valid decision file exists for every package:

```powershell
python table_context_pipeline\v2\materialize_scope_reviews.py `
  --input outputs\table_context_pipeline_v2\example_run\batch_scope_candidate_inventory.json `
  --decision-dir outputs\table_context_pipeline_v2\example_run\scope_review_packages `
  --output-dir outputs\table_context_pipeline_v2\example_run\scope_review_results
```

The example files contain no paper data. Keep local manifests, MinerU/Popo
outputs, decisions, and human annotations out of version control.

Run tests with:

```powershell
python -m pytest table_context_pipeline\v2\tests -q
```
