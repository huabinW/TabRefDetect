# Standard per-paper scope review prompt

Replace only `{PACKAGE_PATH}` and `{OUTPUT_PATH}`. Do not add prior decisions, expected retained
counts, paper-specific heuristics, or gold annotations to a fresh run.

```text
You own semantic scope review for exactly one paper package. You are not alone in the workspace;
do not revert or edit anyone else's files. Read and follow the installed
tabref-table-text-child-selector Skill, then read {PACKAGE_PATH}. Write only the fresh decision
JSON to {OUTPUT_PATH}; do not modify the package, workflow code, or other decision files.

Apply the package task_definition exactly. Review every review_items child once against every
table in the paper when warranted. Table suggestions and ordering scores are non-binding
navigation aids. Retain a relation only when the child adds table-scoped supplementary
information not visible in the table and needed to interpret, reproduce, or verify it. Reject
pure pointers, result restatements, promotional claims, unrelated method descriptions,
figure-only evidence, captions, and evidence for a different experiment. Shared conditions may
map to multiple tables only with actual paper-level support. A table may have zero retained
children.

Output must match decision_schema: schema_version scope-review-decisions-v2, review_stage
scope_supplementary_context_review_v2, the exact paper_id, and exactly one decision for every
child_id. Each retained relation uses label 0, confidence in [0,1], an allowed evidence_role,
and a concrete rationale. Empty relevant_tables requires a rejection_reason. Preserve ids
exactly and do not include historical labels or results. Before finishing, validate complete
child coverage, no duplicates, valid table ids, and valid evidence roles.
```

When context or token budget is constrained, build packages once, dispatch only packages whose
decision files are absent, and run materialization only after all packages validate. Never rerun
completed packages merely to discover their counts.
