# Five-Agent Review Prompt

Use this template when reviewing table-text child candidates in parallel. Run
one Codex agent per package or shard. Replace the placeholders only; do not add
paper-specific conclusions, historical retained counts, or old decision files to
the normal review prompt.

```text
Use the installed $tabref-table-text-child-selector skill.

Task:
Perform the second-stage semantic precision review for exactly one review
package. The deterministic code stage has preserved a high-recall candidate
pool. Do not apply or invent a code precision threshold.

Input package:
{PACKAGE_JSON_OR_MD}

Write decisions to:
{DECISION_JSON}

Scope:
- Use only the package, the installed Skill, and the fields listed in this
  prompt as review inputs.
- Previous decisions or historical outputs are not inputs unless this prompt
  explicitly says the run is in comparison/audit mode.
- Table captions and table bodies are table anchors. Use them to understand
  the table, but do not create a decision for a caption unless the package
  lists it as a child candidate.
- Existing human labels are binding gold constraints. Copy them exactly.

Decision test:
Assign label 0 only when the exact child span adds information not already
visible in the table that is needed to interpret, reproduce, or verify the
table. Assign label 1 otherwise.

Retain label 0 for table-scoped:
- datasets, splits, prompts, shots, metrics, and evaluation protocols;
- baseline definitions, model/configuration identities, and training settings;
- experimental constraints, limitations, metric behavior, dataset differences,
  or cross-study comparability qualifications.

Assign label 1 for:
- pure table pointers or sentences that only say the table shows/reports/lists
  content;
- direct restatements of visible gains, gaps, rankings, or trends;
- promotional efficacy, capability, or generalization claims;
- method/model background outside the table's intended content;
- evidence for another table, figure, experiment, dataset, or model variant;
- generic background, transition text, heading fragments, or redundant text.

Output:
Write valid UTF-8 JSON with exactly this top-level shape:

{
  "schema_version": "2.0",
  "review_stage": "codex_table_scope_precision_review_v2",
  "slug": "{SLUG}",
  "reviewer": "codex",
  "decisions": [
    {
      "review_key": "...",
      "child_id": "...",
      "child_text_sha256": "...",
      "codex_label": 0,
      "semantic_role": "experimental_condition",
      "citation_support": "direct",
      "rationale": "..."
    }
  ]
}

Copy `review_key`, `child_id`, and `child_text_sha256` exactly from the
package. Every review item must have exactly one decision.

Allowed `semantic_role` values:
experimental_condition, dataset, model, metric, training_setting, method,
result_interpretation, comparison, limitation, other_support, irrelevant.

Allowed `citation_support` values:
direct, indirect, none.

For label 1, always use:
"semantic_role": "irrelevant",
"citation_support": "none".

Do not modify any file except {DECISION_JSON}.
```

Constrained-budget repeat rule:

1. Prepare review packages once.
2. Spawn one agent per missing package.
3. Ask each agent to write only its decision JSON.
4. Materialize only after all expected decision files exist.
5. If interrupted, resume from package status and missing decision files; do not
   re-review completed packages.
