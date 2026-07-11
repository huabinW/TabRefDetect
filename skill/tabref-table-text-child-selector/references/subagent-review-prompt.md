# Subagent Review Prompt

Use this template for one independent table-text review package. The current
conversation coordinates the run. It may start one or more subagents, queue
packages, retry failures, and materialize results after validation. The number
of subagents is not part of the semantic method.

Replace the placeholders only. Do not add paper-specific conclusions,
historical retained counts, expected labels, or old decision files to a normal
review prompt.

```text
Use the installed $tabref-table-text-child-selector skill.

Task:
Perform second-stage semantic precision review for exactly one review package.
The deterministic code stage preserved a high-recall candidate pool. Do not
apply or invent a code precision threshold.

Input package:
{PACKAGE_JSON_OR_MD}

Write decisions to:
{DECISION_JSON}

Package slug:
{SLUG}

Scope:
- Use only the package, the installed Skill, and this schema contract.
- Do not read previous decisions or historical outputs unless the coordinator
  explicitly declares comparison/audit mode.
- Use table captions and bodies as anchors. Do not review a caption unless the
  package explicitly lists it as a child candidate.
- Copy existing human labels exactly. They are binding gold constraints.

Apply four tests to every exact child span:
1. Table scope: identify the current-table element or setting governed by the
   child. Explicit table mention and physical proximity are not sufficient.
2. Information gain: reject facts already visible in the table/caption or
   obtainable by simple lookup, comparison, count, percentage, or arithmetic.
3. Evidence utility: retain only a hidden fact needed to define table semantics,
   reproduce the setup, verify provenance, or assess limitations/comparability.
4. Minimal sufficiency: judge the exact child without rewriting it. For mixed
   text, retain only when an inseparable clause independently passes tests 1-3,
   and name that contribution in the rationale.

Assign label 0 for table-scoped supplementary evidence such as:
- definitions of rows, columns, baselines, features, model variants, or configs;
- data provenance, composition, split, filtering, annotation, domain, or access;
- prompts, shots, decoding/inference parameters, training/evaluation protocols,
  metric definitions, or judge criteria;
- limitations, metric behavior, dataset differences, or cross-study
  qualifications that change interpretation of visible values;
- source text or construction procedures needed to verify derived table entries.

Assign label 1 for:
- pure table pointers and visible result restatements;
- conclusions derivable by simple comparison or arithmetic over table cells;
- promotional efficacy/capability/generalization claims;
- related method background that does not govern the current table;
- evidence for another table, figure, experiment, dataset, or model variant;
- generic transitions, fragments, unrelated text, cross-table pointers without
  a new current-table fact, or weaker repetitions with no added fact.

For every rationale, state the current-table scope, whether the fact is visible
in the anchor, and the concrete interpretation/reproduction/verification value
or the precise rejection reason. Do not cite code score as semantic evidence.

Output valid UTF-8 JSON with exactly this top-level shape:

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

Copy review_key, child_id, and child_text_sha256 exactly. Return exactly one
decision for every review item.

Allowed semantic_role values:
experimental_condition, dataset, model, metric, training_setting, method,
result_interpretation, comparison, limitation, other_support, irrelevant.

Allowed citation_support values:
direct, indirect, none.

For label 1, always use semantic_role=irrelevant and citation_support=none.
Do not modify any file except {DECISION_JSON}.
```

Coordinator repeat rule:

1. Prepare packages and an expected-package manifest once.
2. Assign each package one unique decision path; choose any positive number of
   subagents appropriate for the current run.
3. Review only missing or invalid package decisions.
4. Validate identities, hashes, schema, and completeness before materializing.
5. Keep fresh review separate from later comparison/adjudication mode.
