---
name: tabref-table-text-child-selector
description: Select and audit the smallest body-text child blocks that describe a specified table in a TabRefDetect table-text tree. Use when Codex needs to identify table references, experimental conditions, datasets, models, metrics, prompts, shots, splits, baselines, training settings, ablations, or result interpretations associated with a table; return each child with its complete parent paragraph, page, bbox, MinerU indices, selection evidence, and binary labels where 0 means correct/relevant and 1 means incorrect/irrelevant. Also use to compare predictions against user-provided annotations and refine the existing selector without creating a replacement skill.
---

# TabRef Table Text Child Selector

Current version: `0.6.1`.

Find the minimal body-text spans that describe a table while preserving their complete parent paragraphs and original MinerU traceability.

## Scope

- Select table-related child blocks from a local table-text tree.
- Preserve table caption/body, parent paragraph, section, page, bbox, reading order, and MinerU indices.
- Label correct/relevant relations `0`; label incorrect/irrelevant relations `1`.
- Separate deterministic high-recall candidate generation from Codex semantic precision review.
- Treat code-stage child labels as candidates, not final semantic labels.
- Compare later user annotations against current predictions and update this same Skill.
- Support a controlled self-learning loop where candidate thresholds and rule weights can be optimized from human gold, while Skill text changes are staged for human approval.

Do not:

- Summarize, truncate, or rewrite OCR evidence.
- Infer citation errors between citing and cited papers.
- Compare table values across papers.
- Replace manual gold annotations with heuristic predictions.
- Treat Codex decisions as training gold.
- Apply a self-generated Skill patch without explicit human approval and a preserved history snapshot.
- Create another child-selector Skill when refining this workflow.

## Workflow

1. Locate the full candidate dataset and manual parent annotations.
2. Confirm parent labels use `0 = correct/relevant`, `1 = incorrect/irrelevant`.
3. Run `scripts/select_table_description_child_blocks.py`.
4. Preserve every child under a correct parent so the code stage maximizes recall.
5. For strict Popo review sets, send the complete high-recall set to Codex. Do not apply a code precision filter before semantic review.
6. Run `scripts/prepare_codex_child_review_packages.py`.
7. Partition independent papers or shards into review packages. Let the current
   conversation coordinate the run and assign packages to subagents with
   `references/subagent-review-prompt.md`.
8. Choose the subagent count from the number of independent packages, available
   concurrency, token budget, and retry state. Do not assume a fixed count. Use
   one subagent per package when practical, or queue several packages through
   fewer subagents while keeping one decision file per package.
9. Keep a coordinator manifest of expected packages, assigned decision paths,
   completion state, and validation failures. On resume, review only missing or
   invalid packages.
10. Run `scripts/run_codex_child_semantic_review.py`, or let each assigned
    subagent write exactly one package decision file.
11. Run `scripts/materialize_codex_child_review_results.py` only after every
    expected decision file passes identity, hash, schema, and completeness
    validation.
12. Report preserved code-stage candidates, semantic-review candidates, Codex
    final labels, adjudicated regression labels, and human gold separately.

Default project command:

```powershell
python scripts/select_table_description_child_blocks.py `
  --manifest manifest.json `
  --annotations outputs/manual_table_body_text_annotations/batch_manual_table_body_text_annotations.json `
  --candidates outputs/classifier_candidate_dataset/batch_table_text_classifier_candidates.json
```

The bundled script imports project helpers. Run it with the project root on `PYTHONPATH`, or execute the project copy through `run_local_table_text_training_pipeline.py`.

## Selection Logic

For a parent labeled `1`, assign all children label `1`.

For a parent labeled `0`, preserve every child and score it using:

- Exact `Table N` or `Tab. N` reference.
- Lexical overlap with table caption and table body.
- Lexical overlap with the manual parent-selection rationale.
- Evidence-family match: dataset, model, metric, training, method, or result.
- Experimental-condition and result-description cues.

The score only orders review work and controls `send_to_semantic_review`.
It must not remove children from the preserved candidate dataset or become the final label.

An active candidate policy may update:

- Signal weights.
- The review threshold.
- Force-review signals, such as explicit table references.

Policy learning must use human child-level feedback only. It may rank the
review queue, but it must not remove strict candidates before Codex review
unless human-label validation has demonstrated the required recall and
per-table coverage. Keep all original child evidence in the full dataset.

## Codex Precision Review

Judge each high-recall candidate from the table caption/body, complete parent paragraph, and exact child text.

Apply all four tests in order:

1. **Table scope:** Identify the specific current-table field, row, column,
   method, dataset, metric, protocol, comparison, or limitation governed by the
   child. An explicit table mention is neither necessary nor sufficient.
   Nearby text is not automatically table-scoped. A shared configuration may
   support several tables only when it actually governs each table.
2. **Information gain:** Check whether the same fact is already visible in the
   table or caption, or follows from a simple lookup, comparison, or arithmetic
   calculation over visible cells. If so, the child adds no supplementary
   information.
3. **Evidence utility:** Apply the counterfactual removal test: if the child were
   removed, would a careful reader lose a hidden fact needed to define table
   semantics, reproduce the setup, verify provenance, or assess limitations and
   comparability?
4. **Minimal sufficiency:** Keep the smallest complete candidate that carries
   the useful fact. Do not rewrite or truncate OCR evidence. When a candidate
   mixes a result restatement with an inseparable supplementary clause, retain
   it only if that clause independently passes the first three tests and name
   the clause-level contribution in the rationale.

Assign `0` when the child passes the table-scope, information-gain, and
evidence-utility tests. Common transferable evidence functions include:

- Defining a row, column, baseline, feature, model variant, abbreviation, or
  configuration whose meaning is not expanded in the table.
- Giving data provenance, composition, split, sampling, filtering, annotation,
  availability, domain, coverage, or difficulty conditions.
- Giving table-governing prompts, shots, decoding or inference parameters,
  training settings, evaluation protocols, metric definitions, or judge
  criteria.
- Giving a necessary limitation, metric behavior, dataset difference, or
  cross-study comparability qualification that changes how visible values
  should be interpreted.
- Linking a derived table entry to its source text, resource, or construction
  procedure when that trace is required to verify or reproduce the table.

Assign `1` when the child:

- Only says the table shows, reports, summarizes, lists, or compares content
  already visible in the table.
- Rephrases or aggregates visible values, or states a conclusion obtainable by
  a simple comparison, count, percentage, or arithmetic calculation.
- Describes a model innovation, method background, implementation detail, or
  general performance claim that is related to the topic but outside the
  table's intended content.
- Belongs to another table, figure, experiment, dataset, model variant, or
  evaluation scope.
- Directly restates a gain, gap, trend, best score, or comparison already
  visible in the table.
- Uses the table to make a promotional efficacy, capability, or generalizability
  claim without adding separate table-scoped conditions.
- Is a cross-table pointer without a new current-table fact, generic background,
  a transition, a heading fragment, unrelated, insufficiently table-scoped, or
  a weaker repetition that adds no fact beyond a more direct candidate.

Use the supplementary-context test:

```text
Does this child add information not already visible in the table that is
needed to interpret, reproduce, or verify it?
```

If the answer is no, assign `1`. A relevant parent paragraph does not make
every child relevant. A table may legitimately retain zero child spans.

Being supported by the table is not sufficient for label `0`. A result
interpretation is retained only when it contributes a necessary limitation,
metric behavior, dataset difference, protocol condition, or cross-study
qualification beyond the visible values. Do not use section distance, lexical
overlap, an explicit table reference, or a high code score as a semantic veto
or automatic acceptance rule.

Existing human labels are binding gold constraints during review. Codex must
copy them exactly and use their rationales as feedback. Code scores and lexical
overlap are ranking aids only.

Do not hard-code historical retained counts, paper-specific outcomes, benchmark
paper names, table ids, example wording, or previous decision files into a
normal review prompt. A standard review prompt should name only the current
package, output path, slug, and schema contract. Previous decisions are inputs
only in an explicit comparison or audit mode.

Record `semantic_role`, `citation_support`, and a concise rationale. The code may validate and materialize these fields, but it must not invent them.

Always return:

- `child_text`, exact offsets, and child label.
- `full_parent_text` and parent label.
- Table anchor and table body.
- Page, bbox, section, reading-order, and MinerU indices.
- Selection signals, reason, source, and review status.

## Outputs

Maintain four distinct artifacts:

- Full candidate dataset with parent and child labels.
- JSONL training samples containing table, parent, and child together.
- Codex review packages containing complete evidence.
- Final Codex-reviewed JSON/JSONL.
- Human-readable Markdown listing retained label `0` and Codex-demoted label `1` separately after every table.
- Slim human annotation JSON/Markdown containing only necessary annotation
  fields, with a top-level candidate list per table.
- Full audit JSON files containing all original table, parent, child, score,
  decision, page, bbox, and MinerU traceability fields.

Treat the score as an interpretable ranking aid, not a calibrated probability.

Mark code-stage candidates:

```text
child_label_source = code_high_recall_v2
child_review_status = awaiting_codex_semantic_review
send_to_semantic_review = true|false
candidate_policy_version = candidate-policy-...
```

Mark final model decisions:

```text
final_child_label_source = codex_skill_0_6_x_supplementary_review
```

Never present code candidates or Codex decisions as human gold labels.

The recommended repeatable path assigns each independent package to exactly one
decision file and materializes results only after every expected file exists and
passes validation. The coordinator may use any positive number of subagents and
must record package ownership. When token budget is constrained, stop after
package preparation, review only missing packages, and resume from the
coordinator manifest instead of re-reading completed outputs.

## Generalization Guardrails

- Use adjudicated Codex disagreements as a regression corpus, not as human gold
  or automatic threshold-training data.
- Derive rules from recurring evidence functions and failure modes across
  papers. Do not add a rule solely to flip an individual historical candidate.
- Keep normal review blind to prior labels, expected retained counts, and
  paper-specific conclusions.
- Evaluate revisions by evidence-family consistency, false acceptance of
  visible restatements, false rejection of hidden settings/definitions, package
  completeness, and table coverage. Do not optimize only for aggregate agreement
  on the same development papers.
- Require user-confirmed child labels before calibrating code thresholds or
  promoting regression decisions to gold.

## Table Captions

Every table should carry a `table_caption` or `table_anchor.caption` field in
final outputs. The caption is part of the table anchor, not a reviewed child
candidate. Use the caption-resolution stage to fill or normalize captions when
MinerU/Popo table anchors are empty or contain shared captions, and record the
caption source/status for audit.

Human annotation files should be compact: keep table label, table caption, table
body, parent text, child text, offsets, hashes, Codex decision, and empty human
label/rationale fields. Keep full traceability fields in separate full-audit
files.

## Controlled Self-Learning

The agent may maintain local memory artifacts:

- `learning/human_child_feedback.jsonl`: immutable human child-level labels.
- `learning/memory/events.jsonl`: human correction events and signal snapshots.
- `learning/memory/lessons.json`: short lessons derived from human feedback.
- `learning/reflections/*.json`: background review summaries.
- `learning/candidate_policy/active.json`: active rule weights and threshold.
- `learning/candidate_policy/history/*.json`: previous policy snapshots.
- `learning/skill_proposals/pending/*`: proposed Skill edits awaiting approval.
- `learning/skill_history/tabref-table-text-child-selector/*`: historical live Skill snapshots.

Allowed automatic behavior:

1. Read human feedback.
2. Evaluate the current candidate policy.
3. Search conservative rule-weight and threshold updates.
4. Activate a candidate policy only when the configured minimum recall and per-table coverage guardrails pass.
5. Write a reflection and, in propose mode, stage a Skill update proposal.

Forbidden automatic behavior:

1. Use Codex labels as human gold.
2. Delete preserved candidate evidence.
3. Change label semantics.
4. Modify live Skill files directly from a learning run.
5. Approve a pending Skill proposal without a human command.

Skill proposal approval must:

1. Show or inspect the pending proposal first.
2. Verify the live Skill version still matches the proposal base version.
3. Save a complete live Skill snapshot to history.
4. Copy only allowlisted files from the pending proposal.
5. Increment the version in both this file and `references/version.json`.

## Updating With User Gold

When the user supplies child-level annotations:

1. Preserve the supplied annotation unchanged as a gold artifact.
2. Match records using document slug, canonical table label, parent raw indices, and child offsets/text.
3. Produce false-positive and false-negative audits by evidence family and selection signal.
4. Optimize candidate-policy weights and thresholds only if guardrails pass.
5. Stage any Skill wording change in `learning/skill_proposals/pending`.
6. Modify the live Skill only after explicit approval.
7. Re-run a versioned regression corpus of independent packages with the normal
   prompt blind to previous decisions. Keep a separate holdout once sufficient
   user-confirmed labels exist.
8. Increment the version in both this file and `references/version.json`.

Use semantic versioning:

- Patch: thresholds, bug fixes, output metadata.
- Minor: new signals, annotation schema, or selection behavior.
- Major: incompatible label or input/output contract.

Read `references/output-schema.md` when modifying output fields.
