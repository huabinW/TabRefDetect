# TabRef Context Agent Roadmap

## Completed Through v0.3

- Paper-level parallel review with LangGraph `Send`.
- Strict review gate for `prepare`, `codex`, `existing`, and `manual` modes.
- High-recall child preservation with a separate `send_to_semantic_review`
  queue flag.
- Human-feedback learning loop for candidate-rule weights and thresholds.
- Memory, reflection, and candidate-policy history artifacts.
- Pending Skill proposal workflow with explicit approval and history snapshots.

## Primary Next Objective

Move from a clear batch orchestrator to a measurable hybrid decision system:

```text
high-recall rules
  -> local classifier
  -> confidence router
  -> Codex fallback
  -> human review for disagreement
```

## Recommended Improvements

### 1. Human Gold and Evaluation

Import the user's five-paper child annotations and calculate label-0 precision,
recall, F1, per-table evidence recall, and error categories. Add grouped
paper-level splits and regression gates.

### 2. Local Classifier Gateway

Add a model adapter with the same output contract as Codex. Start with a
cross-encoder or reranker that receives table, section, full parent, and child.
Keep model identity, checkpoint hash, score, threshold, and inference time.

### 3. Calibrated Routing

Replace fixed model routing with three confidence bands:

- High-confidence relevant or irrelevant: accept the local model.
- Medium-confidence: ask Codex.
- Low-confidence or disagreement: request human review.

Thresholds must be selected on human child-level annotations, not Codex labels.

### 4. Extend Parallelism Beyond Paper-Level Codex Review

The safe paper-level Codex fan-out is complete. Later work may add table-level
parallel model inference or per-paper deterministic preprocessing after the
underlying scripts stop writing shared batch files. Preserve deterministic
aggregation and independent concurrency limits for Codex and GPU calls.

### 5. Production Persistence

Move checkpoints and task metadata from SQLite to PostgreSQL. Store large
artifacts in MinIO or an immutable filesystem layout, with content hashes and
schema versions.

### 6. Service and Worker Layer

Expose job submission, status, review, and artifact endpoints with FastAPI.
Use Celery or a comparable queue only when processing a large paper collection
requires distributed workers, retries, and resource-aware scheduling.

### 7. Annotation Interface

Build a compact review UI showing table, full parent, selected child, adjacent
context, score, model decision, and provenance. Human edits must be written as
a separate gold field rather than overwriting Codex or classifier predictions.

### 8. Observability

Instrument node duration, failure rate, candidate counts, model latency,
fallback rate, token usage, and disagreement rate. LangSmith is appropriate for
LLM traces and evaluation; OpenTelemetry is preferable for vendor-neutral
service metrics.

### 9. Reliability and Caching

Add content-addressed caching for each deterministic node, explicit artifact
schema versions, retry policies for Codex or API calls, and idempotency tests.

### 10. Downstream Citation-Error Judgment

After child selection is reliable, add a separate downstream graph for
citing-versus-cited table verification. Do not merge citation-error judgment
into the context-selector graph prematurely.

## Acceptance Criteria For the Next Major Version

- Child-level human gold exists for all five pilot papers.
- No paper-level leakage across train, validation, or test sets.
- Label-0 recall and per-table evidence recall meet declared thresholds.
- Every prediction records model, prompt, code, and input versions.
- Interrupted jobs resume without regenerating completed deterministic stages.
- A large pilot batch can be monitored, retried, and audited per document.
