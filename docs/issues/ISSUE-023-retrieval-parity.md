# Issue #23 Plan: Chroma-to-PostgreSQL retrieval parity

[Roadmap ledger](../plans/README.md)

Status: **Open — migration gate**

GitHub: https://github.com/sriharshaguthikonda/easy-local-rag/issues/23

Parent epic: [#17](https://github.com/sriharshaguthikonda/easy-local-rag/issues/17)

Labels: `priority:P0`, `type:todo`

## Goal

Prove, with a versioned query set and reproducible reports, that the PostgreSQL
evidence search path preserves imported content and does not materially regress
retrieval quality before PostgreSQL becomes primary or Chroma is retired.
Answer generation is outside this gate.

## Dependencies and sequence

This issue executes after:

1. [#19](https://github.com/sriharshaguthikonda/easy-local-rag/issues/19)
   produces a read-only Chroma snapshot/export and reconciliation manifest.
2. [#20A](https://github.com/sriharshaguthikonda/easy-local-rag/issues/20)
   freezes deterministic source/chunk identities.
3. [`.memory` #4](https://github.com/sriharshaguthikonda/.memory/issues/4)
   imports the evidence records.
4. #20B reconciles the destination import.
5. [`.memory` #5](https://github.com/sriharshaguthikonda/.memory/issues/5)
   exposes independent lexical, vector and metadata retrieval plus hydration.
6. #21A defines CLI/service ports, transport and categorized errors; #22A
   defines provider capabilities, status, requests, streaming and cancellation.

It is the blocking gate before #21B/#22B integrate the maintained client path
and before #25 can start cutover. Legacy defects #6, #10 and #11 must appear as
regression cases rather than define the expected PostgreSQL behaviour.

Steps 3–5 are satisfied only by reciprocal comments on
[#23](https://github.com/sriharshaguthikonda/easy-local-rag/issues/23),
[#20](https://github.com/sriharshaguthikonda/easy-local-rag/issues/20),
[`.memory` #4](https://github.com/sriharshaguthikonda/.memory/issues/4), and
[`.memory` #5](https://github.com/sriharshaguthikonda/.memory/issues/5).
Each handoff records the exact provider commit SHA, API/schema version,
compatibility-test command and passing result, and package name/version used by
the parity run. Branch names, `latest`, placeholders, or mutable artifacts do
not satisfy the gate.

## Implementation slices

### 23A — Versioned fixtures and judgements

- Add a synthetic, safe fixture corpus covering HTML, medical/PubMed-like text,
  exact path/title queries, rare terms, paraphrases, long sources, neighbours,
  duplicates and deliberate lexical/vector disagreements.
- Add a versioned golden query file with query ID, category, graded relevant
  chunk IDs, exact-match expectations and neighbour requirements.
- Keep private-corpus query text and document bodies outside git. Store only a
  local manifest hash and redacted run metadata for private evaluations.

### 23B — Backend-neutral metrics

- Define one retrieval result record containing stable chunk/source IDs,
  provenance, lane ranks/scores, filters, embedding profile and latency.
- Implement Recall@K, Precision@K, MRR, nDCG@K, source diversity,
  duplicate-hit rate, citation resolvability and neighbour correctness.
- Report per-query values and failures as well as aggregates. Do not collapse
  all results into one pass/fail score.

### 23C — Snapshot adapters and reconciliation

- Run the same query preprocessing and query set against the frozen Chroma
  snapshot/export, PostgreSQL FTS, PostgreSQL vector search and fused hybrid
  search.
- Record source/chunk counts, missing and extra hashes, duplicate hashes,
  quarantined/skipped records, embedding dimensions and model profile.
- Treat Chroma as a comparison backend, not the sole relevance authority.

### 23D — Reports and automated gate

- Produce machine-readable JSON and human-readable Markdown reports with
  configuration, versions, snapshot hashes, per-query comparisons and
  approved exceptions.
- Record every exception immutably before the final run with category
  (`data-quarantine`, `judgement-error`, `known-backend-limit`, or
  `environment-only`), affected query/chunk IDs, measured delta, rationale,
  named approver, approval timestamp, maximum query/record scope, expiry, and
  mandatory retest condition. One exception may cover at most one category.
  Across all active exceptions, the union of waived query IDs must not exceed
  `min(5, floor(0.05 * golden_query_count))`, and the union of affected record
  IDs must not exceed `min(20, floor(0.01 * compared_record_count))`; a zero
  result permits no waiver. The run fails before scoring when either cumulative
  cap is exceeded. Each exception expires after 30 days or any corpus, parser,
  chunker, embedding, retrieval, or schema version change, whichever comes
  first.
- Prohibit retroactive exceptions, aggregate-only waivers, security/privacy/
  citation-resolution/data-loss waivers, threshold changes after results are
  visible, and exceptions that hide an unexplained critical regression.
- Make an unexplained critical regression return a non-zero exit code.
- Run synthetic fixtures in CI; run the private corpus locally against immutable
  snapshots.

## Affected interfaces, files and artifacts

Planned implementation files:

- `retrieval_evaluation.py` — metrics, comparison and gate logic.
- `tests/fixtures/retrieval_golden.yaml` — safe corpus/query judgements.
- `tests/test_retrieval_evaluation.py` — deterministic metric and regression tests.
- `docs/issues/ISSUE-023-retrieval-parity.md` — authoritative plan.

Generated, ignored artifacts:

- `artifacts/evaluation/baseline/manifest.json`
- `artifacts/evaluation/baseline/results.json`
- `artifacts/evaluation/baseline/report.md`

The backend adapter accepts a search request and returns typed search hits. It
must not import GUI, provider or answer-generation modules.

## Concrete actions

1. Freeze and checksum the source and destination snapshots.
2. Create synthetic source/chunk records with stable IDs and relevance grades.
3. Encode known #6/#10/#11 failures as tests.
4. Implement and unit-test each metric against hand-calculated rankings.
5. Implement Chroma-export and PostgreSQL-service adapters.
6. Run each retrieval lane independently, then the fused lane.
7. Reconcile record and hash inventories before comparing rankings.
8. Generate the per-query report and review every exception.
9. Freeze the pre-run exception record, reject any result requiring a new or
   expanded exception, and make #25 consume the final pass/fail artifact.

## Verification

```powershell
python -m pytest tests/test_retrieval_evaluation.py -q
python -m pytest tests -q
python retrieval_evaluation.py prepare --fixture tests/fixtures/retrieval_golden.yaml
python retrieval_evaluation.py run --backend chroma --snapshot $env:EASY_RAG_FROZEN_EXPORT
python retrieval_evaluation.py run --backend postgres --profile hybrid
python retrieval_evaluation.py compare --fail-on-critical-regression
python -m py_compile retrieval_evaluation.py
git diff --check
```

Private paths replace the angle-bracket argument locally and are never written
to tracked reports.

## Measurable closure gate

- 100% source/chunk/hash reconciliation for non-quarantined records.
- 100% of returned citations resolve to the recorded source and location.
- Exact title/path cases rank the expected source at position 1.
- No human-marked relevant source disappears from top 10 without an explicitly
  documented, pre-run approved, unexpired exception within the category/scope
  limits above; the cumulative union of all waived queries/records remains
  within both caps.
- Fused Recall@10 and nDCG@10 are each at least as high as both individual
  lexical and vector lanes on the golden set.
- Duplicate-hit and neighbour-order fixtures pass exactly.
- PostgreSQL p95 latency is no worse than the greater of 2 seconds or 1.5 times
  the frozen Chroma baseline on the recorded local test machine.
- The comparison command exits successfully with no unexplained critical
  regression.

## Rollback and safety constraints

- Evaluation never writes to the frozen Chroma source.
- A failed gate leaves Chroma read-only and blocks #25 cutover; thresholds are
  not weakened after results are known.
- Snapshot manifests, reports and immutable pre-run exception approvals remain
  attached to the migration batch; expired exceptions block reuse until retest.
- Sensitive corpus text never enters git, CI logs or public issue comments.

## Commit boundary

Use separate reviewed commits for:

1. safe fixtures and metric tests;
2. backend adapters and reconciliation;
3. report generation and the cutover gate.

Do not combine retrieval implementation changes, provider work or Chroma
deletion with this issue.
