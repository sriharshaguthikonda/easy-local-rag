# Issue #27 Plan: Structured grounded answers and abstention

Status: **Open — answer-safety contract**

GitHub: https://github.com/sriharshaguthikonda/easy-local-rag/issues/27

Parent epic: [#17](https://github.com/sriharshaguthikonda/easy-local-rag/issues/17)

Labels: `priority:P0`, `type:security`

## Goal

Make model generation an optional, auditable layer over retrieved evidence.
Unsupported prose, invented citations, malformed provider output and
insufficient evidence must not be displayed as a successful grounded answer.
Search-only remains fully usable without any model.

## Relationship to legacy issues

- #5 context/prompt injection becomes the untrusted-evidence fencing and
  instruction-isolation requirement here.
- #13 token budgeting becomes explicit tokenizer/profile and truncation metadata
  in the answer contract.
- #14 is closed as superseded: visible `[N]` prose alone is not validation.
- `.memory` #5 supplies stable evidence IDs, source/location provenance and
  retrieval profiles.

## Dependencies and sequence

#27 starts after:

1. `.memory` #5 exposes stable typed evidence search/hydration;
2. #23 validates retrieval and citation resolvability;
3. #21 exposes the thin CLI/service boundary;
4. #22 exposes explicit providers, capabilities, tokenizer limits and privacy
   modes.

It must pass before #25 cutover. Future GUI work renders its typed result rather
than reimplementing validation.

## Required answer contract

The structured result contains:

- answer status: `grounded`, `partial`, `insufficient_evidence`,
  `validation_failed` or `provider_failed`;
- answer text;
- claim records with answer spans and cited evidence IDs;
- evidence IDs available to the model and IDs actually cited;
- retrieval and provider profiles;
- exact context package hash;
- token budget, tokens used and truncation details;
- unsupported/uncertain claims;
- conflict, age/version, privacy and validation warnings.

Raw provider prose is never the final application contract.

## Implementation slices

### 27A — Context package and structured schema

- Build numbered, source-bounded evidence blocks from active allowed chunks.
- Fence evidence as untrusted data and place system instructions outside it.
- Preserve source title, version/date and location.
- Hash and retain the exact evidence package used for the turn.
- Define strict provider output parsing into the answer contract.

### 27B — Citation and quotation validation

- Reject cited IDs absent from the supplied context.
- Require corpus-derived factual claims to map to one or more supplied evidence
  IDs.
- Validate direct quotations against normalized source text.
- Verify source/chunk status, location resolution and privacy inclusion.
- Surface conflicting sources rather than silently selecting one.

### 27C — Abstention and failure behaviour

- Return `insufficient_evidence` for no/low results, non-answering context,
  decisive truncation, unresolved conflicts or unsupported claims.
- Return `validation_failed` for invalid IDs, quotation mismatch or malformed
  structured output.
- Return `provider_failed` for timeout/cancellation/provider errors.
- Never substitute the legacy “answer this yourself!” text or a generic answer.
- Allow one constrained structured-output repair attempt; if it fails, abstain.

### 27D — Client integration and medical profile

- Make the CLI render status, warnings, claims and citations from the typed
  result.
- Keep `search --no-model` independent of all provider imports/calls.
- Add a stricter medical/clinical profile that highlights source date/version,
  primary/guideline metadata, contradictions and educational-only scope.
- Default to evidence display when medical-profile validation is incomplete.

## Affected interfaces, files and artifacts

Planned files:

- `grounded_answer.py` — answer schema, parser, validator and abstention rules.
- `rag_prompting.py` — context fencing and structured-output instruction.
- `tests/fixtures/grounded_answer_cases.yaml` — safe deterministic cases.
- `tests/test_grounded_answer.py` — contract and validation tests.
- `localrag.py` — typed result rendering only after #21.
- `docs/issues/ISSUE-027-grounded-answers.md` — authoritative plan.

Per-turn audit artifacts contain the evidence-package hash, profiles, token
metadata and validation result. They must not expose secrets or unredacted
private content in public logs.

## Concrete actions

1. Define and serialize the structured answer/status schema.
2. Build a deterministic evidence package from `.memory` #5 search hits.
3. Fence untrusted text and preserve source boundaries/provenance.
4. Parse provider output strictly and validate every claim/citation/quotation.
5. Implement explicit abstention and provider/validation failure results.
6. Add one constrained repair attempt with the same evidence set.
7. Add the medical/clinical policy and source-age/conflict warnings.
8. Integrate the typed result into the CLI; keep search-only isolated.
9. Measure citation precision/recall, unsupported claims and abstention fixtures
   separately from #23 retrieval metrics.

## Verification

```powershell
python -m pytest tests/test_grounded_answer.py -q
python -m pytest tests -q
python -m py_compile grounded_answer.py rag_prompting.py localrag.py
python localrag.py search "evidence-only smoke" --no-model --json
python localrag.py chat --fixture tests/fixtures/grounded_answer_cases.yaml
git diff --check
```

## Measurable closure gate

- 100% of invented/nonexistent citation IDs in fixtures are rejected.
- 100% of quotation mismatches and privacy-excluded citations are detected.
- 100% of no-result, low-evidence, decisive-truncation and malformed-output
  fixtures return a non-success status.
- Fully supported fixtures have 100% citation precision and recall.
- Deterministic fixtures contain zero unmarked unsupported corpus claims.
- Prompt-injection text inside evidence cannot alter the system instruction or
  requested output schema.
- Search-only tests make zero model/provider calls.
- Every generated turn retains the exact evidence-package hash and exposes
  truncation, provider and retrieval profiles.
- Medical fixtures surface undated/old sources and contradictions.

## Rollback and safety constraints

- Generation can be disabled independently; rollback is always search-only
  evidence display.
- Never fall back from validation failure to unvalidated prose.
- A repair attempt cannot add evidence IDs or retrieve new private sources
  silently.
- Preserve failed validation/audit metadata without logging source bodies or
  secrets.
- Provider changes under #22 cannot weaken the answer contract.

## Commit boundary

Use separate reviewed commits for:

1. schema, context package and deterministic tests;
2. validator, abstention and repair boundary;
3. CLI integration and medical profile.

Do not mix retrieval ranking changes (#23), provider routing (#22) or GUI work
with this issue.
