# Issue #14: Render inline source citations

**Status:** CLOSED as a GitHub duplicate; no independent implementation work remains.
**GitHub:** https://github.com/sriharshaguthikonda/easy-local-rag/issues/14
**Labels / priority:** `priority:P1`, `type:enhancement`
**Dependencies:** #27 is the canonical answer-grounding, citation-validation, and abstention issue; #5 and #13 remain separate open acceptance requirements.

## Implementation slices

1. No new slice is authorized by #14.
2. Retain existing numbered-source and missing-citation behavior only as historical evidence.
3. Route invented-ID, quote-match, claim-support, conflict, and abstention work to #27.

## Affected interfaces, files, and artifacts

- Historical evidence: `streamlit_app.py`, `rag_prompting.py`, and citation tests.
- Canonical future contract: `docs/issues/ISSUE-027-grounded-answers.md`.
- No source file is owned or changed by this duplicate issue.

## Concrete actions

- Keep the GitHub duplicate state and link #27 as the active destination.
- Do not claim the closed duplicate proves #27's structured validation gates.
- Reopen #14 only if GitHub issue taxonomy changes and #27 no longer owns the work.

## Verification

```powershell
gh api repos/sriharshaguthikonda/easy-local-rag/issues/14 --jq '{state, state_reason}'
python -m pytest tests/test_streamlit_citations_config.py -q -p no:cacheprovider
```

The current source-string citation assertion is retained as known historical evidence; its separate failure does not convert this duplicate into an implementation issue.

## Closure gate, rollback constraints, and commit boundary

- **Closure gate:** already satisfied administratively by GitHub's `closed` / `duplicate` state and the explicit #27 destination link.
- **Rollback constraint:** do not reopen or create source changes from this plan without a new issue-triage decision.
- **Commit boundary:** none; this file is historical duplicate disposition evidence only.
