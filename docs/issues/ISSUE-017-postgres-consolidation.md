# Issue #17: PostgreSQL/pgvector consolidation epic

**Status:** Canonical epic; open.
**GitHub:** [#17](https://github.com/sriharshaguthikonda/easy-local-rag/issues/17)
**Labels / priority:** `priority:P0`, `type:enhancement`.

## Dependencies

### Scope and dependency order

This epic makes PostgreSQL/pgvector the single production data platform while retaining separate durable-memory and evidence/chunk domains. Raw chunks never enter durable `memories` and evidence is never injected into every prompt.

`#2 containment -> #18 -> early #26A/#26B -> #2 final closure -> #19 -> #20A -> .memory #4 -> #20B -> .memory #5 -> #23 -> #21/#22 -> #27 -> #25 -> future GUI decision/implementation -> late #26C/#26D`

`#18` and early #26 hygiene may only preserve/sanitize and inventory; they must not delete branches, data, or Chroma rollback material. #21 stays blocked until the service contract from `.memory` #5/#23 is stable. #24/future GUI is deferred.

## Implementation slices

1. **Preserve and contain:** pass #2 credential containment, complete #18 inventory, then complete early #26A hygiene and #26B's sole approved rewrite; close #2 from #26B's clean or explicitly accepted immutable-residual terminal evidence before #19.
2. **Export before destination:** #19 creates a read-only Chroma audit plus deterministic export manifest.
3. **Database-neutral ingestion (#20A):** parsers, normalized records, deterministic chunk IDs, and package/reconciliation format run offline without database imports.
4. **Evidence store and destination integration:** `.memory` #4 supplies the evidence/source schema/import API; #20B imports packages idempotently and handles lifecycle reconciliation.
5. **Retrieval then clients:** `.memory` #5 supplies independent lexical/vector/metadata retrieval and hydration; #23 proves parity; #21/#22 consume that stable boundary; #27 validates grounding/citations; #25 controls cutover.

## Affected interfaces, files, and artifacts

- Canonical planning: `docs/plans/README.md`, `docs/plans/branch-inventory.md`, `docs/plans/issue-disposition.md`, and `docs/issues/ISSUE-018-*.md` through `ISSUE-022-*.md`.
- Migration evidence: versioned Chroma audit/export directories, SHA-256 manifests, reconciliation reports, archive-ref register, and rollback runbook.
- Maintained runtime boundary: future `easy-rag` CLI/service contracts, PostgreSQL evidence tables/APIs in `.memory`, and retrieval/citation evaluation fixtures.
- Legacy scope to retire only after acceptance: direct Chroma/Milvus clients, mixed provider/UI cores, unsafe vault writes, and duplicated GUI applications.

## Concrete actions

- Maintain an issue-to-gate matrix in the canonical plans; close legacy #5-#16 through a named maintained-path or verified-retirement mode, except closed duplicate #14, per `issue-disposition.md`.
- Require every migration run to be idempotent and emit counts, hashes, warnings, provenance, and an operator-readable reconciliation report.
- Keep search-only usable with no LLM or speech provider; require independently generated lexical and vector candidates before fusion.
- Keep read-only Chroma rollback data through #25 acceptance; do not copy Chroma persistence directories as a PostgreSQL migration method.
- Record selected UI ideas as requirements for future #24 rather than merging experimental branches wholesale.

## Verification

```powershell
gh issue view 17 --repo sriharshaguthikonda/easy-local-rag
python -m pytest tests -q
python -m py_compile streamlit_app.py rag_gui.py GUI_direct_search.py
```

At each downstream gate, run that issue's focused tests and preserve its generated manifest/report with the release evidence.

## Closure gate

Close only when one documented PostgreSQL source of truth has separate memory/evidence domains; Chroma content reconciles by counts, hashes, and provenance; retrieval and grounded-answer gates pass; the CLI operates without GUI/provider imports; and #25 has tested rollback/retirement evidence.

## Rollback constraints

Never delete, rewrite, or make Chroma unavailable before #25. A failed destination import or parity gate restores operation by using the recorded read-only Chroma snapshot/export and does not mutate the source. Destructive cleanup requires explicit final approval.

## Commit boundary

One documentation-only commit may update the epic sequencing and its linked issue plans after review; implementation commits remain scoped to their individual issue and must include their gate evidence.
