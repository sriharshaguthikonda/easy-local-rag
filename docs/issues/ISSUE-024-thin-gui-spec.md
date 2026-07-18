# Issue #24 Plan: Future thin GUI specification

Status: **Open — specification only**

GitHub: https://github.com/sriharshaguthikonda/easy-local-rag/issues/24

Parent epic: [#17](https://github.com/sriharshaguthikonda/easy-local-rag/issues/17)

Labels: `priority:P2`, `type:enhancement`

## Decision

This issue documents the future GUI contract and the useful UX ideas found in
experimental branches. It does **not** implement a GUI and closes as a
documentation deliverable before #25 cutover.

No existing GUI branch or PR #1 is a merge path.

## Dependencies and sequence

- [#18](https://github.com/sriharshaguthikonda/easy-local-rag/issues/18)
  supplies the branch/PR inventory and salvage decision.
- #21, #22, #23, and #27 supply the planned record names and safety boundaries;
  #24 may specify against their accepted plans and does not wait for their
  implementations.

This specification closes after its documentation and PR #1 disposition are
recorded, before #25. The #25 transition checklist owns opening the later GUI
implementation issue after successful cutover. That later issue is blocked at
minimum by completed #22 provider modes and completed #25 cutover; it also
consumes the completed #21/#23/#27 contracts.

## Implementation slices

### 24A — Salvage matrix

Record each experimental idea as:

- retain as a requirement;
- port later through the service contract;
- discard as architecture-specific;
- preserve only as regression evidence.

Retain the three-pane layout, search-only mode, structured evidence inspection,
score breakdown, background progress/cancellation, connection status, themes
and safe export. Discard direct database access, subprocess retrieval,
provider-owned widgets, hidden fallback, module-import initialization and
unvalidated conversation import.

### 24B — Client contract

Specify the GUI-facing records and operations:

- search request/result and filters;
- chunk hydration and neighbours;
- structured grounded answer and citations;
- provider/privacy status;
- health, progress, cancellation and typed errors;
- non-secret resolved configuration.

The browser/widget layer receives records from a narrow local service. It never
holds PostgreSQL credentials or imports Chroma.

### 24C — Interaction and safety rules

Specify search-only as the default usable path, explicit generation opt-in,
responsive cancellation, preserved unsent text, safe source opening, redacted
exports and visible provider/data-egress state.

### 24D — Post-cutover transition

Specify the title, scope, prerequisites, and handoff text that #25 will use to
open the implementation issue after cutover. Toolkit selection, issue creation,
and application code are intentionally outside #24.

## Affected interfaces, files and artifacts

This specification changes documentation only:

- `docs/issues/ISSUE-024-thin-gui-spec.md`
- `docs/plans/branch-inventory.md` as the source inventory
- `docs/plans/issue-disposition.md` as the legacy disposition source

Inspected salvage sources include the experimental `rag_gui.py`,
`GUI_direct_search.py`, `GUI_chromadb.py`, `GUI_workers.py`, `streamlit_app.py`
and PR #1. They are not edited or merged by this issue.

The future implementation issue must consume #21/#22/#23/#27 interfaces rather
than define replacements in UI code.

## Concrete actions

1. Link each retained UX concept to a stable service record or operation.
2. List prohibited direct dependencies and lifecycle behaviours.
3. Record PR #1 as superseded after its isolated ideas and SHA are preserved by
   #18.
4. Define loading, empty, error, cancellation, offline and abstention states.
5. Define safe path-opening and export boundaries inherited from closed #4 and
   the open #9 import-safety contract.
6. Add the post-#25 issue template/handoff requirement, including completed #22
   and #25 as prerequisites.

## Verification

```powershell
rg -n "specification only|separate future issue|No existing GUI branch" docs/issues/ISSUE-024-thin-gui-spec.md
rg -n "direct database|subprocess|search-only|cancellation|provider" docs/issues/ISSUE-024-thin-gui-spec.md
rg -n "PR #1" docs/plans/branch-inventory.md docs/issues/ISSUE-024-thin-gui-spec.md
git diff --check
```

## Measurable closure gate

- Every useful concept listed in GitHub #24 appears in the salvage matrix.
- Every prohibited architecture listed in GitHub #24 is explicitly forbidden.
- Search, evidence, status/settings and error/cancellation states have defined
  service inputs and outputs.
- PR #1 has a recorded SHA/disposition and is closed or retargeted only after
  its useful isolated concepts are documented.
- No GUI source, dependency or toolkit is added by #24.
- The post-#25 transition text assigns issue creation to #25 and names completed
  #22/#25 as hard prerequisites plus #21/#23/#27 as consumed contracts.

## Rollback and safety constraints

- Documentation can be reverted without changing runtime behaviour.
- Do not merge, rebase or delete any experimental branch under this issue.
- Do not add direct Chroma/PostgreSQL/provider access to a UI prototype.
- No local source may be opened without an explicit action and an allowlisted,
  resolved root.

## Commit boundary

One documentation-only commit:

```text
docs(#24): specify future thin GUI boundary
```

Any static mock, toolkit dependency or executable UI belongs to the separate
future implementation issue.
