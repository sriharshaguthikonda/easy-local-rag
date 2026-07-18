# Issue #2: Rotate and remove committed Groq credentials

**Status:** OPEN — immediate containment blocker. Rotation/revocation and active-source cleanup unlock #18; the issue remains open until early #26B supplies one of the two explicit post-rewrite terminal outcomes in the final closure gate.
**GitHub:** https://github.com/sriharshaguthikonda/easy-local-rag/issues/2
**Labels / priority:** `priority:P0`, `type:security`
**Dependencies:** Owner access to the Groq console; #18's immutable ref inventory; early #26B as the sole executor of any separately approved shared-history rewrite.

## Implementation slices

1. **Contain and rotate.** Rotate the affected credentials in the provider console; revoke old credentials. Never put credential-shaped text in code, documentation, test fixtures, console output, or commits.
2. **Remove active-source use.** Replace hardcoded client construction in `groq_lama_MIlvs_RAG_ETTS.py` and `Ollama_MIlvus_RAG_chat_TTS.py` with `GROQ_API_KEY` from the environment; fail clearly when it is absent without echoing its value.
3. **Prevent recurrence and hand off history remediation.** Keep `.env` ignored, add a non-secret example-only `.env.example`, and add a secret-scanner gate. Record rewrite approval and affected ref/finding types without values; #26B alone performs the coordinated rewrite after #18.

## Affected interfaces, files, and artifacts

- Environment contract: `GROQ_API_KEY` only.
- Runtime clients in `groq_lama_MIlvs_RAG_ETTS.py` and `Ollama_MIlvus_RAG_chat_TTS.py`.
- `.env.example`, `.gitignore`, and `.gitleaks.toml`.
- Private rotation receipt and affected-ref/finding-type inventory without values.
- #26B's post-rewrite all-ref manifest, redacted report, scanner version, config hash, and exit status are consumed as closure evidence; #2 does not create rewritten refs.

## Concrete actions

- Search tracked text without displaying values; replace only the two known active call sites and audit all callers of Groq construction.
- Add a missing-key error path before API use; do not log the environment value.
- Run the active-tree scan below. After rotation, record explicit rewrite approval and hand the #18 inventory to #26B.
- Accept #26B's post-rewrite evidence only when its pinned command covers local
  heads, remote heads, tags, and fetched PR refs and reaches either the clean
  exit-`0` outcome or the narrowly defined, user-approved immutable-residual
  outcome in the final closure gate.

## Verification

```powershell
python -m py_compile groq_lama_MIlvs_RAG_ETTS.py Ollama_MIlvus_RAG_chat_TTS.py
Get-ChildItem -Recurse -File -Include '*.py','*.md','.env.example','requirements*.txt' | Select-String -Pattern 'gsk_' | ForEach-Object { "$($_.Path):$($_.LineNumber):REDACTED" }
$evidence = $env:EASY_RAG_SECURITY_EVIDENCE_DIR
if (-not $evidence) { throw 'Set EASY_RAG_SECURITY_EVIDENCE_DIR to an access-controlled directory outside the repository.' }
gitleaks dir --redact --config .gitleaks.toml --report-format json `
  --report-path (Join-Path $evidence 'active-tree-gitleaks.json') --exit-code 1 .
if ($LASTEXITCODE -ne 0) { throw 'Active-tree scan found a secret or failed.' }
```

The active-tree scan is not history evidence. Final closure consumes the exact
post-rewrite command and private evidence contract defined by #26B; no baseline
or allowlist may suppress the known revoked credential type.

## Closure gate, rollback, and commit boundary

- **Containment gate (unlocks #18):** old credentials are revoked; both clients use `GROQ_API_KEY`; and the missing-key path plus active-tree scan pass. Passing this gate does not close #2.
- **Final closure gate (after early #26B, before #19):** rewrite approval is
  recorded and #26B supplies one of two terminal outcomes: **clean** — GitHub
  Support remediation is confirmed when needed and the fresh-clone all-ref scan
  exits `0` with zero findings; or **accepted immutable residual** — Support
  declines specifically because revocation/rotation mitigated the risk, every
  remaining finding is reachable only from the recorded read-only PR/cached
  refs, writable refs scan clean, and the user explicitly accepts the dated,
  hashed residual-risk record. Any other Support disposition or missing
  acceptance keeps #2 and migration blocked.
- **Rollback constraint:** never restore an exposed value; revert only code/config while retaining revoked credentials and scanner protections. History rewrite requires collaborator coordination and recovery guidance.
- **Commit:** `fix(#2): remove hardcoded Groq keys`. Rotation records are external operational evidence; history rewrite/ref replacement belongs only to #26B.
