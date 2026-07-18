# Issue #2: Rotate and remove committed Groq credentials

**Status:** OPEN — immediate security blocker; active-source cleanup is incomplete until credentials are rotated and repository history is handled.
**GitHub:** https://github.com/sriharshaguthikonda/easy-local-rag/issues/2
**Labels / priority:** `priority:P0`, `type:security`
**Dependencies:** Owner access to the Groq console; coordinated approval before any shared-history rewrite.

## Implementation slices

1. **Contain and rotate.** Rotate the affected credentials in the provider console; revoke old credentials. Never put credential-shaped text in code, documentation, test fixtures, console output, or commits.
2. **Remove active-source use.** Replace hardcoded client construction in `groq_lama_MIlvs_RAG_ETTS.py` and `Ollama_MIlvus_RAG_chat_TTS.py` with `GROQ_API_KEY` from the environment; fail clearly when it is absent without echoing its value.
3. **Prevent recurrence and remediate history.** Keep `.env` ignored, add a non-secret example-only `.env.example`, add a secret-scanner gate, then perform a separately approved `git filter-repo` history cleanup and force-push coordination.

## Affected interfaces, files, and artifacts

- Environment contract: `GROQ_API_KEY` only.
- Runtime clients in `groq_lama_MIlvs_RAG_ETTS.py` and `Ollama_MIlvus_RAG_chat_TTS.py`.
- `.env.example`, `.gitignore`, secret-scanner configuration, and rewritten git history (operational artifact).

## Concrete actions

- Search tracked text without displaying values; replace only the two known active call sites and audit all callers of Groq construction.
- Add a missing-key error path before API use; do not log the environment value.
- Run a redacted working-tree scan and a scanner such as gitleaks. Coordinate history rewrite only after rotation.

## Verification

```powershell
python -m py_compile groq_lama_MIlvs_RAG_ETTS.py Ollama_MIlvus_RAG_chat_TTS.py
Get-ChildItem -Recurse -File -Include '*.py','*.md','.env.example','requirements*.txt' | Select-String -Pattern 'gsk_' | ForEach-Object { "$($_.Path):$($_.LineNumber):REDACTED" }
gitleaks detect --redact --no-git
```

## Closure gate, rollback, and commit boundary

- **Close only when:** old credentials are revoked, no credential-shaped material remains in active source or scanned history, both clients use `GROQ_API_KEY`, the missing-key path is safe, and the scanner passes.
- **Rollback constraint:** never restore an exposed value; revert only code/config while retaining revoked credentials and scanner protections. History rewrite requires collaborator coordination and recovery guidance.
- **Commit:** `fix(#2): remove hardcoded Groq keys` (history rewrite is a separate coordinated operation).
