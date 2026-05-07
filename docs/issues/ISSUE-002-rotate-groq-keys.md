# Issue #2 Plan: Rotate and remove hardcoded Groq API keys

GitHub: https://github.com/sriharshaguthikonda/easy-local-rag/issues/2

Priority: P0 security

## Goal

Remove every committed Groq secret from active source code, make the runtime read
`GROQ_API_KEY` from the environment, and add checks that stop this mistake from
coming back.

Do not paste, print, commit, or copy any real key into notes, issues, tests, or
commit messages.

## Files to inspect

- `groq_lama_MIlvs_RAG_ETTS.py`
- `Ollama_MIlvus_RAG_chat_TTS.py`
- `.gitignore`
- `.env.example` if it exists after issue #16
- `requirements.txt`
- optional new files: `.gitleaks.toml`, `.pre-commit-config.yaml`

## Implementation steps

1. Stop and ask the repo owner to rotate the exposed Groq keys in the Groq
   dashboard before considering this fixed. Code cleanup alone is not enough.
2. Find active hardcoded key call sites without printing secrets:

   ```powershell
   Get-ChildItem -Recurse -File -Include '*.py' |
     Select-String -Pattern 'gsk_' |
     ForEach-Object { "$($_.Path):$($_.LineNumber):REDACTED" }
   ```

3. In `groq_lama_MIlvs_RAG_ETTS.py`, replace `Groq(api_key="...")` with:

   - `from dotenv import load_dotenv` if missing
   - `load_dotenv()` before reading env vars
   - `api_key = os.getenv("GROQ_API_KEY")`
   - a clear `RuntimeError` or user-facing message if the value is missing

4. Apply the same change in `Ollama_MIlvus_RAG_chat_TTS.py`.
5. Do not log the API key. Error text may say `GROQ_API_KEY is missing`; it must
   never include the value.
6. Add or update `.env.example` with only this placeholder:

   ```dotenv
   GROQ_API_KEY=
   ```

7. Keep `.env` ignored in `.gitignore`.
8. Add a secret scanner gate. Preferred minimal path:

   - add `.gitleaks.toml`
   - add `.pre-commit-config.yaml` using gitleaks if pre-commit is already in
     use or introduced in issue #16
   - document the manual command in `README.md` or `AGENTS.md`

9. History cleanup must be coordinated separately because it rewrites shared git
   history. Use `git filter-repo --replace-text` only after all collaborators
   agree and after keys are rotated.

## Tests and verification

Run these before committing:

```powershell
python -m py_compile groq_lama_MIlvs_RAG_ETTS.py Ollama_MIlvus_RAG_chat_TTS.py
Get-ChildItem -Recurse -File -Include '*.py','*.md','.env.example','requirements*.txt' |
  Select-String -Pattern 'gsk_' |
  ForEach-Object { "$($_.Path):$($_.LineNumber):REDACTED" }
```

Expected result: the second command prints nothing.

If gitleaks is installed:

```powershell
gitleaks detect --redact --no-git
```

Expected result: no leaks in the working tree.

## Acceptance checklist

- [ ] Both exposed Groq keys are rotated outside git.
- [ ] No active source file contains a `gsk_` value.
- [ ] Both affected scripts read `GROQ_API_KEY` from env.
- [ ] Missing env var fails clearly without exposing secrets.
- [ ] `.env.example` contains placeholders only.
- [ ] Secret scan command is documented and passes.
- [ ] Any history rewrite plan is explicit and coordinated.

## Commit boundary

Use one commit for this issue only:

```text
fix(#2): remove hardcoded Groq keys
```
