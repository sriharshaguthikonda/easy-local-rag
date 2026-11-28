# Issue #13 maintained-Chroma request admission

**State** Planning pending a separate activation merge.

**Target** `GUI-BM25-hyb-kkro-tkn-lmt-synms-mon-chngs-streamlit-chromadb-docs` at `a8926023f8c4ef56ec632137200e6ff9904ca675`.

**Mode** One-time maintained ChromaDB exception.

**Owner decision, 28 November 2026** “I approve a separate, one-time maintained-Chroma implementation exception for #13 while #2 remains open.” The decision covers this packet, separate activation, tests-only RED commits, implementation and independent synthetic verification. The exclusion of #27 live smoke `Q-20261128-001` is explicit. [Issue #13](https://github.com/sriharshaguthikonda/easy-local-rag/issues/13) · [Canonical scope](../../issues/ISSUE-013-tiktoken-counter.md) · [Ledger](../../issues/README.md) · [Pinned audit](../../Research/issue-013-token-context-audit.md).

## Maintained paths

PyQt `GUI_workers.py`, Streamlit `streamlit_app.py`, both CLI/programmatic paths in `groq_lama_chromadb_RAG_ETTS.py` and `streamlit_groq_lama_chromadb_RAG_ETTS.py`, and their maintained query rewrites all reach `generation_models.py`. Keep preparation and validation in `rag_prompting.py`. Put counting and whole-turn selection in `token_budget.py`. Remove Streamlit's independent count and fixed ceiling. Keep search-only independent of generation. Historical vault, email and Milvus paths retain their existing status.

## Admission and termination contract

- Count the whole rendered text-only request for each candidate, including all role delimiters, generation framing, system instructions, history, the question and the serialized evidence. Account for any provider-added framing and request options which affect input. A counter is exact only for a verified tokenizer and complete matching template. A conservative count needs a documented bound on the actual rendered input. Return unknown for missing assets, model or template mismatches and unbounded request fields, then refuse dispatch. Never download assets in the request path. Remove model guessing, arbitrary encoding substitutions and word-count fallbacks.
- Bind assets to provider, model identity and version or digest as available and record tokenizer/template provenance and count quality. A configured number is an operator ceiling. Hosted admission needs verified capacity for that model route. Local admission needs verified model and effective server settings, with an explicit matching request limit or independently verified allocation. Account for a separate output cap and reserve it before dispatch. Organization TPM and retry policy remain distinct from model context.
- Use one admission path before dispatch. Only drop complete oldest user/assistant pairs. Freeze the final messages before the first request. If Ollama discovery proves the saved preferred model absent, freeze the original prepared messages before selecting a replacement. An undispatchable candidate never grants permission to trim a retry. Never slice an instruction, current query, or evidence block and never replan or trim on fallback. Recheck the exact messages, output reserve and evidence for each permitted same-provider replacement. Preserve exact canonical snapshot identities, alias order and both SHA-256 hashes. Fail closed before dispatch if protected content will not fit.
- Use the verified reject-on-overflow mode where available. Reject reported truncation, capacity errors, incomplete output and comparable prompt usage proving input loss. Keep the #27 buffer, terminal and structural validator ahead of publication. Metadata may state what the client prepared and what the provider reported. Normal stops, usage counters and client hashes do not attest provider-side consumption. No raw failed output reaches history, export, display or speech.
- Roll back to evidence/search-only when an asset or adapter cannot establish safe admission. Retain ChromaDB, all 494 baseline regressions, 11 native synthetic Chroma cases and the existing #5/#10/#11/#27 contracts. Keep the consolidation order, #2, full #27 and migration gates. Preserve the real corpus, recovery modifications, credentials and the resident LM Studio configuration.

## Review, activation and tests

1. Record the owner decision, review this packet through a bounded Foreman single-slot brief and lock it by merging the documentation PR to `main`.
2. Merge a separate ledger activation to `main` referencing the locked packet merge and frozen GUI base. Only then commit tests-only REDs on the GUI lineage. Reproduce model/blind-count mismatches and protected request/capacity failures against the frozen product tree. Separate RED cases for publication and truncation should pass through real adapters with fake transports and the real structural validator. Record observed assertion failures before changing product code.
3. Verify effective-limit and reserve boundaries, unbounded metadata, large fitting input, exact tokenizer provenance, CJK, mixed Unicode, code and biomedical terminology. Cover missing and wrong assets/templates, complete oldest-turn removal, same messages/hashes on fallback and exclusion of smaller candidates. Simulate reported truncation and a normal-looking incomplete provider stream. Verify evidence-only behavior on failures and unchanged search-only behavior.
4. Run focused, full and native suites under the TCP/DNS-denied Python launcher pinned to the #27 audit. Use Python 3.12, ChromaDB 1.0.16, synthetic collections, offscreen Qt and no provider credentials. Give independent Ornith reviewers bounded, line-numbered packets from the exact candidate via the existing single-slot Foreman harness. Run independent exact-head verification from a separate checkout, and compare tree identity when merging. Publish only scoped changes and record RED SHAs, exact reviewed head, evidence and known provider limits in the ledger.

```powershell
py -3.12 "$audit/run_pytest.py" -q -p no:cacheprovider tests/test_request_admission.py
py -3.12 "$audit/run_pytest.py" -q -p no:cacheprovider tests/test_structural_answers.py tests/test_generation_models.py tests/test_generation_streamlit.py tests/test_generation_qt.py tests/test_generation_cli.py
py -3.12 "$audit/run_pytest.py" -q -p no:cacheprovider tests/chroma_hybrid_native_cases.py
py -3.12 "$audit/run_pytest.py" -q -p no:cacheprovider tests
git diff --check
```

## Packet review

A bounded Foreman review through the resident Ornith completed with exit 0. It checked freeze-on-first-attempt, per-candidate admission and preservation of output reserve. The review check exposed the need to state when to freeze a request if discovery proves the preferred local model absent. That rule is explicit above.
