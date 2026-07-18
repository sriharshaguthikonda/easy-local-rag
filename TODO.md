# Immediate work

The authoritative status, dependencies, closure gates, and complete execution
order are in the [canonical issue-plan index](docs/issues/README.md).

1. Pass #2's containment gate: rotate/revoke credentials, remove active-source
   use, and record the separately approved history-remediation decision.
2. Complete #18's branch, PR, local-ref, salvage, and secret-scan inventory.
3. Complete early #26A/#26B hygiene and the sole approved rewrite, then close
   #2 from the clean post-rewrite all-ref evidence.
4. Only then implement #19's read-only Chroma audit and deterministic export.
5. Implement the offline, database-neutral #20A ingestion package.
6. Hand the verified #20A contract to `.memory` #4; do not start #20B before
   that destination contract is stable.

Do not start cutover, GUI implementation, branch deletion, Chroma deletion, or
late repository cleanup from this TODO.
