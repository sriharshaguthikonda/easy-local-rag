# Immediate work

The authoritative status, dependencies, closure gates, and complete execution
order are in the [canonical issue-plan index](docs/issues/README.md).

1. Complete #2 credential rotation, revocation, active-source cleanup, and the
   separately approved history-remediation decision.
2. Complete #18's branch, PR, local-ref, salvage, and secret-scan inventory.
3. Perform only the early hygiene slice of #26.
4. Implement #19's read-only Chroma audit and deterministic export.
5. Implement the offline, database-neutral #20A ingestion package.
6. Hand the verified #20A contract to `.memory` #4; do not start #20B before
   that destination contract is stable.

Do not start cutover, GUI implementation, branch deletion, Chroma deletion, or
late repository cleanup from this TODO.
