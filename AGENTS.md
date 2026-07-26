# Ledger validation

The root orchestrator updates the canonical ledger and runs
`python scripts/validate_issue_ledger.py --repo OWNER/REPO` immediately after
an activation, implementation merge, issue closure, branch archive, or PR
disposition change, and before the next activation or closure. This command is
mandatory before every activation and closure.
