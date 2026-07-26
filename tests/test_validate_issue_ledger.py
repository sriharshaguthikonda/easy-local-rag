import importlib.util
import json
import subprocess
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "validate_issue_ledger.py"
SHA_MAIN = "a" * 40
SHA_MERGE = "b" * 40


def ledger_module():
    spec = importlib.util.spec_from_file_location("validate_issue_ledger", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def roadmap(active="#37"):
    state_37 = "active" if active == "#37" else "queued"
    active_row = active if active != "none" else "none"
    return "\n".join(
        (
            "## Current JIT state",
            "| Issue | State | Packet / next gate |",
            "|---|---|---|",
            f"| #37 | {state_37} | Ledger synchronization |",
            "| #38 | queued | Streamlit evidence contract |",
            f"| Active issue | {active_row} | declared sole active issue |",
        )
    )


def issue_15(status="CLOSED", extra=""):
    return f"# Issue #15: Atomically write vault JSON\n\n**Status:** {status} — completed.\n{extra}\n"


def inventory(branch_sha=SHA_MAIN, pr36_sha=SHA_MERGE, pr1_state="CLOSED, unmerged"):
    return "\n".join(
        (
            "## `main`",
            f"- Execution gate tip: `{branch_sha}`",
            "",
            "## PR #1 — GUI",
            f"- State at inspection: **{pr1_state}**.",
            "",
            "## PR #36 — Atomic vault write",
            "- State at inspection: **MERGED**.",
            f"- GitHub Create-a-merge-commit SHA: `{pr36_sha}`.",
        )
    )


def snapshot(*, issue_37="OPEN", issue_38="OPEN", issue_15="CLOSED", pr36_sha=SHA_MERGE, pr1_state="CLOSED", pr1_sha=None, branch_sha=SHA_MAIN):
    return {
        "issues": {
            "37": {"state": issue_37},
            "38": {"state": issue_38},
            "15": {"state": issue_15},
        },
        "prs": {
            "1": {"state": pr1_state, "merge_sha": pr1_sha},
            "36": {"state": "MERGED", "merge_sha": pr36_sha},
        },
        "branches": {"main": branch_sha},
    }


def validate(**changes):
    module = ledger_module()
    values = {
        "roadmap_text": roadmap(),
        "issue_texts": {"ISSUE-015-atomic-vault-write.md": issue_15()},
        "inventory_text": inventory(),
        "snapshot": snapshot(),
    }
    values.update(changes)
    return module.validate_ledger(**values)


def test_reconciled_activated_sole_active_ledger_passes():
    assert validate() == []


def test_reconciled_zero_active_ledger_passes():
    assert validate(roadmap_text=roadmap("none")) == []


def test_stale_tracked_issue_row_reports_expected_and_actual_state():
    errors = validate(snapshot=snapshot(issue_37="CLOSED"))
    assert "issue #37: expected OPEN, actual CLOSED" in errors


def test_tracked_issues_table_validates_priority_type_rows():
    tracked = "\n".join(
        (
            "## Tracked issues",
            "| Issue | Priority and type | Status | Canonical plan |",
            "|---|---|---|---|",
            "| #15 | P0 bug | **Closed — completed** | Atomic vault write |",
        )
    )
    errors = validate(roadmap_text=roadmap() + "\n" + tracked, issue_texts={}, snapshot=snapshot(issue_15="OPEN"))
    assert "issue #15: expected CLOSED, actual OPEN" in errors


def test_stale_canonical_issue_status_and_lifecycle_phrase_are_mismatches():
    errors = validate(
        issue_texts={"ISSUE-015-atomic-vault-write.md": issue_15("ACTIVE", "future activation merge")}
    )
    assert "ISSUE-015: expected ACTIVE, actual CLOSED" in errors
    assert "ISSUE-015: expected no obsolete activation wording, actual future activation merge" in errors


def test_stale_merged_pr_sha_is_a_mismatch():
    errors = validate(snapshot=snapshot(pr36_sha="c" * 40))
    assert f"PR #36 merge SHA: expected {SHA_MERGE}, actual {'c' * 40}" in errors


def test_closed_unmerged_pr_disposition_requires_no_merge_sha():
    errors = validate(snapshot=snapshot(pr1_sha="d" * 40))
    assert f"PR #1 merge SHA: expected none, actual {'d' * 40}" in errors


def test_stale_execution_gate_branch_tip_is_a_mismatch():
    errors = validate(snapshot=snapshot(branch_sha="e" * 40))
    assert f"branch main: expected {SHA_MAIN}, actual {'e' * 40}" in errors


def test_duplicate_active_issues_are_a_mismatch():
    duplicate = roadmap().replace("| #38 | queued |", "| #38 | active |")
    errors = validate(roadmap_text=duplicate)
    assert "active issues: expected at most one, actual #37, #38" in errors


def test_missing_37_or_38_is_a_mismatch():
    errors = validate(roadmap_text=roadmap().replace("| #38 | queued | Streamlit evidence contract |\n", ""))
    assert "roadmap #38: expected present, actual missing" in errors


def test_obsolete_activation_wording_in_roadmap_is_a_mismatch():
    errors = validate(roadmap_text=roadmap() + "\nfuture activation merge\n")
    assert "roadmap: expected no obsolete activation wording, actual future activation merge" in errors


def test_stale_phrase_after_an_allowed_explanation_is_still_a_mismatch():
    text = roadmap() + "\nknown stale lifecycle phrases include future activation merge\nfuture activation merge\n"
    errors = validate(roadmap_text=text)
    assert "roadmap: expected no obsolete activation wording, actual future activation merge" in errors


def test_malformed_required_canonical_data_is_a_mismatch():
    errors = validate(issue_texts={"ISSUE-015-atomic-vault-write.md": "# Issue #15\n"})
    assert "ISSUE-015: expected documented Status, actual malformed" in errors


def test_established_canonical_open_status_forms_validate_against_live_state():
    live = snapshot()
    live["issues"].update({"5": {"state": "OPEN"}, "17": {"state": "OPEN"}, "23": {"state": "OPEN"}})
    errors = validate(
        issue_texts={
            "ISSUE-005-prompt-injection.md": "**Status:** OPEN\n",
            "ISSUE-017-postgres-consolidation.md": "**Status:** Canonical epic; open.\n",
            "ISSUE-023-retrieval-parity.md": "Status: **Open — migration gate**\n",
        },
        snapshot=live,
    )
    assert errors == []


def test_bad_usage_returns_2(capsys):
    assert ledger_module().main(["--repo", "not-a-repository"]) == 2
    assert "not-a-repository" not in capsys.readouterr().out


def test_unknown_usage_argument_does_not_echo_credential_like_input(capsys):
    assert ledger_module().main(["--secret", "ghp_not_for_output"]) == 2
    output = capsys.readouterr()
    assert "ghp_not_for_output" not in output.out + output.err
    assert output.out == "usage: python scripts/validate_issue_ledger.py --repo OWNER/REPO\n"
    assert output.err == ""


def test_gh_and_api_failures_return_2(tmp_path):
    root = tmp_path
    (root / "docs" / "issues").mkdir(parents=True)
    (root / "docs" / "plans").mkdir()
    (root / "docs" / "issues" / "README.md").write_text(roadmap(), encoding="utf-8")
    (root / "docs" / "issues" / "ISSUE-015-atomic-vault-write.md").write_text(issue_15(), encoding="utf-8")
    (root / "docs" / "plans" / "branch-inventory.md").write_text(inventory(), encoding="utf-8")

    def broken_runner(*_args, **_kwargs):
        return subprocess.CompletedProcess([], 1, "", "authentication failed")

    assert ledger_module().main(["--repo", "owner/repo"], root=root, runner=broken_runner) == 2


def test_live_collector_uses_only_the_locked_gh_queries():
    calls = []
    responses = iter(
        (
            [{"number": 15, "state": "CLOSED"}, {"number": 37, "state": "OPEN"}, {"number": 38, "state": "OPEN"}],
            [
                {"number": 1, "state": "CLOSED", "mergeCommit": None},
                {"number": 36, "state": "MERGED", "mergeCommit": {"oid": SHA_MERGE}},
            ],
            {"object": {"sha": SHA_MAIN}},
        )
    )

    def runner(command, **_kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, json.dumps(next(responses)), "")

    actual = ledger_module().collect_snapshot("owner/repo", {"main": SHA_MAIN}, runner)
    assert actual == snapshot()
    assert calls == [
        ["gh", "issue", "list", "--repo", "owner/repo", "--state", "all", "--limit", "100", "--json", "number,state"],
        ["gh", "pr", "list", "--repo", "owner/repo", "--state", "all", "--limit", "100", "--json", "number,state,mergeCommit"],
        ["gh", "api", "repos/owner/repo/git/ref/heads/main"],
    ]


def test_zero_exit_malformed_gh_payloads_return_2_without_a_traceback(tmp_path, capsys):
    root = tmp_path
    (root / "docs" / "issues").mkdir(parents=True)
    (root / "docs" / "plans").mkdir()
    (root / "docs" / "issues" / "README.md").write_text(roadmap(), encoding="utf-8")
    (root / "docs" / "issues" / "ISSUE-015-atomic-vault-write.md").write_text(issue_15(), encoding="utf-8")
    (root / "docs" / "plans" / "branch-inventory.md").write_text(inventory(), encoding="utf-8")
    payloads = (
        ([{}], []),
        ([{"number": True, "state": "OPEN"}], [], {"object": {"sha": SHA_MAIN}}),
        ([{"number": 15, "state": "CLOSED"}], [{}]),
        (
            [{"number": 15, "state": "CLOSED"}],
            [{"number": True, "state": "OPEN", "mergeCommit": None}],
            {"object": {"sha": SHA_MAIN}},
        ),
        (
            [{"number": 15, "state": "CLOSED"}],
            [{"number": 1, "state": "CLOSED", "mergeCommit": None}],
            {"object": {}},
        ),
    )

    for responses in payloads:
        response_iter = iter(responses)

        def runner(command, **_kwargs):
            return subprocess.CompletedProcess(command, 0, json.dumps(next(response_iter)), "")

        assert ledger_module().main(["--repo", "owner/repo"], root=root, runner=runner) == 2
        assert "Traceback" not in capsys.readouterr().out
