"""Validate the canonical issue ledger against GitHub's public state."""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


SHA = r"[0-9a-f]{40}"
OBSOLETE = "future activation merge"


class LedgerCliError(Exception):
    pass


class SafeArgumentParser(argparse.ArgumentParser):
    def error(self, _message):
        raise ValueError


def _state(value):
    return "CLOSED" if value.lower() == "closed" else "OPEN"


def _obsolete(text):
    for match in re.finditer(re.escape(OBSOLETE), text, re.IGNORECASE):
        line_start = text.rfind("\n", 0, match.start()) + 1
        line = text[line_start:text.find("\n", match.start()) if "\n" in text[match.start():] else len(text)]
        if not re.search(r"stale lifecycle phrases\s+(?:including|include)\s+`?future activation merge`?", line, re.IGNORECASE):
            return True
    return False


def _roadmap_rows(text):
    rows = {}
    for line in text.splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if not cells or not re.fullmatch(r"#\d+", cells[0]):
            continue
        number = cells[0][1:]
        if len(cells) > 1 and cells[1] in {"queued", "planning", "active", "review", "blocked-awaiting-user", "closed"}:
            rows[number] = cells[1]
        elif len(cells) > 2:
            status = re.match(r"\*{0,2}(Open|Closed)\b", cells[2], re.IGNORECASE)
            if status:
                rows.setdefault(number, status.group(1).lower())
    return rows


def _canonical_status(name, text):
    number = re.fullmatch(r"ISSUE-(\d{3})-[^.]+\.md", name)
    if not number:
        return None, None
    status = re.search(r"^\*\*Status:\*\*\s*([A-Za-z-]+)", text, re.MULTILINE)
    return number.group(1).lstrip("0") or "0", status.group(1).upper() if status else None


def _inventory(inventory_text):
    branches = {}
    for block in re.split(r"(?=^## )", inventory_text, flags=re.MULTILINE):
        heading = re.match(r"## `([^`]+)`\s*$", block, re.MULTILINE)
        gate = re.search(r"^[-*]\s+Execution gate tip:\s*`(" + SHA + r")`", block, re.MULTILINE)
        if heading and gate:
            branches[heading.group(1)] = gate.group(1)

    prs = {}
    for block in re.split(r"(?=^## PR #)", inventory_text, flags=re.MULTILINE):
        heading = re.match(r"## PR #(\d+)\b", block)
        if not heading:
            continue
        state = re.search(r"State at inspection:\s*\*\*([A-Z]+)(?:,\s*(unmerged))?\*\*", block)
        if not state:
            prs[heading.group(1)] = None
            continue
        expected = {"state": state.group(1), "merge_sha": None}
        if expected["state"] == "MERGED":
            sha = re.search(r"merge-commit SHA:\s*`?\s*(" + SHA + r")", block, re.IGNORECASE)
            expected["merge_sha"] = sha.group(1) if sha else "MALFORMED"
        elif state.group(2) != "unmerged":
            expected["merge_sha"] = "MALFORMED"
        prs[heading.group(1)] = expected
    return branches, prs


def validate_ledger(*, roadmap_text, issue_texts, inventory_text, snapshot):
    errors = []
    rows = _roadmap_rows(roadmap_text)
    for number in ("37", "38"):
        if number not in rows:
            errors.append(f"roadmap #{number}: expected present, actual missing")

    active = sorted(f"#{number}" for number, state in rows.items() if state == "active")
    if len(active) > 1:
        errors.append(f"active issues: expected at most one, actual {', '.join(active)}")
    declared = re.search(r"^\|\s*Active issue\s*\|\s*(#\d+|none)\s*\|", roadmap_text, re.MULTILINE)
    actual_active = active[0] if len(active) == 1 else "none"
    if not declared or declared.group(1) != actual_active:
        errors.append(f"declared active issue: expected {actual_active}, actual {declared.group(1) if declared else 'missing'}")
    if _obsolete(roadmap_text):
        errors.append(f"roadmap: expected no obsolete activation wording, actual {OBSOLETE}")

    issues = snapshot.get("issues", {})
    for number, lifecycle in rows.items():
        if number not in issues:
            errors.append(f"issue #{number}: expected {_state(lifecycle)}, actual missing")
        elif issues[number].get("state") != _state(lifecycle):
            errors.append(f"issue #{number}: expected {_state(lifecycle)}, actual {issues[number].get('state', 'missing')}")

    for name, text in issue_texts.items():
        number, status = _canonical_status(name, text)
        if number is None:
            continue
        label = f"ISSUE-{int(number):03d}"
        if status is None:
            errors.append(f"{label}: expected documented Status, actual malformed")
            continue
        if status not in {"ACTIVE", "PLANNING", "QUEUED", "REVIEW", "BLOCKED-AWAITING-USER", "CLOSED"}:
            errors.append(f"{label}: expected documented Status, actual {status}")
        elif number not in issues:
            errors.append(f"issue #{number}: expected {_state(status)}, actual missing")
        elif issues[number].get("state") != _state(status):
            errors.append(f"{label}: expected {status}, actual {issues[number].get('state', 'missing')}")
        if _obsolete(text):
            errors.append(f"{label}: expected no obsolete activation wording, actual {OBSOLETE}")

    branches, prs = _inventory(inventory_text)
    for branch, expected in branches.items():
        actual = snapshot.get("branches", {}).get(branch, "missing")
        if actual != expected:
            errors.append(f"branch {branch}: expected {expected}, actual {actual}")
    for number, expected in prs.items():
        if expected is None or expected["merge_sha"] == "MALFORMED":
            errors.append(f"PR #{number}: expected documented disposition, actual malformed")
            continue
        actual = snapshot.get("prs", {}).get(number)
        if actual is None or actual.get("state") != expected["state"]:
            errors.append(f"PR #{number}: expected {expected['state']}, actual {actual.get('state', 'missing') if actual else 'missing'}")
            continue
        actual_sha = actual.get("merge_sha") or "none"
        expected_sha = expected["merge_sha"] or "none"
        if actual_sha != expected_sha:
            errors.append(f"PR #{number} merge SHA: expected {expected_sha}, actual {actual_sha}")
    return errors


def _run(command, runner):
    try:
        result = runner(command, capture_output=True, text=True)
    except (OSError, subprocess.SubprocessError) as exc:
        raise LedgerCliError("gh command failed") from exc
    if result.returncode:
        raise LedgerCliError("gh command failed")
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise LedgerCliError("gh returned invalid JSON") from exc


def collect_snapshot(repo, branches, runner=subprocess.run):
    issues = _run(["gh", "issue", "list", "--repo", repo, "--state", "all", "--limit", "100", "--json", "number,state"], runner)
    prs = _run(["gh", "pr", "list", "--repo", repo, "--state", "all", "--limit", "100", "--json", "number,state,mergeCommit"], runner)
    if not isinstance(issues, list) or not isinstance(prs, list):
        raise LedgerCliError("gh returned invalid JSON")
    try:
        normalized_issues = {
            str(item["number"]): {"state": item["state"]}
            for item in issues
            if isinstance(item["number"], int) and item["number"] > 0 and item["state"] in {"OPEN", "CLOSED"}
        }
        normalized_prs = {
            str(item["number"]): {
                "state": item["state"],
                "merge_sha": (item.get("mergeCommit") or {}).get("oid"),
            }
            for item in prs
            if isinstance(item["number"], int)
            and item["number"] > 0
            and item["state"] in {"OPEN", "CLOSED", "MERGED"}
            and (item.get("mergeCommit") is None or isinstance(item.get("mergeCommit"), dict))
            and (item.get("mergeCommit") is None or isinstance(item["mergeCommit"].get("oid"), str))
        }
    except (KeyError, TypeError):
        raise LedgerCliError("gh returned invalid JSON") from None
    if len(normalized_issues) != len(issues) or len(normalized_prs) != len(prs):
        raise LedgerCliError("gh returned invalid JSON")
    snapshot = {
        "issues": normalized_issues,
        "prs": normalized_prs,
        "branches": {},
    }
    for branch in branches:
        ref = _run(["gh", "api", f"repos/{repo}/git/ref/heads/{branch}"], runner)
        try:
            sha = ref["object"]["sha"]
            if not isinstance(sha, str) or not re.fullmatch(SHA, sha):
                raise ValueError
            snapshot["branches"][branch] = sha
        except (KeyError, TypeError, ValueError):
            raise LedgerCliError("gh returned invalid ref JSON") from None
    return snapshot


def main(argv=None, *, root=None, runner=subprocess.run):
    parser = SafeArgumentParser(add_help=False)
    parser.add_argument("--repo", required=True)
    try:
        args = parser.parse_args(argv)
    except (SystemExit, ValueError):
        print("usage: python scripts/validate_issue_ledger.py --repo OWNER/REPO")
        return 2
    if not re.fullmatch(r"[^/\s]+/[^/\s]+", args.repo):
        print("usage: python scripts/validate_issue_ledger.py --repo OWNER/REPO")
        return 2
    root = Path(root or Path(__file__).parents[1])
    try:
        issue_dir = root / "docs" / "issues"
        roadmap_text = (issue_dir / "README.md").read_text(encoding="utf-8")
        inventory_text = (root / "docs" / "plans" / "branch-inventory.md").read_text(encoding="utf-8")
        issue_texts = {path.name: path.read_text(encoding="utf-8") for path in issue_dir.glob("ISSUE-*.md")}
        branches, _ = _inventory(inventory_text)
        errors = validate_ledger(
            roadmap_text=roadmap_text,
            issue_texts=issue_texts,
            inventory_text=inventory_text,
            snapshot=collect_snapshot(args.repo, branches, runner),
        )
    except (OSError, UnicodeError, LedgerCliError):
        print("ledger validator: expected readable documents and gh JSON, actual unavailable")
        return 2
    for error in errors:
        print(error)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
