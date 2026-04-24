#!/usr/bin/env python3
"""Check sync status between nixi fork and upstream Hermes repository.

Compares origin/main and nixi branch against upstream/main.
Reports commits behind/ahead and warns about conflict-prone file changes.

Exit codes:
    0 — up-to-date (no upstream changes)
    1 — behind upstream (upstream has new commits)
    2 — conflict-risk (conflict-prone files changed in upstream)
"""

import subprocess
import sys

# Files that nixi modifies — changes here require careful merge
CONFLICT_PRONE_FILES = [
    "gateway/config.py",
    "gateway/run.py",
    "gateway/platforms/slack.py",
    "toolsets.py",
    "agent/prompt_builder.py",
    "cron/scheduler.py",
]


def run_git(*args: str) -> str:
    """Run a git command and return stdout. Exit on failure."""
    result = subprocess.run(
        ["git", *args],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"git {' '.join(args)} failed: {result.stderr.strip()}", file=sys.stderr)
        sys.exit(1)
    return result.stdout.strip()


def count_commits(ref_a: str, ref_b: str) -> int:
    """Count commits in ref_a that are not in ref_b."""
    output = run_git("rev-list", "--count", f"{ref_a}...{ref_b}", "--left-only")
    return int(output) if output else 0


def changed_files(ref_a: str, ref_b: str) -> list[str]:
    """List files changed between ref_a and ref_b."""
    output = run_git("diff", "--name-only", f"{ref_a}...{ref_b}")
    return output.splitlines() if output else []


def main() -> None:
    # Step 1: Fetch upstream
    print("Fetching upstream...")
    run_git("fetch", "upstream")

    # Step 2: Compare upstream/main vs origin/main
    behind = count_commits("upstream/main", "origin/main")
    ahead = count_commits("origin/main", "upstream/main")

    print(f"\norigin/main vs upstream/main:")
    print(f"  Behind: {behind} commit(s)")
    print(f"  Ahead:  {ahead} commit(s)")

    # Step 3: Compare upstream/main vs nixi branch
    nixi_behind = count_commits("upstream/main", "nixi")
    nixi_ahead = count_commits("nixi", "upstream/main")

    print(f"\nnixi vs upstream/main:")
    print(f"  Behind: {nixi_behind} commit(s)")
    print(f"  Ahead:  {nixi_ahead} commit(s)")

    # Step 4: List files changed in upstream since last sync
    if behind > 0 or nixi_behind > 0:
        upstream_changed = changed_files("origin/main", "upstream/main")
        if upstream_changed:
            print(f"\nFiles changed in upstream since last sync:")
            for f in upstream_changed:
                print(f"  - {f}")
    else:
        upstream_changed = []

    # Step 5: Check conflict-prone files
    conflict_files = [f for f in upstream_changed if f in CONFLICT_PRONE_FILES]

    if conflict_files:
        print(f"\nWARNING: Conflict-prone files changed in upstream:")
        for f in conflict_files:
            print(f"  - {f}")

    # Step 6: Exit code
    if conflict_files:
        print("\nExit code: 2 (conflict-risk)")
        sys.exit(2)
    elif behind > 0 or nixi_behind > 0:
        print("\nExit code: 1 (behind upstream)")
        sys.exit(1)
    else:
        print("\nExit code: 0 (up-to-date)")
        sys.exit(0)


if __name__ == "__main__":
    main()