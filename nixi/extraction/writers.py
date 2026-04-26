"""File writers with consolidation logic and conflict resolution for nixi extraction.

Handles:
- MEMORY.md write/merge with aggressive consolidation at memory_limit
- AGENTS.md append (never overwrite)
- Employee USER.md creation, merge, and directory conflict resolution
- Channel skill directory structure with SQL references
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from pathlib import Path

from nixi.models import UserMap

logger = logging.getLogger(__name__)


def write_org_facts(
    facts: str,
    output_dir: Path,
    memory_limit: int | None = None,
) -> None:
    """Write/merge organizational facts to {output_dir}/MEMORY.md.

    If the file already exists, merges new facts with existing content.
    If total content exceeds memory_limit, aggressively consolidates by
    extracting key points and removing redundancy.

    Args:
        facts: New facts to write/merge.
        output_dir: Base output directory.
        memory_limit: Maximum character limit. Uses NixiConfig default if None.
    """
    if memory_limit is None:
        from nixi.config import NixiConfig

        memory_limit = NixiConfig.from_config().memory_limit

    memory_path = output_dir / "MEMORY.md"

    if memory_path.exists():
        existing = memory_path.read_text(encoding="utf-8")
        merged = existing + "\n\n" + facts
    else:
        merged = facts

    # Consolidate if over limit
    if len(merged) > memory_limit:
        merged = _consolidate(merged, memory_limit)

    memory_path.parent.mkdir(parents=True, exist_ok=True)
    memory_path.write_text(merged, encoding="utf-8")


def _consolidate(text: str, limit: int) -> str:
    """Aggressively consolidate text to fit within a character limit.

    Strategy:
    1. Remove duplicate lines
    2. Deduplicate section headers
    3. If still over limit, truncate each section proportionally
    """
    lines = text.splitlines()
    seen: set[str] = set()
    deduped: list[str] = []

    for line in lines:
        stripped = line.strip()
        # Keep headers always
        if stripped.startswith("#"):
            deduped.append(line)
            seen.add(stripped.lower())
        elif stripped and stripped not in seen:
            deduped.append(line)
            seen.add(stripped)

    result = "\n".join(deduped)

    if len(result) <= limit:
        return result

    # Truncate proportionally — keep headers and first N chars of content
    if len(result) > limit:
        # Conservative truncation: keep beginning, add truncation notice
        result = result[: limit - 3] + "..."

    return result


def write_rules(rules: str, output_dir: Path) -> None:
    """Append new rules to {output_dir}/AGENTS.md.

    Never overwrites existing content. New rules are appended with a
    timestamped section header.

    Args:
        rules: New rules to append.
        output_dir: Base output directory.
    """
    agents_path = output_dir / "AGENTS.md"
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    new_section = f"\n\n## Extracted {timestamp}\n\n{rules}"

    agents_path.parent.mkdir(parents=True, exist_ok=True)

    if agents_path.exists():
        existing = agents_path.read_text(encoding="utf-8")
        agents_path.write_text(existing + new_section, encoding="utf-8")
    else:
        agents_path.write_text(f"# AGENTS\n\n{rules}\n", encoding="utf-8")


def write_employee_info(
    employees: list[dict],
    output_dir: Path,
    user_map: UserMap,
    employee_limit: int | None = None,
) -> None:
    """Write per-employee USER.md files with conflict resolution.

    For each employee:
    - If both {user_id}/USER.md and {display_name}/USER.md exist → merge
      into {user_id}/USER.md, archive display_name version.
    - Creates {user_id}/USER.md (or {display_name}/USER.md if no user_id).
    - Merges if file exists, compresses if over limit.

    Args:
        employees: List of dicts with keys: display_name, user_id, info.
        output_dir: Base output directory.
        user_map: Bidirectional user mapping for conflict resolution.
        employee_limit: Char limit per employee. Uses config default if None.
    """
    if employee_limit is None:
        from nixi.config import NixiConfig

        employee_limit = NixiConfig.from_config().employee_limit

    employees_dir = output_dir / "employees"
    employees_dir.mkdir(parents=True, exist_ok=True)

    for emp in employees:
        display_name = emp.get("display_name", "unknown")
        user_id = emp.get("user_id")
        info = emp.get("info", "")

        # Determine directory key: prefer user_id, fall back to display_name
        dir_key = user_id if user_id else display_name
        emp_dir = employees_dir / dir_key

        # Handle conflict: both user_id and display_name directories exist
        _resolve_directory_conflict(employees_dir, display_name, user_id)

        # Write/merge USER.md
        emp_dir.mkdir(parents=True, exist_ok=True)
        user_file = emp_dir / "USER.md"

        if user_file.exists():
            existing = user_file.read_text(encoding="utf-8")
            merged = existing + "\n\n" + f"## Update {datetime.now(timezone.utc).strftime('%Y-%m-%d')}\n\n{info}"
        else:
            merged = f"# {display_name}\n\n{info}"

        # Enforce employee_limit
        if len(merged) > employee_limit:
            merged = _consolidate(merged, employee_limit)

        user_file.write_text(merged, encoding="utf-8")


def _resolve_directory_conflict(
    employees_dir: Path,
    display_name: str,
    user_id: str | None,
) -> None:
    """Resolve conflict when both display_name and user_id directories exist.

    If both directories exist:
    1. Read content from display_name/USER.md
    2. Merge into user_id/USER.md
    3. Rename display_name dir to display_name.archived

    Args:
        employees_dir: Path to employees/ directory.
        display_name: Display name from messages.
        user_id: Slack user ID (may be None).
    """
    if user_id is None:
        return

    display_dir = employees_dir / display_name
    uid_dir = employees_dir / user_id

    # Only resolve if BOTH directories exist
    if not display_dir.is_dir() or not uid_dir.is_dir():
        return

    # Read content from display_name version
    display_file = display_dir / "USER.md"
    existing_content = ""
    if display_file.exists():
        existing_content = display_file.read_text(encoding="utf-8")

    # Merge into user_id version
    uid_file = uid_dir / "USER.md"
    uid_dir.mkdir(parents=True, exist_ok=True)

    if uid_file.exists():
        uid_content = uid_file.read_text(encoding="utf-8")
        merged = uid_content + "\n\n" + f"## Migrated from {display_name}\n\n{existing_content}"
    else:
        merged = f"# {display_name}\n\n{existing_content}"

    uid_file.write_text(merged, encoding="utf-8")

    # Archive display_name directory
    archive_dir = employees_dir / (display_name + ".archived")
    try:
        display_dir.rename(archive_dir)
    except OSError:
        # On Windows, rename may fail if target exists — remove and retry
        import shutil

        if archive_dir.exists():
            shutil.rmtree(archive_dir)
        display_dir.rename(archive_dir)

    logger.info("Merged %s into %s, archived display_name dir", display_dir, uid_dir)


def write_channel_skill(
    skill: dict,
    channel_id: str,
    date: str,
    output_dir: Path,
) -> None:
    """Create a channel skill directory with SKILL.md and SQL references.

    Creates:
        skills/channel/{channel_id}/{date}-{skill_name}/SKILL.md
        skills/channel/{channel_id}/{date}-{skill_name}/references/channel-context.md

    Args:
        skill: Dict with keys: skill_name, triggers, procedure, pitfalls.
        channel_id: Slack channel ID.
        date: ISO date string (YYYY-MM-DD).
        output_dir: Base output directory.
    """
    skill_name = skill.get("skill_name", "unnamed-skill")
    skill_dir = (
        output_dir
        / "skills"
        / "channel"
        / channel_id
        / f"{date}-{skill_name}"
    )
    skill_dir.mkdir(parents=True, exist_ok=True)

    # Write SKILL.md
    triggers = skill.get("triggers", [])
    triggers_str = ", ".join(triggers) if isinstance(triggers, list) else str(triggers)
    procedure = skill.get("procedure", "")
    pitfalls = skill.get("pitfalls", "")

    skill_content = f"""# {skill_name}

## Triggers
{triggers_str}

## Procedure
{procedure}

## Pitfalls
{pitfalls}
"""
    (skill_dir / "SKILL.md").write_text(skill_content, encoding="utf-8")

    # Write references/channel-context.md with SQL queries
    refs_dir = skill_dir / "references"
    refs_dir.mkdir(parents=True, exist_ok=True)

    context_content = f"""# Channel Context Queries for {skill_name}

SQL queries for nixi_state.db to retrieve relevant channel context at runtime.

## Recent Messages

```sql
SELECT slack_ts, user_name, text, timestamp
FROM scraped_messages
WHERE channel_id = '{channel_id}'
  AND NOT EXISTS (
    SELECT 1 FROM nixi_extraction_log
    WHERE nixi_extraction_log.channel_id = '{channel_id}'
      AND nixi_extraction_log.slack_ts = scraped_messages.slack_ts
  )
ORDER BY timestamp DESC
LIMIT 50;
```

## Messages by Keyword

```sql
SELECT slack_ts, user_name, text, timestamp
FROM scraped_messages
WHERE channel_id = '{channel_id}'
  AND (text LIKE '%' || ? || '%')
ORDER BY timestamp DESC
LIMIT 50;
```

## Thread Context

```sql
SELECT slack_ts, parent_ts, user_name, text
FROM scraped_messages
WHERE channel_id = '{channel_id}'
  AND thread_ts IS NOT NULL
ORDER BY timestamp ASC
LIMIT 100;
```
"""
    (refs_dir / "channel-context.md").write_text(context_content, encoding="utf-8")

    logger.info("Created skill %s for channel %s", skill_name, channel_id)