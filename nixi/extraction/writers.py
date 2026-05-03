"""File writers with consolidation logic and conflict resolution for nixi extraction.

Handles:
- ORG_FACTS.md write/merge with aggressive consolidation at memory_limit
- RULES.md append with deduplication and truncation at rules_limit
- Employee USER.md creation, merge, and directory conflict resolution
- Channel skill directory structure with SQL references

All writers target hermes-consumed paths under HERMES_HOME and validate
paths with safe_path() to prevent path traversal in multi-tenant deployments.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from pathlib import Path

from nixi.models import UserMap
from nixi.path_validator import PathTraversalError, safe_path

logger = logging.getLogger(__name__)


def write_org_facts(
    facts: str,
    hermes_home: Path,
    memory_limit: int | None = None,
) -> None:
    """Write/merge organizational facts to {hermes_home}/nixi/ORG_FACTS.md.

    If the file already exists, merges new facts with existing content.
    If total content exceeds memory_limit, aggressively consolidates by
    extracting key points and removing redundancy.

    Args:
        facts: New facts to write/merge.
        hermes_home: HERMES_HOME base directory.
        memory_limit: Maximum character limit. Uses NixiConfig default if None.

    Raises:
        PathTraversalError: If the resolved path escapes hermes_home.
    """
    if memory_limit is None:
        from nixi.config import NixiConfig

        memory_limit = NixiConfig.from_config().memory_limit

    # Validate path to prevent traversal outside hermes_home
    safe_path(hermes_home, "nixi/ORG_FACTS.md")
    facts_path = hermes_home / "nixi" / "ORG_FACTS.md"

    if facts_path.exists():
        existing = facts_path.read_text(encoding="utf-8")
        merged = existing + "\n\n" + facts
    else:
        merged = facts

    # Consolidate if over limit
    if len(merged) > memory_limit:
        merged = _consolidate(merged, memory_limit)

    facts_path.parent.mkdir(parents=True, exist_ok=True)
    facts_path.write_text(merged, encoding="utf-8")


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


def write_rules(rules: str, hermes_home: Path, rules_limit: int | None = None) -> None:
    """Append new rules to {hermes_home}/nixi/RULES.md.

    Never overwrites existing content. New rules are appended with a
    timestamped section header. If total content exceeds rules_limit,
    deduplicates and truncates oldest sections.

    Args:
        rules: New rules to append.
        hermes_home: HERMES_HOME base directory.
        rules_limit: Maximum character limit for RULES.md.
            Uses NixiConfig default if None.

    Raises:
        PathTraversalError: If the resolved path escapes hermes_home.
    """
    if rules_limit is None:
        from nixi.config import NixiConfig

        rules_limit = NixiConfig.from_config().rules_limit

    # Validate path to prevent traversal outside hermes_home
    safe_path(hermes_home, "nixi/RULES.md")
    rules_path = hermes_home / "nixi" / "RULES.md"
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    new_section = f"\n\n## Extracted {timestamp}\n\n{rules}"

    rules_path.parent.mkdir(parents=True, exist_ok=True)

    if rules_path.exists():
        existing = rules_path.read_text(encoding="utf-8")
        content = existing + new_section
    else:
        content = f"# RULES\n\n{rules}\n"

    # Enforce rules_limit via consolidation
    if len(content) > rules_limit:
        content = _consolidate_rules(content, rules_limit)

    rules_path.write_text(content, encoding="utf-8")


def _consolidate_rules(text: str, limit: int) -> str:
    """Consolidate rules text to fit within a character limit.

    Strategy (adapted for timestamped sections):
    1. Parse text into sections split by ``## Extracted`` headers.
    2. Deduplicate: if multiple sections have identical content (after
       stripping headers), keep only the latest.
    3. If still over limit: truncate from the oldest section upward
       (keep newest content, drop oldest sections).
    4. Add a consolidation notice when truncation occurs.

    Args:
        text: Full RULES.md content with timestamped sections.
        limit: Maximum character count.

    Returns:
        Consolidated text under ``limit`` characters.
    """
    # Split into timestamped sections — keep the file header (before first
    # ## Extracted) as the preamble.
    section_pattern = re.compile(r"(?=## Extracted )", re.MULTILINE)
    parts = section_pattern.split(text)

    preamble = parts[0] if parts else ""
    sections = parts[1:] if len(parts) > 1 else []

    # Deduplicate: if two sections have identical content (ignoring the
    # ## Extracted ... header line), keep only the latest.
    seen_content: dict[str, int] = {}
    deduped_sections: list[str] = []
    for idx, section in enumerate(sections):
        # Strip the header line to compare body content
        lines = section.splitlines()
        body = "\n".join(lines[1:]).strip() if len(lines) > 1 else ""
        if body in seen_content:
            # Remove the earlier duplicate, keep this one (latest)
            earlier_idx = seen_content[body]
            # Mark earlier for removal by replacing with empty string
            deduped_sections[earlier_idx] = ""
        deduped_sections.append(section)
        seen_content[body] = len(deduped_sections) - 1

    # Filter out emptied duplicates
    deduped_sections = [s for s in deduped_sections if s]

    result = preamble + "".join(deduped_sections)
    if len(result) <= limit:
        return result

    # Truncate from oldest section upward — drop earliest sections first.
    # Build output by starting with the newest section and adding older
    # sections as long as they fit within the limit.
    notice = "<!-- rules consolidated, oldest entries dropped -->\n"
    notice_len = len(notice)
    header_budget = len(preamble) + notice_len

    # Always include at least the newest section (possibly truncated)
    if not deduped_sections:
        return preamble

    truncated: list[str] = []
    for section in reversed(deduped_sections):
        candidate_len = header_budget + sum(len(s) for s in truncated) + len(section)
        if candidate_len <= limit:
            truncated.insert(0, section)
        elif not truncated:
            # First (newest) section doesn't fit — include it truncated
            remaining = limit - header_budget
            if remaining > 0:
                truncated.append(section[:remaining])
            break
        # else: skip this (older) section

    result = preamble + notice + "".join(truncated)

    if len(result) > limit:
        # Last resort: hard truncate
        result = result[:limit]

    return result


def write_employee_info(
    employees: list[dict],
    hermes_home: Path,
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
        hermes_home: HERMES_HOME base directory.
        user_map: Bidirectional user mapping for conflict resolution.
        employee_limit: Char limit per employee. Uses config default if None.

    Raises:
        PathTraversalError: If any employee path resolves outside hermes_home.
    """
    if employee_limit is None:
        from nixi.config import NixiConfig

        employee_limit = NixiConfig.from_config().employee_limit

    employees_dir = hermes_home / "employees"
    employees_dir.mkdir(parents=True, exist_ok=True)

    for emp in employees:
        display_name = emp.get("display_name", "unknown")
        user_id = emp.get("user_id")
        info = emp.get("info", "")

        # Determine directory key: prefer user_id, fall back to display_name
        dir_key = user_id if user_id else display_name

        # Validate path to prevent traversal — dir_key comes from employee data
        safe_path(hermes_home, f"employees/{dir_key}/USER.md")
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
    hermes_home: Path,
) -> None:
    """Create a channel skill directory with SKILL.md and SQL references.

    Creates:
        skills/channel/{channel_id}/{date}-{skill_name}/SKILL.md
        skills/channel/{channel_id}/{date}-{skill_name}/references/channel-context.md

    Args:
        skill: Dict with keys: skill_name, triggers, procedure, pitfalls.
        channel_id: Slack channel ID.
        date: ISO date string (YYYY-MM-DD).
        hermes_home: HERMES_HOME base directory.

    Raises:
        PathTraversalError: If channel_id or skill_name resolve outside hermes_home.
    """
    skill_name = skill.get("skill_name", "unnamed-skill")

    # Validate path to prevent traversal — channel_id comes from Slack (external input)
    safe_path(hermes_home, f"skills/channel/{channel_id}/{date}-{skill_name}")
    skill_dir = (
        hermes_home
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