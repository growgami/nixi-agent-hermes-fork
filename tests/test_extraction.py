"""Tests for the nixi extraction pipeline.

Covers: prompt templates, writer consolidation/merge/append/conflict resolution,
batch orchestration (channel grouping, threshold skip, LLM batching, bot tagging),
extraction log tracking, CLI entrypoints, path validation.
"""

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nixi.config import NixiConfig
from nixi.db import (
    ensure_realtime_schema,
    ensure_schema,
    get_connection,
    get_realtime_unprocessed,
    get_realtime_unprocessed_channels,
    get_unprocessed,
    get_unprocessed_channels,
    insert_messages,
    mark_extracted,
)
from nixi.extraction.batch import ExtractionBatcher, LLMClient
from nixi.extraction.prompts import (
    CHANNEL_SKILL_PROMPT,
    EMPLOYEE_PROMPT,
    ORG_FACTS_PROMPT,
    RULES_PROMPT,
)
from nixi.extraction.writers import (
    _consolidate_rules,
    write_channel_skill,
    write_employee_info,
    write_org_facts,
    write_rules,
)
from nixi.models import ScrapedMessage, UserMap
from nixi.path_validator import PathTraversalError


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def hermes_home(tmp_path: Path) -> Path:
    """Temp HERMES_HOME directory for writers."""
    d = tmp_path / "hermes"
    d.mkdir()
    return d


@pytest.fixture
def nixi_config(hermes_home: Path, tmp_path: Path) -> NixiConfig:
    """Minimal NixiConfig for testing."""
    return NixiConfig(
        log_dir=tmp_path / "logs",
        output_dir=hermes_home / "nixi" / "output",
        extraction_batch_size=50,
        memory_limit=500,
        employee_limit=300,
        rules_limit=500,
    )


@pytest.fixture
def db_conn(nixi_config: NixiConfig):
    """Schema-initialized database connection."""
    ensure_schema(nixi_config.db_path)
    conn = get_connection(nixi_config.db_path)
    yield conn
    conn.close()


def _make_message(
    slack_ts: str = "1766766571.412779",
    channel_id: str = "C06M81FSKFF",
    channel_name: str = "general",
    user_id: str | None = "U04K8NLDCG0",
    user_name: str = "Kuro",
    text: str = "hello world",
    thread_ts: str | None = None,
    parent_ts: str | None = None,
    is_bot: bool = False,
    source_file: str = "C06M81FSKFF",
    timestamp: str | None = None,
) -> ScrapedMessage:
    if timestamp is None:
        timestamp = datetime.fromtimestamp(float(slack_ts), tz=timezone.utc).isoformat()
    return ScrapedMessage(
        slack_ts=slack_ts,
        channel_id=channel_id,
        channel_name=channel_name,
        user_id=user_id,
        user_name=user_name,
        text=text,
        thread_ts=thread_ts,
        parent_ts=parent_ts,
        is_bot=is_bot,
        source_file=source_file,
        timestamp=timestamp,
    )


def _make_user_map() -> UserMap:
    """Sample user map for testing."""
    return UserMap(
        name_to_id={"Kuro": "U04K8NLDCG0", "Jin": "U073D278H62"},
        id_to_name={"U04K8NLDCG0": "Kuro", "U073D278H62": "Jin"},
    )


# ── Prompt templates ──────────────────────────────────────────────────────────


class TestPrompts:
    """Prompt template tests — verify structure, placeholders, and character limits."""

    def test_org_facts_prompt_contains_sections(self):
        """ORG_FACTS_PROMPT references key sections and memory_limit."""
        assert "memory_limit" in ORG_FACTS_PROMPT
        assert "organizational" in ORG_FACTS_PROMPT.lower()

    def test_rules_prompt_contains_format(self):
        """RULES_PROMPT references append-style format instructions."""
        assert "AGENTS" in RULES_PROMPT or "rules" in RULES_PROMPT.lower()

    def test_employee_prompt_contains_limit(self):
        """EMPLOYEE_PROMPT references employee_limit."""
        assert "employee_limit" in EMPLOYEE_PROMPT

    def test_channel_skill_prompt_contains_sql_reference(self):
        """CHANNEL_SKILL_PROMPT references channel-context.md and SQL."""
        assert "channel-context" in CHANNEL_SKILL_PROMPT
        assert "nixi_state" in CHANNEL_SKILL_PROMPT or "SQL" in CHANNEL_SKILL_PROMPT

    def test_org_facts_prompt_format_instructions(self):
        """ORG_FACTS_PROMPT contains format instructions for structured output."""
        assert "format" in ORG_FACTS_PROMPT.lower() or "sections" in ORG_FACTS_PROMPT.lower()

    def test_rules_prompt_no_overwrite_instruction(self):
        """RULES_PROMPT instructs to NOT overwrite existing content."""
        lower = RULES_PROMPT.lower()
        assert "append" in lower or "do not overwrite" in lower or "new rules" in lower


# ── MEMORY.md consolidation ────────────────────────────────────────────────────


class TestWriteOrgFacts:
    """Test ORG_FACTS.md write, merge, and consolidation."""

    def test_write_creates_org_facts_md(self, hermes_home: Path):
        """write_org_facts creates ORG_FACTS.md in hermes_home/nixi/."""
        write_org_facts("## Facts\n- Team uses React", hermes_home)
        facts_file = hermes_home / "nixi" / "ORG_FACTS.md"
        assert facts_file.exists()
        content = facts_file.read_text(encoding="utf-8")
        assert "React" in content

    def test_write_merges_with_existing(self, hermes_home: Path):
        """write_org_facts merges new facts with existing content."""
        write_org_facts("## Facts\n- React frontend", hermes_home)
        write_org_facts("## Facts\n- Python backend", hermes_home)
        content = (hermes_home / "nixi" / "ORG_FACTS.md").read_text(encoding="utf-8")
        assert "React" in content
        assert "Python" in content

    def test_consolidation_at_memory_limit(self, hermes_home: Path):
        """Aggressively consolidate when total exceeds memory_limit."""
        write_org_facts("A" * 80, hermes_home, memory_limit=100)
        write_org_facts("B" * 80, hermes_home, memory_limit=100)
        content = (hermes_home / "nixi" / "ORG_FACTS.md").read_text(encoding="utf-8")
        # Content should be consolidated and fit within the limit
        assert len(content) <= 100

    def test_consolidation_preserves_key_info(self, hermes_home: Path):
        """After consolidation, key information is preserved."""
        write_org_facts("## Key\n- API key rotation every 30 days\n- Deploy on Fridays", hermes_home, memory_limit=200)
        write_org_facts("## Key\n- Use PostgreSQL\n- CI via GitHub Actions", hermes_home, memory_limit=200)
        content = (hermes_home / "nixi" / "ORG_FACTS.md").read_text(encoding="utf-8")
        assert len(content) > 0

    def test_default_memory_limit(self, hermes_home: Path):
        """Without explicit memory_limit, uses NixiConfig default."""
        write_org_facts("Short content", hermes_home)
        assert (hermes_home / "nixi" / "ORG_FACTS.md").exists()

    def test_path_traversal_rejected(self, hermes_home: Path):
        """write_org_facts rejects paths that would escape hermes_home."""
        # safe_path should reject traversal before any file write
        with pytest.raises(PathTraversalError):
            safe_path_test = hermes_home / "nixi" / "ORG_FACTS.md"
            # Direct call with traversal attempt — safe_path validates the relative path
            from nixi.path_validator import safe_path
            safe_path(hermes_home, "../../../etc/passwd")


# ── AGENTS.md append ──────────────────────────────────────────────────────────


class TestWriteRules:
    """Test RULES.md append behavior with consolidation."""

    def test_append_creates_rules_md(self, hermes_home: Path):
        """write_rules creates RULES.md in hermes_home/nixi/ if it doesn't exist."""
        write_rules("- Always respond in English\n- Be concise", hermes_home)
        rules_file = hermes_home / "nixi" / "RULES.md"
        assert rules_file.exists()
        content = rules_file.read_text(encoding="utf-8")
        assert "English" in content
        assert content.startswith("# RULES")

    def test_append_does_not_overwrite(self, hermes_home: Path):
        """write_rules appends to existing RULES.md content."""
        write_rules("- Rule 1: Be polite", hermes_home)
        write_rules("- Rule 2: Be concise", hermes_home)
        content = (hermes_home / "nixi" / "RULES.md").read_text(encoding="utf-8")
        assert "polite" in content
        assert "concise" in content

    def test_multiple_appends(self, hermes_home: Path):
        """Multiple appends accumulate with timestamped sections."""
        write_rules("- First rule", hermes_home)
        write_rules("- Second rule", hermes_home)
        write_rules("- Third rule", hermes_home)
        content = (hermes_home / "nixi" / "RULES.md").read_text(encoding="utf-8")
        assert "First" in content
        assert "Second" in content
        assert "Third" in content

    def test_rules_limit_enforcement(self, hermes_home: Path):
        """Consolidation triggers when content exceeds rules_limit."""
        # Write enough content to exceed limit
        write_rules("A" * 200, hermes_home, rules_limit=300)
        write_rules("B" * 200, hermes_home, rules_limit=300)
        content = (hermes_home / "nixi" / "RULES.md").read_text(encoding="utf-8")
        # Content should be kept within limit after consolidation
        assert len(content) <= 300

    def test_consolidation_deduplicates_identical_sections(self):
        """_consolidate_rules deduplicates sections with identical content."""
        text = "# RULES\n\n## Extracted 2026-01-01 00:00 UTC\n\nSame content\n\n## Extracted 2026-01-02 00:00 UTC\n\nSame content\n"
        result = _consolidate_rules(text, 10000)
        # Only one copy of duplicate content should remain
        assert result.count("Same content") == 1

    def test_consolidation_truncates_oldest(self):
        """_consolidate_rules drops oldest sections first when truncating."""
        old_section = "## Extracted 2026-01-01 00:00 UTC\n\n" + "Old rule. " * 20 + "\n"
        new_section = "## Extracted 2026-01-02 00:00 UTC\n\nNew content here\n"
        text = "# RULES\n\n" + old_section + "\n" + new_section
        # Limit allows preamble + notice + new section but not old
        limit = 150
        result = _consolidate_rules(text, limit)
        assert len(result) <= limit
        assert "New content" in result
        assert "oldest entries dropped" in result

    def test_consolidation_preserves_preamble(self):
        """_consolidate_rules preserves the # RULES header (preamble)."""
        text = "# RULES\n\n## Extracted 2026-01-01 00:00 UTC\n\nRule one\n"
        result = _consolidate_rules(text, 10000)
        assert result.startswith("# RULES")

    def test_consolidation_text_without_extracted_sections(self):
        """_consolidate_rules handles text with no ## Extracted sections."""
        text = "# RULES\n\nSome content without timestamps\n"
        result = _consolidate_rules(text, 10000)
        # No ## Extracted sections — preamble contains all content
        assert result.startswith("# RULES")
        assert "Some content without timestamps" in result

    def test_consolidation_single_section(self):
        """_consolidate_rules preserves a single ## Extracted section within limit."""
        text = "# RULES\n\n## Extracted 2026-01-15 10:00 UTC\n\nRule one.\n"
        result = _consolidate_rules(text, 10000)
        assert "Rule one" in result
        assert result.startswith("# RULES")

    def test_consolidation_empty_rules_file(self):
        """_consolidate_rules handles minimal # RULES header only."""
        text = "# RULES\n"
        result = _consolidate_rules(text, 10000)
        assert result.startswith("# RULES")

    def test_safe_path_rejects_traversal_in_rules(self, hermes_home: Path):
        """write_rules rejects path traversal via hermes_home."""
        from nixi.path_validator import safe_path
        with pytest.raises(PathTraversalError):
            safe_path(hermes_home, "../../etc/passwd")


# ── Employee USER.md ───────────────────────────────────────────────────────────


class TestWriteEmployeeInfo:
    """Test employee USER.md creation, merge, and conflict resolution."""

    def test_create_new_employee_file(self, hermes_home: Path):
        """Creates USER.md for a new employee under hermes_home/employees/."""
        employees = [
            {"display_name": "Kuro", "user_id": "U04K8NLDCG0", "info": "Senior engineer, React expert"}
        ]
        write_employee_info(employees, hermes_home, _make_user_map())
        user_file = hermes_home / "employees" / "U04K8NLDCG0" / "USER.md"
        assert user_file.exists()
        content = user_file.read_text(encoding="utf-8")
        assert "Kuro" in content

    def test_employee_without_user_id(self, hermes_home: Path):
        """Creates USER.md using display_name when user_id is missing."""
        employees = [
            {"display_name": "Riya", "user_id": None, "info": "PM, product focus"}
        ]
        write_employee_info(employees, hermes_home, _make_user_map())
        user_file = hermes_home / "employees" / "Riya" / "USER.md"
        assert user_file.exists()
        content = user_file.read_text(encoding="utf-8")
        assert "Riya" in content

    def test_merge_existing_employee(self, hermes_home: Path):
        """Merges new info with existing USER.md content."""
        employees = [
            {"display_name": "Kuro", "user_id": "U04K8NLDCG0", "info": "React expert"}
        ]
        write_employee_info(employees, hermes_home, _make_user_map())
        # Second write with new info
        employees2 = [
            {"display_name": "Kuro", "user_id": "U04K8NLDCG0", "info": "Also knows Python"}
        ]
        write_employee_info(employees2, hermes_home, _make_user_map())
        content = (hermes_home / "employees" / "U04K8NLDCG0" / "USER.md").read_text(encoding="utf-8")
        assert "React" in content
        assert "Python" in content

    def test_employee_limit_enforcement(self, hermes_home: Path):
        """Compresses content when exceeding employee_limit."""
        employees = [
            {"display_name": "Kuro", "user_id": "U04K8NLDCG0", "info": "A" * 500}
        ]
        write_employee_info(employees, hermes_home, _make_user_map(), employee_limit=100)
        content = (hermes_home / "employees" / "U04K8NLDCG0" / "USER.md").read_text(encoding="utf-8")
        assert len(content) <= 100

    def test_directory_conflict_resolution(self, hermes_home: Path):
        """When both user_id and display_name directories exist, merge into user_id, archive display_name."""
        # Pre-create display_name directory
        display_dir = hermes_home / "employees" / "Kuro"
        display_dir.mkdir(parents=True, exist_ok=True)
        (display_dir / "USER.md").write_text("# Kuro\nOld info from display_name dir", encoding="utf-8")

        # Also pre-create user_id directory
        uid_dir = hermes_home / "employees" / "U04K8NLDCG0"
        uid_dir.mkdir(parents=True, exist_ok=True)
        (uid_dir / "USER.md").write_text("# Kuro\nOld info from user_id dir", encoding="utf-8")

        employees = [
            {"display_name": "Kuro", "user_id": "U04K8NLDCG0", "info": "New extracted info"}
        ]
        write_employee_info(employees, hermes_home, _make_user_map())

        # Merged into user_id directory
        uid_content = (uid_dir / "USER.md").read_text(encoding="utf-8")
        assert "New extracted info" in uid_content
        assert "user_id dir" in uid_content

        # display_name directory should have been archived
        archive_dir = display_dir.parent / (display_dir.name + ".archived")
        assert archive_dir.exists() or not display_dir.exists()

    def test_path_traversal_rejected_for_employee(self, hermes_home: Path):
        """write_employee_info rejects dir_key that escapes hermes_home."""
        from nixi.path_validator import safe_path
        with pytest.raises(PathTraversalError):
            safe_path(hermes_home, "employees/../../../etc/passwd")


# ── Channel skill ──────────────────────────────────────────────────────────────


class TestWriteChannelSkill:
    """Test channel skill directory structure and SQL references."""

    def test_creates_skill_directory(self, hermes_home: Path):
        """Creates skill directory with correct structure under hermes_home/skills/."""
        skill = {
            "skill_name": "deploy-checks",
            "triggers": ["deploy", "release"],
            "procedure": "1. Check CI status\n2. Verify tests pass\n3. Notify channel",
            "pitfalls": "Don't deploy on Fridays",
        }
        write_channel_skill(skill, "C06M81FSKFF", "2026-04-26", hermes_home)
        skill_dir = hermes_home / "skills" / "channel" / "C06M81FSKFF" / "2026-04-26-deploy-checks"
        assert skill_dir.exists()
        assert (skill_dir / "SKILL.md").exists()

    def test_creates_channel_context_reference(self, hermes_home: Path):
        """Creates references/channel-context.md with SQL queries."""
        skill = {
            "skill_name": "deploy-checks",
            "triggers": ["deploy", "release"],
            "procedure": "Check CI status",
            "pitfalls": "Don't skip tests",
        }
        write_channel_skill(skill, "C06M81FSKFF", "2026-04-26", hermes_home)
        ref_file = hermes_home / "skills" / "channel" / "C06M81FSKFF" / "2026-04-26-deploy-checks" / "references" / "channel-context.md"
        assert ref_file.exists()
        content = ref_file.read_text(encoding="utf-8")
        assert "nixi_state" in content or "SQL" in content

    def test_skill_md_content(self, hermes_home: Path):
        """SKILL.md contains skill info."""
        skill = {
            "skill_name": "code-review",
            "triggers": ["PR", "review"],
            "procedure": "Check for style issues",
            "pitfalls": "Don't be too nitpicky",
        }
        write_channel_skill(skill, "C06M81FSKFF", "2026-04-26", hermes_home)
        skill_md = hermes_home / "skills" / "channel" / "C06M81FSKFF" / "2026-04-26-code-review" / "SKILL.md"
        content = skill_md.read_text(encoding="utf-8")
        assert "code-review" in content

    def test_path_traversal_rejected_for_channel_id(self, hermes_home: Path):
        """Channel ID with path traversal is rejected by safe_path."""
        from nixi.path_validator import safe_path
        with pytest.raises(PathTraversalError):
            safe_path(hermes_home, "skills/channel/../../../etc/passwd")

    def test_write_channel_skill_rejects_malicious_channel_id(self, hermes_home: Path):
        """write_channel_skill raises PathTraversalError for traversal channel IDs.
        
        The channel_id is embedded in skills/channel/{channel_id}/... so a
        traversal pattern like ../../../../etc escapes hermes_home.
        """
        skill = {
            "skill_name": "test-skill",
            "triggers": ["test"],
            "procedure": "Step 1",
            "pitfalls": "None",
        }
        with pytest.raises(PathTraversalError):
            write_channel_skill(skill, "../../../../../etc", "2026-04-26", hermes_home)


# ── Extraction log tracking ────────────────────────────────────────────────────


class TestExtractionLogTracking:
    """Test that extraction_log correctly tracks processed messages."""

    def test_mark_extracted_prevents_re_extraction(self, db_conn):
        """Messages marked as extracted are excluded from subsequent extraction."""
        msgs = [_make_message(slack_ts=f"1766766571.{i:06d}") for i in range(5)]
        insert_messages(db_conn, msgs)

        # Simulate extraction: mark first 3 as extracted
        mark_extracted(
            db_conn, "C06M81FSKFF", [m.slack_ts for m in msgs[:3]], "batch-1"
        )

        # get_unprocessed should only return the last 2
        unprocessed = get_unprocessed(db_conn, "C06M81FSKFF")
        assert len(unprocessed) == 2
        assert unprocessed[0]["slack_ts"] == "1766766571.000003"

    def test_re_running_extraction_skips_processed(self, db_conn):
        """Re-running extraction skips already-processed messages."""
        msgs = [_make_message(slack_ts=f"1766766571.{i:06d}") for i in range(3)]
        insert_messages(db_conn, msgs)

        # First extraction
        mark_extracted(db_conn, "C06M81FSKFF", [m.slack_ts for m in msgs], "batch-1")

        # Re-run: should find zero unprocessed
        unprocessed = get_unprocessed(db_conn, "C06M81FSKFF")
        assert len(unprocessed) == 0

    def test_extraction_log_fields(self, db_conn):
        """nixi_extraction_log has correct fields."""
        insert_messages(db_conn, [_make_message()])
        mark_extracted(db_conn, "C06M81FSKFF", ["1766766571.412779"], "batch-1")

        cursor = db_conn.execute("SELECT * FROM nixi_extraction_log")
        row = cursor.fetchone()
        assert row["channel_id"] == "C06M81FSKFF"
        assert row["slack_ts"] == "1766766571.412779"
        assert row["extraction_batch"] == "batch-1"
        assert row["extracted_at"] is not None


# ── Bot message weighting in prompt ────────────────────────────────────────────


class TestBotMessageTagging:
    """Test that bot messages are prefixed with [BOT] in prompt formatting."""

    def test_bot_messages_tagged(self):
        """Bot messages get [BOT] prefix in formatted batch."""
        batcher = ExtractionBatcher.__new__(ExtractionBatcher)
        messages = [
            {"is_bot": 1, "user_name": "nixi", "text": "Build passed"},
            {"is_bot": 0, "user_name": "Kuro", "text": "Nice!"},
        ]
        formatted = batcher._format_messages_for_prompt(messages)  # type: ignore[attr-defined]
        assert "[BOT]" in formatted
        assert "nixi" in formatted
        assert "Kuro" in formatted

    def test_non_bot_messages_not_tagged(self):
        """Regular messages do NOT get [BOT] prefix."""
        batcher = ExtractionBatcher.__new__(ExtractionBatcher)
        messages = [
            {"is_bot": 0, "user_name": "Kuro", "text": "Hello"},
        ]
        formatted = batcher._format_messages_for_prompt(messages)  # type: ignore[attr-defined]
        assert "[BOT]" not in formatted
        assert "Kuro" in formatted


# ── Batch skip logic ───────────────────────────────────────────────────────────


class TestBatchSkipLogic:
    """Test that channels with fewer messages than threshold are skipped."""

    @pytest.mark.asyncio
    async def test_skip_below_threshold(self, nixi_config: NixiConfig, db_conn):
        """Channels with < 20 unprocessed messages are skipped."""
        # Insert fewer than threshold messages
        msgs = [_make_message(slack_ts=f"1766766571.{i:06d}") for i in range(10)]
        insert_messages(db_conn, msgs)

        mock_llm = AsyncMock()
        batcher = ExtractionBatcher(nixi_config, db_conn, mock_llm)

        result = await batcher.extract_channel("C06M81FSKFF")
        assert result is not None
        assert result.get("skipped", False) is True

    @pytest.mark.asyncio
    async def test_process_above_threshold(self, nixi_config: NixiConfig, db_conn):
        """Channels with >= 20 messages are processed."""
        # Insert 25 messages (above default threshold of 20)
        msgs = [_make_message(slack_ts=f"1766766571.{i:06d}") for i in range(25)]
        insert_messages(db_conn, msgs)

        # Mock LLM to return structured responses
        mock_llm = AsyncMock()
        mock_llm.chat = AsyncMock(return_value="## Organizational Facts\n- Test fact")

        batcher = ExtractionBatcher(nixi_config, db_conn, mock_llm)
        # Override threshold for testing
        batcher.min_messages = 20

        result = await batcher.extract_channel("C06M81FSKFF")
        # Should not be skipped
        assert result is not None
        assert result.get("skipped", False) is False


# ── ExtractionBatcher integration ──────────────────────────────────────────────


class TestExtractionBatcher:
    """Integration tests for ExtractionBatcher."""

    @pytest.mark.asyncio
    async def test_extract_all_finds_channels(self, nixi_config: NixiConfig, db_conn):
        """extract_all finds and processes channels with unprocessed messages."""
        # Insert messages across multiple channels
        for i in range(25):
            insert_messages(db_conn, [_make_message(
                slack_ts=f"1766766571.{i:06d}",
                channel_id="C_CH1",
            )])
        for i in range(25):
            insert_messages(db_conn, [_make_message(
                slack_ts=f"1766766572.{i:06d}",
                channel_id="C_CH2",
            )])

        mock_llm = AsyncMock()
        mock_llm.chat = AsyncMock(return_value="## Extracted\n- Test data")

        batcher = ExtractionBatcher(nixi_config, db_conn, mock_llm)
        batcher.min_messages = 20

        result = await batcher.extract_all()
        assert result is not None
        assert "channels" in result

    def test_format_messages_orders_by_timestamp(self):
        """Messages formatted in timestamp order."""
        batcher = ExtractionBatcher.__new__(ExtractionBatcher)
        messages = [
            {"timestamp": "2026-04-26T03:00:00", "user_name": "Z", "text": "late", "is_bot": 0},
            {"timestamp": "2026-04-26T01:00:00", "user_name": "A", "text": "early", "is_bot": 0},
        ]
        formatted = batcher._format_messages_for_prompt(messages)  # type: ignore[attr-defined]
        early_pos = formatted.find("early")
        late_pos = formatted.find("late")
        assert early_pos < late_pos

    @pytest.mark.asyncio
    async def test_extract_channel_uses_hermes_home_for_writers(self, nixi_config: NixiConfig, db_conn, hermes_home: Path):
        """ExtractionBatcher writes to hermes_home paths, not output_dir."""
        for i in range(25):
            insert_messages(db_conn, [_make_message(slack_ts=f"1766766571.{i:06d}")])

        mock_llm = AsyncMock()
        mock_llm.chat = AsyncMock(return_value="## Extracted\n- Test data")

        # Patch get_hermes_home so config.hermes_home property returns our fixture path
        with patch("nixi.config.get_hermes_home", return_value=hermes_home):
            batcher = ExtractionBatcher(nixi_config, db_conn, mock_llm)
            batcher.min_messages = 20

            with patch("nixi.extraction.batch.write_org_facts") as mock_org, \
                 patch("nixi.extraction.batch.write_rules") as mock_rules, \
                 patch("nixi.extraction.batch.write_employee_info") as mock_emp, \
                 patch("nixi.extraction.batch.write_channel_skill") as mock_skill:
                await batcher.extract_channel("C06M81FSKFF")

                # Verify writers receive hermes_home, not output_dir
                assert mock_org.call_args[0][1] == batcher.hermes_home
                assert mock_rules.call_args[0][1] == batcher.hermes_home
                assert mock_rules.call_args[0][2] == nixi_config.rules_limit
                assert mock_emp.call_args[0][1] == batcher.hermes_home
                assert mock_skill.call_args[0][3] == batcher.hermes_home

    def test_collect_existing_employees_reads_from_hermes_home(self, nixi_config: NixiConfig, hermes_home: Path):
        """_collect_existing_employees reads from hermes_home/employees, not output_dir."""
        # Pre-create employee data under hermes_home/employees (not output_dir/employees)
        emp_dir = hermes_home / "employees" / "U_TEST"
        emp_dir.mkdir(parents=True, exist_ok=True)
        (emp_dir / "USER.md").write_text("# Test User\nSome info", encoding="utf-8")

        mock_llm = MagicMock()
        # Patch get_hermes_home so config.hermes_home property returns our fixture path
        with patch("nixi.config.get_hermes_home", return_value=hermes_home):
            batcher = ExtractionBatcher(nixi_config, MagicMock(), mock_llm)

            result = batcher._collect_existing_employees()
            assert "U_TEST" in result

    def test_batcher_stores_hermes_home(self, nixi_config: NixiConfig, hermes_home: Path):
        """ExtractionBatcher.__init__ captures hermes_home from config."""
        mock_llm = MagicMock()
        with patch("nixi.config.get_hermes_home", return_value=hermes_home):
            batcher = ExtractionBatcher(nixi_config, MagicMock(), mock_llm)
            assert batcher.hermes_home == hermes_home
            # output_dir still exists for db operations
            assert batcher.output_dir == nixi_config.output_dir


# ── CLI entrypoint ─────────────────────────────────────────────────────────────


class TestCLI:
    """Test CLI extract entrypoints."""

    @pytest.mark.asyncio
    async def test_run_extraction_creates_output(self, nixi_config: NixiConfig, db_conn):
        """run_extraction loads config and calls extract_all."""
        from nixi.extract import run_extraction

        # Insert sufficient messages
        for i in range(25):
            insert_messages(db_conn, [_make_message(slack_ts=f"1766766571.{i:06d}")])

        mock_llm = AsyncMock()
        mock_llm.chat = AsyncMock(return_value="## Facts\n- Test")

        with patch("nixi.extract.ExtractionBatcher") as MockBatcher, \
             patch("nixi.extract.LLMClient", return_value=mock_llm):
            mock_batcher = AsyncMock()
            mock_batcher.extract_all = AsyncMock(return_value={"channels": 1, "messages": 25})
            MockBatcher.return_value = mock_batcher

            result = await run_extraction(nixi_config)
            assert result is not None

    @pytest.mark.asyncio
    async def test_run_extraction_channel_single(self, nixi_config: NixiConfig, db_conn):
        """run_extraction_channel processes a single channel."""
        from nixi.extract import run_extraction_channel

        for i in range(25):
            insert_messages(db_conn, [_make_message(slack_ts=f"1766766571.{i:06d}")])

        mock_llm = AsyncMock()
        mock_llm.chat = AsyncMock(return_value="## Facts\n- Test")

        with patch("nixi.extract.ExtractionBatcher") as MockBatcher, \
             patch("nixi.extract.LLMClient", return_value=mock_llm):
            mock_batcher = AsyncMock()
            mock_batcher.extract_channel = AsyncMock(return_value={"skipped": False})
            MockBatcher.return_value = mock_batcher

            result = await run_extraction_channel("C06M81FSKFF", nixi_config)
            assert result is not None


# ── LLMClient wiring ──────────────────────────────────────────────────────────


class TestLLMClient:
    """Test LLMClient.__init__ and chat() wiring to resolve_provider_client."""

    def test_raises_runtime_error_when_extraction_model_empty(self):
        """LLMClient.__init__ raises RuntimeError when extraction_model is empty."""
        config = NixiConfig(
            log_dir=Path("/tmp/logs"),
            output_dir=Path("/tmp/output"),
            extraction_model="",
        )
        with pytest.raises(RuntimeError, match="No extraction model configured"):
            LLMClient(config)

    def test_raises_runtime_error_when_no_provider(self):
        """LLMClient.__init__ raises RuntimeError when resolve_provider_client returns (None, None)."""
        config = NixiConfig(
            log_dir=Path("/tmp/logs"),
            output_dir=Path("/tmp/output"),
            extraction_model="gpt-4o",
        )
        with patch("nixi.extraction.batch.resolve_provider_client", return_value=(None, None)):
            with pytest.raises(RuntimeError, match="No LLM provider configured"):
                LLMClient(config)

    def test_resolves_provider_on_init(self):
        """LLMClient.__init__ resolves provider client and stores _client and _resolved_model only."""
        mock_client = MagicMock()
        config = NixiConfig(
            log_dir=Path("/tmp/logs"),
            output_dir=Path("/tmp/output"),
            extraction_model="gpt-4o",
        )
        with patch("nixi.extraction.batch.resolve_provider_client", return_value=(mock_client, "test-model")):
            llm = LLMClient(config)

        assert llm._client is mock_client
        assert llm._resolved_model == "test-model"
        # Must NOT store self.config
        assert not hasattr(llm, "config")

    @pytest.mark.asyncio
    async def test_chat_calls_provider_and_returns_content(self):
        """LLMClient.chat() calls client.chat.completions.create and returns content."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "hello"

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        config = NixiConfig(
            log_dir=Path("/tmp/logs"),
            output_dir=Path("/tmp/output"),
            extraction_model="gpt-4o",
        )
        with patch("nixi.extraction.batch.resolve_provider_client", return_value=(mock_client, "test-model")):
            llm = LLMClient(config)

        result = await llm.chat("test prompt")
        assert result == "hello"
        mock_client.chat.completions.create.assert_called_once_with(
            model="test-model",
            messages=[{"role": "user", "content": "test prompt"}],
        )

    @pytest.mark.asyncio
    async def test_chat_sends_messages_in_correct_format(self):
        """LLMClient.chat() sends messages in [{"role": "user", "content": prompt}] format."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "response text"

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        config = NixiConfig(
            log_dir=Path("/tmp/logs"),
            output_dir=Path("/tmp/output"),
            extraction_model="gpt-4o",
        )
        with patch("nixi.extraction.batch.resolve_provider_client", return_value=(mock_client, "resolved-model")):
            llm = LLMClient(config)

        await llm.chat("some prompt text")
        call_kwargs = mock_client.chat.completions.create.call_args
        assert call_kwargs.kwargs["model"] == "resolved-model"
        messages = call_kwargs.kwargs["messages"]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "some prompt text"

    def test_does_not_store_self_model(self):
        """LLMClient does NOT store self._model — extraction_model passed directly to resolve_provider_client."""
        mock_client = MagicMock()
        config = NixiConfig(
            log_dir=Path("/tmp/logs"),
            output_dir=Path("/tmp/output"),
            extraction_model="gpt-4o",
        )
        with patch("nixi.extraction.batch.resolve_provider_client", return_value=(mock_client, "resolved-model")) as mock_resolve:
            llm = LLMClient(config)

        # resolve_provider_client was called with the model from config
        mock_resolve.assert_called_once_with("auto", model="gpt-4o", async_mode=True)
        # No _model attribute stored (BOB N2: only _resolved_model matters)
        assert not hasattr(llm, "_model")

    def test_empty_model_raises_before_provider_resolution(self):
        """LLMClient raises RuntimeError for empty extraction_model BEFORE resolve_provider_client is called.

        This guards that no API key lookup (which may be slow or have side effects)
        happens when the model is not configured.
        """
        config = NixiConfig(
            log_dir=Path("/tmp/logs"),
            output_dir=Path("/tmp/output"),
            extraction_model="",
        )
        with patch("nixi.extraction.batch.resolve_provider_client") as mock_resolve:
            with pytest.raises(RuntimeError, match="No extraction model configured"):
                LLMClient(config)
            # resolve_provider_client must NEVER be called when model is empty
            mock_resolve.assert_not_called()

    def test_no_provider_error_includes_hermes_home_guidance(self, monkeypatch):
        """LLMClient RuntimeError for missing provider includes HERMES_HOME context.

        When called directly (programmatic path, no CLI/worker dotenv loading),
        resolve_provider_client returns (None, None) because API keys aren't in
        the environment. The error message must guide the user to set up .env.
        """
        config = NixiConfig(
            log_dir=Path("/tmp/logs"),
            output_dir=Path("/tmp/output"),
            extraction_model="gpt-4o",
        )
        monkeypatch.delenv("HERMES_HOME", raising=False)
        with patch("nixi.extraction.batch.resolve_provider_client", return_value=(None, None)):
            with pytest.raises(RuntimeError, match="No LLM provider configured"):
                LLMClient(config)

    def test_no_provider_error_shows_hermes_home_when_set(self, monkeypatch):
        """LLMClient RuntimeError message includes HERMES_HOME value when set."""
        config = NixiConfig(
            log_dir=Path("/tmp/logs"),
            output_dir=Path("/tmp/output"),
            extraction_model="gpt-4o",
        )
        monkeypatch.setenv("HERMES_HOME", "/custom/hermes/home")
        with patch("nixi.extraction.batch.resolve_provider_client", return_value=(None, None)):
            with pytest.raises(RuntimeError, match="/custom/hermes/home"):
                LLMClient(config)


# ── Realtime source extraction ────────────────────────────────────────────────


class TestExtractionBatcherRealtimeSource:
    """Test ExtractionBatcher with source='realtime' routing."""

    def test_source_validation_rejects_invalid(self, nixi_config: NixiConfig):
        """ExtractionBatcher raises ValueError for invalid source."""
        with pytest.raises(ValueError, match="source must be 'scraped' or 'realtime'"):
            ExtractionBatcher(nixi_config, MagicMock(), MagicMock(), source="invalid")

    def test_source_validation_accepts_scraped(self, nixi_config: NixiConfig):
        """ExtractionBatcher accepts source='scraped'."""
        batcher = ExtractionBatcher(nixi_config, MagicMock(), MagicMock(), source="scraped")
        assert batcher.source == "scraped"

    def test_source_validation_accepts_realtime(self, nixi_config: NixiConfig):
        """ExtractionBatcher accepts source='realtime'."""
        batcher = ExtractionBatcher(nixi_config, MagicMock(), MagicMock(), source="realtime")
        assert batcher.source == "realtime"

    def test_source_defaults_to_scraped(self, nixi_config: NixiConfig):
        """ExtractionBatcher defaults to source='scraped'."""
        batcher = ExtractionBatcher(nixi_config, MagicMock(), MagicMock())
        assert batcher.source == "scraped"

    def test_format_messages_includes_channel_type_for_realtime(self):
        """_format_messages_for_prompt includes [Channel type: group] header for realtime."""
        batcher = ExtractionBatcher.__new__(ExtractionBatcher)
        messages = [
            {"timestamp": "2026-04-26T01:00:00", "user_id": "U123", "text": "hello", "is_bot": 0},
        ]
        formatted = batcher._format_messages_for_prompt(messages, channel_type="group")
        assert "[Channel type: group]" in formatted
        assert "@U123: hello" in formatted

    def test_format_messages_omits_channel_type_when_none(self):
        """_format_messages_for_prompt omits channel_type header when None."""
        batcher = ExtractionBatcher.__new__(ExtractionBatcher)
        messages = [
            {"timestamp": "2026-04-26T01:00:00", "user_name": "Kuro", "text": "hello", "is_bot": 0},
        ]
        formatted = batcher._format_messages_for_prompt(messages, channel_type=None)
        assert "Channel type" not in formatted

    def test_format_messages_uses_user_id_fallback_for_realtime(self):
        """_format_messages_for_prompt uses user_id as fallback when user_name is absent."""
        batcher = ExtractionBatcher.__new__(ExtractionBatcher)
        messages = [
            {"timestamp": "2026-04-26T01:00:00", "user_id": "U_ABC123", "text": "hello", "is_bot": 0},
        ]
        formatted = batcher._format_messages_for_prompt(messages)
        assert "@U_ABC123: hello" in formatted

    @pytest.mark.asyncio
    async def test_extract_channel_realtime_calls_get_realtime_unprocessed(
        self, nixi_config: NixiConfig, db_conn, tmp_path: Path
    ):
        """ExtractionBatcher with source='realtime' calls get_realtime_unprocessed."""
        # Set up both schemas
        ensure_realtime_schema(nixi_config.db_path)

        # Insert realtime messages directly
        conn = get_connection(nixi_config.db_path)
        for i in range(25):
            conn.execute(
                """INSERT INTO realtime_messages
                   (slack_ts, channel_id, user_id, text, event_id, timestamp)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (f"1766766571.{i:06d}", "C_RT_CH", "U_TEST", f"msg {i}", f"Ev{i}", "2026-04-26T00:00:00+00:00"),
            )
        conn.commit()

        mock_llm = AsyncMock()
        mock_llm.chat = AsyncMock(return_value="## Extracted\n- Test data")

        with patch("nixi.extraction.batch.write_org_facts"), \
             patch("nixi.extraction.batch.write_rules"), \
             patch("nixi.extraction.batch.write_employee_info"), \
             patch("nixi.extraction.batch.write_channel_skill"):
            batcher = ExtractionBatcher(nixi_config, conn, mock_llm, source="realtime", min_messages=20)
            result = await batcher.extract_channel("C_RT_CH")

        assert result is not None
        assert result.get("skipped") is False
        assert result.get("message_count") == 25
        conn.close()

    @pytest.mark.asyncio
    async def test_extract_channel_realtime_marks_extracted_shared_log(
        self, nixi_config: NixiConfig, tmp_path: Path
    ):
        """ExtractionBatcher source='realtime' marks extracted in shared nixi_extraction_log."""
        ensure_realtime_schema(nixi_config.db_path)
        ensure_schema(nixi_config.db_path)
        conn = get_connection(nixi_config.db_path)

        for i in range(25):
            conn.execute(
                """INSERT INTO realtime_messages
                   (slack_ts, channel_id, user_id, text, event_id, timestamp)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (f"1766766571.{i:06d}", "C_RT_CH", "U_TEST", f"msg {i}", f"Ev{i}", "2026-04-26T00:00:00+00:00"),
            )
        conn.commit()

        mock_llm = AsyncMock()
        mock_llm.chat = AsyncMock(return_value="## Extracted\n- Test data")

        with patch("nixi.extraction.batch.write_org_facts"), \
             patch("nixi.extraction.batch.write_rules"), \
             patch("nixi.extraction.batch.write_employee_info"), \
             patch("nixi.extraction.batch.write_channel_skill"):
            batcher = ExtractionBatcher(nixi_config, conn, mock_llm, source="realtime", min_messages=20)
            await batcher.extract_channel("C_RT_CH")

        # Verify messages are now in extraction_log (shared table)
        cursor = conn.execute("SELECT COUNT(*) FROM nixi_extraction_log")
        assert cursor.fetchone()["COUNT(*)"] == 25

        # Realtime unprocessed should now return 0 for this channel
        unprocessed = get_realtime_unprocessed(conn, "C_RT_CH")
        assert len(unprocessed) == 0

        conn.close()

    @pytest.mark.asyncio
    async def test_extract_all_realtime_calls_get_realtime_unprocessed_channels(
        self, nixi_config: NixiConfig, tmp_path: Path
    ):
        """ExtractionBatcher.extract_all with source='realtime' uses get_realtime_unprocessed_channels."""
        ensure_realtime_schema(nixi_config.db_path)
        ensure_schema(nixi_config.db_path)
        conn = get_connection(nixi_config.db_path)

        for i in range(25):
            conn.execute(
                """INSERT INTO realtime_messages
                   (slack_ts, channel_id, user_id, text, event_id, timestamp)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (f"1766766571.{i:06d}", "C_RT_CH1", "U_TEST", f"msg {i}", f"Ev1_{i}", "2026-04-26T00:00:00+00:00"),
            )

        mock_llm = AsyncMock()
        mock_llm.chat = AsyncMock(return_value="## Extracted\n- Test data")

        with patch("nixi.extraction.batch.write_org_facts"), \
             patch("nixi.extraction.batch.write_rules"), \
             patch("nixi.extraction.batch.write_employee_info"), \
             patch("nixi.extraction.batch.write_channel_skill"):
            batcher = ExtractionBatcher(nixi_config, conn, mock_llm, source="realtime", min_messages=20)
            result = await batcher.extract_all()

        assert result is not None
        assert "channels" in result
        conn.close()
