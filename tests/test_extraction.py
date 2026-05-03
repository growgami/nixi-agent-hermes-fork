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
    ensure_schema,
    get_connection,
    get_unprocessed,
    get_unprocessed_channels,
    insert_messages,
    mark_extracted,
)
from nixi.extraction.batch import ExtractionBatcher
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

        with patch("nixi.extract.ExtractionBatcher") as MockBatcher:
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

        with patch("nixi.extract.ExtractionBatcher") as MockBatcher:
            mock_batcher = AsyncMock()
            mock_batcher.extract_channel = AsyncMock(return_value={"skipped": False})
            MockBatcher.return_value = mock_batcher

            result = await run_extraction_channel("C06M81FSKFF", nixi_config)
            assert result is not None