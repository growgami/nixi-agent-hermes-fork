"""Batch orchestration for the nixi extraction pipeline.

Groups unprocessed messages by channel, formats them for LLM extraction,
and writes results to the appropriate output files.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent.auxiliary_client import resolve_provider_client
from nixi.config import NixiConfig
from nixi.db import (
    build_user_map,
    get_realtime_unprocessed,
    get_realtime_unprocessed_channels,
    get_unprocessed,
    get_unprocessed_channels,
    mark_extracted,
)
from nixi.extraction.prompts import (
    CHANNEL_SKILL_PROMPT,
    EMPLOYEE_PROMPT,
    ORG_FACTS_PROMPT,
    RULES_PROMPT,
)
from nixi.extraction.writers import (
    write_channel_skill,
    write_employee_info,
    write_org_facts,
    write_rules,
)
from nixi.models import UserMap

logger = logging.getLogger(__name__)


class LLMClient:
    """Thin wrapper for LLM API calls.

    Uses the same provider/model as configured in hermes config
    (extraction_model in NixiConfig). Falls back to env-based provider.
    """

    def __init__(self, config: NixiConfig) -> None:
        if not config.extraction_model:
            raise RuntimeError(
                "No extraction model configured. "
                "Set extraction_model in config.yaml or NIXI_MODEL environment variable."
            )

        client, resolved_model = resolve_provider_client(
            "auto", model=config.extraction_model, async_mode=True
        )
        if client is None or resolved_model is None:
            raise RuntimeError(
                "No LLM provider configured. "
                "Set an API key in .env (e.g. OPENAI_API_KEY, OPENROUTER_API_KEY) "
                f"or configure a provider via hermes model. "
                f"HERMES_HOME: {os.environ.get('HERMES_HOME', 'not set')}"
            )

        self._client = client
        self._resolved_model = resolved_model

    async def chat(self, prompt: str) -> str:
        """Send a prompt to the LLM and return the response text.

        Args:
            prompt: Full prompt string to send.

        Returns:
            LLM response text.
        """
        messages = [{"role": "user", "content": prompt}]
        response = await self._client.chat.completions.create(
            model=self._resolved_model, messages=messages
        )
        return response.choices[0].message.content


class ExtractionBatcher:
    """Orchestrates batch extraction of unprocessed Slack messages.

    Groups messages by channel, feeds them to the LLM with appropriate
    prompts, and writes extracted organizational memory to output files.
    """

    def __init__(
        self,
        config: NixiConfig,
        conn: Any,
        llm: Any,
        min_messages: int = 20,
        source: str = "scraped",
    ) -> None:
        self.config = config
        self.conn = conn
        self.llm = llm
        self.min_messages = min_messages
        self.output_dir = config.output_dir
        self.hermes_home = config.hermes_home
        if source not in ("scraped", "realtime"):
            raise ValueError(f"source must be 'scraped' or 'realtime', got '{source}'")
        self.source = source

    def _format_messages_for_prompt(
        self,
        messages: list[dict[str, Any]],
        channel_type: str | None = None,
    ) -> str:
        """Format messages for LLM prompt, prefixing bot messages with [BOT].

        Messages are sorted by timestamp (oldest first) and formatted as:
            [BOT] @name: text  (for bot messages)
            @name: text        (for regular messages)

        When channel_type is provided (realtime source), a header line
        is prepended with the channel context (e.g. "[Channel type: group]").

        For realtime messages without user_name, user_id is used as fallback.

        Args:
            messages: List of message dicts from get_unprocessed() or
                get_realtime_unprocessed().
            channel_type: Optional channel type context (e.g. "channel",
                "group", "im", "mpim") from realtime_messages.

        Returns:
            Formatted string for LLM prompt.
        """
        # Sort by timestamp ASC (oldest first)
        sorted_msgs = sorted(messages, key=lambda m: m.get("timestamp", ""))

        lines: list[str] = []
        if channel_type:
            lines.append(f"[Channel type: {channel_type}]")

        for msg in sorted_msgs:
            # Realtime messages may not have user_name; fall back to user_id
            name = msg.get("user_name") or msg.get("user_id") or "unknown"
            text = msg.get("text", "")
            is_bot = bool(msg.get("is_bot", 0))

            prefix = "[BOT] " if is_bot else ""
            lines.append(f"{prefix}@{name}: {text}")

        return "\n".join(lines)

    async def extract_channel(self, channel_id: str) -> dict[str, Any]:
        """Extract organizational memory from a single channel.

        Steps:
        1. Get unprocessed messages for this channel
        2. If count < threshold, skip
        3. Format and feed to LLM with each prompt
        4. Write results via writer functions
        5. Mark messages as extracted

        Args:
            channel_id: Slack channel ID to extract.

        Returns:
            Dict with extraction results (channel_id, message_count, skipped, etc.)
        """
        if self.source == "realtime":
            messages = get_realtime_unprocessed(
                self.conn, channel_id, limit=self.config.extraction_batch_size
            )
        else:
            messages = get_unprocessed(
                self.conn, channel_id, limit=self.config.extraction_batch_size
            )

        # Skip channels with insufficient signal
        if len(messages) < self.min_messages:
            logger.info(
                "Skipping channel %s: %d messages below threshold %d",
                channel_id,
                len(messages),
                self.min_messages,
            )
            return {
                "channel_id": channel_id,
                "message_count": len(messages),
                "skipped": True,
                "reason": f"Below threshold ({len(messages)} < {self.min_messages})",
            }

        # Build user map for employee resolution
        user_map = build_user_map(self.conn, self.config.cooccurrence_threshold)

        # Format messages for prompt (include channel_type context for realtime source)
        channel_type = None
        if self.source == "realtime" and messages:
            channel_type = messages[0].get("channel_type")
        formatted = self._format_messages_for_prompt(messages, channel_type=channel_type)
        batch_id = str(uuid.uuid4())[:8]
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        extraction_results: dict[str, Any] = {
            "channel_id": channel_id,
            "message_count": len(messages),
            "skipped": False,
            "batch_id": batch_id,
        }

        # 1. Org facts extraction
        try:
            org_facts_prompt = ORG_FACTS_PROMPT.format(
                messages=formatted,
                memory_limit=self.config.memory_limit,
            )
            org_facts_response = await self.llm.chat(org_facts_prompt)
            write_org_facts(org_facts_response, self.hermes_home, self.config.memory_limit)
            extraction_results["org_facts"] = True
        except Exception as e:
            logger.error("Org facts extraction failed for %s: %s", channel_id, e)
            extraction_results["org_facts"] = False

        # 2. Rules extraction
        try:
            rules_prompt = RULES_PROMPT.format(
                messages=formatted,
                memory_limit=self.config.memory_limit,
            )
            rules_response = await self.llm.chat(rules_prompt)
            write_rules(rules_response, self.hermes_home, self.config.rules_limit)
            extraction_results["rules"] = True
        except Exception as e:
            logger.error("Rules extraction failed for %s: %s", channel_id, e)
            extraction_results["rules"] = False

        # 3. Employee extraction
        try:
            # Collect existing employee data for merge context
            existing_employees = self._collect_existing_employees()
            employee_prompt = EMPLOYEE_PROMPT.format(
                messages=formatted,
                employee_limit=self.config.employee_limit,
                existing_employees=existing_employees,
            )
            employee_response = await self.llm.chat(employee_prompt)
            employees = self._parse_employee_response(employee_response)
            write_employee_info(
                employees, self.hermes_home, user_map, self.config.employee_limit
            )
            extraction_results["employees"] = len(employees)
        except Exception as e:
            logger.error("Employee extraction failed for %s: %s", channel_id, e)
            extraction_results["employees"] = 0

        # 4. Channel skill extraction
        try:
            skill_prompt = CHANNEL_SKILL_PROMPT.format(messages=formatted)
            skill_response = await self.llm.chat(skill_prompt)
            skills = self._parse_skill_response(skill_response)
            for skill in skills:
                write_channel_skill(skill, channel_id, date_str, self.hermes_home)
            extraction_results["skills"] = len(skills)
        except Exception as e:
            logger.error("Channel skill extraction failed for %s: %s", channel_id, e)
            extraction_results["skills"] = 0

        # Mark all processed messages in extraction log
        slack_ts_list = [m["slack_ts"] for m in messages]
        mark_extracted(self.conn, channel_id, slack_ts_list, batch_id)

        logger.info(
            "Extracted channel %s: %d messages, batch %s",
            channel_id,
            len(messages),
            batch_id,
        )
        return extraction_results

    async def extract_all(self) -> dict[str, Any]:
        """Extract all channels with unprocessed messages.

        Returns:
            Dict with per-channel results and aggregate summary.
        """
        if self.source == "realtime":
            channels = get_realtime_unprocessed_channels(self.conn)
        else:
            channels = get_unprocessed_channels(self.conn)
        logger.info("Found %d channels with unprocessed messages", len(channels))

        results: dict[str, Any] = {
            "channels": {},
            "total_messages": 0,
            "total_skipped": 0,
            "total_extracted": 0,
        }

        for channel_id in channels:
            channel_result = await self.extract_channel(channel_id)
            results["channels"][channel_id] = channel_result

            msg_count = channel_result.get("message_count", 0)
            results["total_messages"] += msg_count

            if channel_result.get("skipped", False):
                results["total_skipped"] += 1
            else:
                results["total_extracted"] += 1

        logger.info(
            "Extraction complete: %d channels, %d extracted, %d skipped",
            len(channels),
            results["total_extracted"],
            results["total_skipped"],
        )
        return results

    def _collect_existing_employees(self) -> str:
        """Collect existing employee USER.md content for merge context.

        Returns:
            JSON string of existing employee data, or empty string.
        """
        employees_dir = self.hermes_home / "employees"
        if not employees_dir.is_dir():
            return "[]"

        existing: list[dict[str, str]] = []
        for emp_dir in sorted(employees_dir.iterdir()):
            if not emp_dir.is_dir() or emp_dir.name.endswith(".archived"):
                continue
            user_file = emp_dir / "USER.md"
            if user_file.exists():
                content = user_file.read_text(encoding="utf-8")
                # Truncate for prompt context limits
                if len(content) > 500:
                    content = content[:500] + "..."
                existing.append({
                    "directory": emp_dir.name,
                    "content": content,
                })

        return json.dumps(existing, indent=2) if existing else "[]"

    def _parse_employee_response(self, response: str) -> list[dict]:
        """Parse LLM response into employee data list.

        Tries JSON parse first, falls back to structured text extraction.

        Args:
            response: Raw LLM response text.

        Returns:
            List of employee dicts with display_name, user_id, info.
        """
        # Try JSON first
        try:
            employees = json.loads(response)
            if isinstance(employees, list):
                return employees
        except json.JSONDecodeError:
            pass

        # Fallback: extract from structured text
        # Look for lines that might be employee descriptions
        employees: list[dict] = []
        current_name = None
        current_info: list[str] = []

        for line in response.splitlines():
            line = line.strip()
            if not line:
                continue

            # Pattern: "## Name" or "**Name**" or "- Name:"
            if line.startswith("## ") or line.startswith("**"):
                # Save previous employee
                if current_name and current_info:
                    employees.append({
                        "display_name": current_name,
                        "user_id": None,
                        "info": "\n".join(current_info),
                    })
                current_name = line.lstrip("#* ").rstrip("#* ")
                current_info = []
            elif line.startswith("- ") and current_name:
                current_info.append(line.lstrip("- "))

        # Save last employee
        if current_name and current_info:
            employees.append({
                "display_name": current_name,
                "user_id": None,
                "info": "\n".join(current_info),
            })

        return employees

    def _parse_skill_response(self, response: str) -> list[dict]:
        """Parse LLM response into skill data list.

        Tries JSON parse first, falls back to structured text extraction.

        Args:
            response: Raw LLM response text.

        Returns:
            List of skill dicts with skill_name, triggers, procedure, pitfalls.
        """
        # Try JSON first
        try:
            skills = json.loads(response)
            if isinstance(skills, list):
                return skills
        except json.JSONDecodeError:
            pass

        # Fallback: single skill from text
        skill_name = "extracted-pattern"
        triggers: list[str] = []
        procedure_lines: list[str] = []
        pitfalls_lines: list[str] = []
        current_section = None

        for line in response.splitlines():
            line = line.strip()
            if not line:
                continue

            lower = line.lower()
            if "skill_name" in lower or "name:" in lower:
                skill_name = line.split(":", 1)[-1].strip().strip('"')
            elif "trigger" in lower:
                current_section = "triggers"
            elif "procedure" in lower or "step" in lower:
                current_section = "procedure"
            elif "pitfall" in lower or "avoid" in lower:
                current_section = "pitfalls"
            elif current_section == "triggers" and line.startswith("- "):
                triggers.append(line.lstrip("- "))
            elif current_section == "procedure" and line.startswith(("-", "1.", "2.", "3.")):
                procedure_lines.append(line.lstrip("- ").lstrip("0123456789. "))
            elif current_section == "pitfalls" and line.startswith("- "):
                pitfalls_lines.append(line.lstrip("- "))

        return [{
            "skill_name": skill_name,
            "triggers": triggers or ["general"],
            "procedure": "\n".join(procedure_lines) or "No procedure extracted",
            "pitfalls": "\n".join(pitfalls_lines) or "None identified",
        }]