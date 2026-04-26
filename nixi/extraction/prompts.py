"""LLM extraction prompt templates for the nixi extraction pipeline.

Each prompt includes format instructions, character limits (from config),
and extraction guidelines. Prompts are parameterizable via str.format() for
per-extraction configuration.
"""

from __future__ import annotations

ORG_FACTS_PROMPT = """\
You are analyzing Slack channel messages to extract organizational memory.

Extract organizational facts, conventions, and patterns from the messages below.
Focus on:
- Team structures and reporting relationships
- Recurring processes and workflows
- Technical conventions (naming, coding, deployment)
- Key decisions and their context
- Important dates, deadlines, and milestones
- Cultural norms and communication patterns

Output format: Write structured sections in Markdown.
- Use ## headers for major categories
- Use bullet points for individual facts
- Be concise but specific — extract facts, not opinions
- Total output MUST NOT exceed {memory_limit} characters

Messages:
{messages}
"""

RULES_PROMPT = """\
You are analyzing Slack channel messages to extract company rules, policies, and dos/don'ts.

Extract actionable rules that an AI agent should follow when interacting with this organization.
Focus on:
- Explicit policies (security, deployment, communication)
- Implicit conventions (how things are done here)
- Dos and don'ts (what works, what doesn't)
- Boundary conditions (when to escalate, when to act autonomously)

Output format: Write as a Markdown bullet list. Each rule should be:
- Concise and actionable
- Prefixed with a category tag like [POLICY], [CONVENTION], [DANGER], etc.
- Appropriate for appending to an AGENTS.md file

IMPORTANT: These are NEW rules to append. Do NOT overwrite or repeat existing rules.
Output MUST NOT exceed {memory_limit} characters.

Messages:
{messages}
"""

EMPLOYEE_PROMPT = """\
You are analyzing Slack channel messages to extract per-employee information.

For each distinct person you can identify in the messages, extract:
- Role and responsibilities
- Technical expertise and specializations
- Communication style and preferences
- Working patterns (timezone, availability patterns)
- Key relationships and collaboration patterns
- Notable contributions or decisions

Output format: JSON array of objects, each with:
  "display_name": string — the person's display name from messages
  "user_id": string or null — Slack user ID if identifiable, null otherwise
  "info": string — concise summary of this person (NOT a biography — focus on what an AI agent needs to know)

Each employee's "info" field MUST NOT exceed {employee_limit} characters.
If an employee already has a USER.md file (mentioned in existing info), MERGE new insights rather than replacing.

Messages:
{messages}

Existing employee data (merge, don't overwrite):
{existing_employees}
"""

CHANNEL_SKILL_PROMPT = """\
You are analyzing Slack channel messages to identify repeated behavioral patterns
that could be codified as a "skill" — a reusable procedure that an AI agent can follow.

Look for patterns like:
- Repeated question-answer sequences
- Common troubleshooting workflows
- Recurring deployment or release procedures
- Regular status update or review patterns
- Onboarding or orientation sequences

For each pattern you identify, provide:
1. "skill_name": A short kebab-case name (e.g., "deploy-checks", "pr-review-flow")
2. "triggers": List of keywords/phrases that would activate this skill
3. "procedure": Step-by-step procedure in numbered Markdown
4. "pitfalls": What to watch out for or avoid

Output format: JSON array of skill objects.

Additionally, for each skill, create a `references/channel-context.md` file that contains
SQL queries for nixi_state.db to retrieve relevant channel context at runtime. The SQL queries
should:
- Query scraped_messages for the relevant channel
- Filter by time range and keywords related to the skill
- Join with nixi_extraction_log to avoid re-extracting

Messages:
{messages}
"""