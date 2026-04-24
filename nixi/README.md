# Nixi-Agent — Multi-Tenant AI Agent for Hermes

Nixi turns a Hermes agent into a tenant-scoped Machine that receives Slack events from [Sludge](https://github.com/nickvdp/sludge), enriches them with per-employee context, and delivers replies through Slack — all while maintaining strict tenant isolation.

## Architecture

```
┌─────────┐    HTTP POST     ┌──────────────┐    gateway_runner    ┌────────────┐
│  Sludge  │ ──────────────► │ NixiAdapter   │ ──────────────────► │   Slack     │
│ (Slack    │   /nixi/event  │ (port 8080)    │    .send()          │   Adapter   │
│  gateway) │                │                │    .send_image()     │             │
└─────────┘                 └───────┬────────┘    .send_document()   └──────┬──────┘
                                    │                                  │
                                    │ channel_prompt                    │ xoxb- token
                                    │ (employee overlay)                │
                                    ▼                                  ▼
                            ┌──────────────┐                    ┌────────────┐
                            │  Agent Loop   │                    │  Slack API  │
                            │  (AIAgent)     │                    │             │
                            └──────────────┘                    └────────────┘
```

Inbound: Sludge authenticates at the adapter level (bearer token) and routes workspace events to `/nixi/event`. The adapter validates auth, extracts user info, loads employee overlays, and dispatches to the agent.

Outbound: NixiAdapter has **no direct Slack API client**. All outbound delivery (`send`, `send_image`, `send_document`) delegates to the Slack adapter via `gateway_runner`, following the same cross-platform pattern as WebhookAdapter.

## Quick Start — Deploy a New Tenant in 5 Commands

```bash
# 1. Set required environment variables
export NIXI_INTERNAL_SECRET="<shared-secret-between-sludge-and-nixi>"
export NIXI_TEAM_ID="T01XYZ567AB"
export SLACK_BOT_TOKEN="xoxb-xxxx-xxxx"
export HERMES_HOME="/data/tenants/acme"

# 2. Create the tenant home directory
mkdir -p "$HERMES_HOME"

# 3. Start nixi-agent
python -m nixi
```

That's it. On first start:

1. `validate_env()` checks all required env vars exist
2. `seed_if_needed()` writes `config.yaml`, `SOUL.md`, `AGENTS.md`, and directory structure into `$HERMES_HOME`
3. `NIXI_MODE=1` is set automatically (disables Socket Mode, enables send-only Slack)
4. The gateway starts — NixiAdapter listens on port 8080, Slack connects in send-only mode

Sludge must be configured to POST to `http://<host>:8080/nixi/event` with headers:

```
Authorization: Bearer <NIXI_INTERNAL_SECRET>
X-Nixi-Team-Id: T01XYZ567AB
X-Nixi-User-Id: U1234567890
X-Nixi-User-Name: jane.doe
```

## Environment Variables

### Required

| Variable | Description | Example |
|---|---|---|
| `NIXI_INTERNAL_SECRET` | Shared secret between Sludge and nixi-agent. Validated with `hmac.compare_digest` (constant-time). | `my-secret-key-123` |
| `NIXI_TEAM_ID` | Slack team/workspace ID this tenant serves. Rejects events with mismatched team ID (403). | `T01XYZ567AB` |
| `SLACK_BOT_TOKEN` | Slack bot token for outbound message delivery. Comma-separated for multi-workspace. | `xoxb-xxxx-xxxx` |
| `HERMES_HOME` | Path to tenant home directory. Path validation jails all file ops here. | `/data/tenants/acme` |

### Optional

| Variable | Description | Default | Example |
|---|---|---|---|
| `NIXI_PORT` | HTTP listen port for the NixiAdapter server. | `8080` | `8081` |
| `NIXI_MODE` | Set automatically by `start_nixi()`. Disables Socket Mode, enables send-only Slack. | (auto) | `1` |
| `NIXI_ALLOWED_USERS` | Comma-separated Slack user IDs for future per-employee restriction. Currently mapped but not enforced — all workspace members allowed. | (all) | `U123,U456` |
| `NIXI_COMPANY_NAME` | Display name written to seeded `SOUL.md`. | `Tenant` | `Acme Corp` |
| `HERMES_MODEL_PROVIDER` | LLM provider for seeded config. | `openai` | `anthropic` |
| `HERMES_MODEL` | LLM model slug for seeded config. | `gpt-4o` | `claude-sonnet-4-20250514` |

> **Note:** `SLACK_APP_TOKEN` is **not required** for nixi deployments. `NIXI_MODE=1` disables Socket Mode entirely — the Slack adapter operates in send-only mode using `SLACK_BOT_TOKEN` alone.

## Directory Structure

Each tenant gets an isolated `HERMES_HOME` directory:

```
HERMES_HOME/                          # /data/tenants/{company_id}/
├── config.yaml                        # Seeded config (see below)
├── SOUL.md                            # Company personality prompt
├── AGENTS.md                          # Agent behavior configuration
├── employees/                         # Per-employee overlay files
│   └── {user_id}/
│       └── USER.md                    # Employee context (ephemeral injection)
├── skills/                            # Agent skills
│   ├── seeded/
│   ├── channel/
│   ├── event/
│   ├── learned/
│   └── archive/
├── sessions/                          # Conversation history (FTS5)
└── logs/                              # Agent and gateway logs
```

Path traversal protection: every file operation in the nixi package goes through `safe_path()` from `nixi.path_validator`. It resolves symlinks and verifies the result stays within `HERMES_HOME`. This prevents malicious `user_id` values like `../../etc/passwd` from escaping the tenant directory.

## Seeded config.yaml

When `HERMES_HOME/config.yaml` doesn't exist, `seed_if_needed()` generates one:

```yaml
_config_version: <dynamic>   # Read from DEFAULT_CONFIG at seed time
model: gpt-4o
gateway:
  slack:
    enabled: true
    workspace_id: T01XYZ567AB
  nixi:
    enabled: true
memory:
  scope: organization
terminal:
  backend: local
  timeout: 180
```

Key points:

- **`_config_version` is dynamic** — read from `hermes_cli.config.DEFAULT_CONFIG` at seed time, not hardcoded. This avoids triggering the setup wizard or migration flow on every Hermes update.
- **`gateway.slack.workspace_id`** — set from `NIXI_TEAM_ID`.
- **`gateway.nixi.enabled: true`** — enables the Nixi adapter.
- **`memory.scope: organization`** — memories shared across all employees in the tenant.
- Seeding completely **bypasses the setup wizard** by writing a valid `config.yaml` directly.

## Employee Overlay System

Each employee can have a context overlay stored at:

```
HERMES_HOME/employees/{user_id}/USER.md
```

When a message arrives from Sludge with `X-Nixi-User-Id: U123456`, the adapter:

1. Calls `load_overlay("U123456")`
2. Reads `HERMES_HOME/employees/U123456/USER.md` (if it exists)
3. Passes the content as `channel_prompt` on the `MessageEvent`

**`channel_prompt` is ephemeral** — it's injected into the conversation context for that turn only. It is **not** stored in the FTS5 session index. This means:

- Overlay changes take effect immediately (no cache invalidation)
- Overlay content never pollutes persistent conversation search
- Different employees can have different context without cross-contamination

> On first interaction, the overlay file won't exist yet. `load_overlay()` returns an empty string and logs a debug message. The agent or a skill can create the file later via `get_or_create_employee_dir()`.

## Path Validator and Tenant Isolation

The nixi package enforces tenant isolation at the filesystem level:

```python
from nixi.path_validator import safe_path, validate_hermes_home

# Every file operation MUST go through safe_path()
overlay_path = safe_path(home, f"employees/{user_id}/USER.md")
# → resolves symlinks, checks result stays within home

# Startup validation
home = validate_hermes_home()  # Raises if HERMES_HOME unset or missing
```

Guarantees:

1. **No path traversal** — `safe_path("../../../etc/passwd")` raises `PathTraversalError`
2. **No symlink escape** — resolves symlinks before checking boundaries
3. **All nixi file ops jailed** — employee overlays, config reads, skill writes
4. **Tenant-scoped HERMES_HOME** — each container/Machine maps its own isolated directory

## Cross-Platform Delivery via gateway_runner

NixiAdapter doesn't implement its own Slack client. Instead, it delegates all outbound delivery:

```python
class NixiAdapter(BasePlatformAdapter):
    async def send(self, chat_id, content, reply_to=None, metadata=None):
        slack_adapter = self.gateway_runner.adapters.get(Platform.SLACK)
        return await slack_adapter.send(chat_id, content, reply_to=reply_to, metadata=metadata)
```

This follows the same pattern as `WebhookAdapter`. The `gateway_runner` reference is set by `_create_adapter()` in `gateway/run.py`:

```python
elif platform == Platform.NIXI:
    adapter = NixiAdapter(config)
    adapter.gateway_runner = self  # Cross-platform delivery
    return adapter
```

When running in `NIXI_MODE=1`, the Slack adapter's `connect()` method skips Socket Mode entirely. It initializes `_primary_client` as an `AsyncWebClient` for API calls only — no event listener, no socket handler. Messages flow: Sludge → HTTP POST → NixiAdapter → Agent → gateway_runner → Slack WebClient → Slack API.

## Auth Model

Auth operates at two levels:

### 1. Adapter Level (Bearer Token)

Sludge authenticates to nixi-agent using a shared secret in the `Authorization` header:

```
Authorization: Bearer <NIXI_INTERNAL_SECRET>
```

Validated with `hmac.compare_digest()` (constant-time comparison to prevent timing attacks). Requests without a valid token receive `401 Unauthorized`.

### 2. Workspace Membership (Sludge)

Sludge enforces workspace membership — only messages from authorized Slack workspaces reach nixi-agent. The `X-Nixi-Team-Id` header is validated against `NIXI_TEAM_ID`:

- Match → process the event
- Mismatch → `403 Team ID mismatch`

### 3. Per-Employee Restriction (Future)

`NIXI_ALLOWED_USERS` is mapped in the platform env map alongside other platforms' allowlists, but **not currently enforced** for nixi. All workspace members are allowed by the bypass at `_is_user_authorized()`:

```python
if source.platform in (Platform.HOMEASSISTANT, Platform.WEBHOOK, Platform.NIXI):
    return True  # Auth handled at adapter + Sludge level
```

To enforce per-employee restriction in the future, populate `NIXI_ALLOWED_USERS` with comma-separated Slack user IDs. The env var infrastructure is already wired — only the enforcement check needs activation.

## Slack Send-Only Mode (`NIXI_MODE=1`)

When deployed as a nixi tenant Machine, the Slack adapter operates in send-only mode:

- **Socket Mode is disabled** — no `SLACK_APP_TOKEN` needed
- **No event listener** — `AsyncApp` and `AsyncSocketModeHandler` are not created
- **`_primary_client`** — single `AsyncWebClient` for outbound API calls
- **Multi-workspace** — comma-separated `SLACK_BOT_TOKEN` still works; all tokens are authenticated
- **`send()` works normally** — `_get_client()` falls back to `_primary_client` when `_app` is None

This mode is automatically activated by `deploy.start_nixi()`, which sets `NIXI_MODE=1` in the environment **before** importing gateway modules (ordering matters — modules check this env var at `connect()` time).

## Test Files

| File | Scope |
|---|---|
| `tests/test_nixi_core.py` | Unit tests: `path_validator`, `employee_provider`, `seed_config`, `config_seeder` |
| `tests/test_nixi_gateway_adapter.py` | Unit tests: NixiAdapter HTTP handlers, auth, dispatch, send delegation |
| `tests/test_nixi_deploy.py` | Unit tests: `deploy` module (env validation, config seeding, gateway startup) |
| `tests/gateway/test_slack.py` | Slack NIXI_MODE tests: send-only mode, `_primary_client` fallback |
| `tests/test_nixi_integration.py` | Cross-component tests: Sludge → NixiAdapter → Agent → Slack reply flow |

## Upstream Sync

For keeping the nixi-agent fork synchronized with upstream Hermes, see the sync procedures document:

→ [Upstream Sync Procedures](../docs/plans/2026-04-24-upstream-sync-procedures.md)

This covers adding the upstream remote, merge cadence, conflict resolution strategies per file, and regression testing checklists.

## Nixi Package Files

| File | Purpose |
|---|---|
| `nixi/__init__.py` | Package marker, version |
| `nixi/__main__.py` | `python -m nixi` entry point |
| `nixi/gateway_adapter.py` | NixiAdapter — HTTP server, auth, event dispatch, cross-platform send |
| `nixi/employee_provider.py` | Employee overlay loader (reads `USER.md` files) |
| `nixi/path_validator.py` | `safe_path()` jail validator, `validate_hermes_home()` |
| `nixi/config_seeder.py` | `seed_hermes_home()` — writes config.yaml, SOUL.md, AGENTS.md, directories |
| `nixi/seed_config.py` | `generate_seed_config()` — builds config.yaml dict with dynamic `_config_version` |
| `nixi/deploy.py` | `start_nixi()` — main entry point: validate env, seed config, set NIXI_MODE, start gateway |

## Modified Upstream Files

The nixi package extends Hermes without forking core files. Merge conflicts are minimal and always additive:

| File | Nixi Addition | Conflict Risk |
|---|---|---|
| `gateway/config.py` | `NIXI = "nixi"` in Platform enum, NIXI env var overrides | Low (additive) |
| `gateway/run.py` | `Platform.NIXI` case in `_create_adapter()`, `NIXI_ALLOWED_USERS` in env map, NIXI in auth bypass | Low (additive) |
| `gateway/platforms/slack.py` | `NIXI_MODE` check in `connect()`, `_connect_nixi_mode()`, `_primary_client` fallback | Medium (method refactors) |
| `toolsets.py` | `hermes-nixi` toolset entry | Low (additive) |
| `agent/prompt_builder.py` | `nixi` PLATFORM_HINTS entry | Low (additive) |
| `cron/scheduler.py` | `nixi` in platform_map | Low (additive) |