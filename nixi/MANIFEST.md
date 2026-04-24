# Nixi-Specific Files Manifest

Definitive list of all nixi-specific files and modifications. Use this as a checklist during sync regression testing.

## New Files (nixi-only, never in upstream)

These files live in the `nixi/` package or are nixi-specific test files. They will never appear in upstream and should never conflict during sync.

### Package Files
- `nixi/__init__.py`
- `nixi/__main__.py`
- `nixi/gateway_adapter.py`
- `nixi/employee_provider.py`
- `nixi/path_validator.py`
- `nixi/config_seeder.py`
- `nixi/seed_config.py`
- `nixi/deploy.py`
- `nixi/README.md`
- `nixi/SYNC.md`
- `nixi/MANIFEST.md`
- `nixi/check_sync.py`

### Test Files
- `tests/test_nixi_core.py`
- `tests/test_nixi_gateway_adapter.py`
- `tests/test_nixi_deploy.py`
- `tests/test_nixi_integration.py`
- `tests/gateway/test_config.py`
- `tests/gateway/test_slack.py`

## Modified Files (upstream + nixi patches)

For each file, the nixi patch is additive — it extends existing structures (enums, dicts, conditionals) without altering upstream behavior.

| File | Nixi Addition | Location | Conflict Risk |
|------|---------------|----------|---------------|
| `gateway/config.py` | `NIXI = "nixi"` Platform enum | Platform enum, after QQBOT | Low (additive) |
| `gateway/config.py` | NIXI env var overrides | `_apply_env_overrides()` | Low (additive block) |
| `gateway/run.py` | NIXI case in `_create_adapter()` | elif branch in adapter factory | Low (additive) |
| `gateway/run.py` | NIXI in `_is_user_authorized()` | bypass set | Low (additive) |
| `gateway/platforms/slack.py` | `NIXI_MODE` check in `connect()` | Early return at start of connect() | Medium (upstream refactors may touch connect()) |
| `toolsets.py` | `hermes-nixi` toolset | toolset dict + composite includes | Low (additive) |
| `agent/prompt_builder.py` | `nixi` PLATFORM_HINTS entry | PLATFORM_HINTS dict | Low (additive) |
| `cron/scheduler.py` | `nixi` in platform_map | platform_map dict | Low (additive) |

**Note on conflict risk:** All modifications are additive (adding enum members, dict entries, elif branches). The only Medium-risk file is `gateway/platforms/slack.py` because upstream may refactor the `connect()` method, requiring the NIXI_MODE conditional to be re-applied in a new location.