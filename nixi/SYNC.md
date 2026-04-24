# Nixi Upstream Sync Procedures

This document covers how to keep the nixi-agent fork synchronized with the upstream [Hermes Agent](https://github.com/NousResearch/hermes-agent) repository.

## Branch Strategy

| Branch | Purpose | Tracks |
|--------|---------|--------|
| `main` | Auto-merge target | upstream `main` (Hermes) |
| `nixi` | Nixi-specific development | Rebased onto `main` after each sync |
| Feature branches | Individual features | Branched from `nixi` |

**Remotes:**

| Remote | URL |
|--------|-----|
| `origin` | Nixi fork (`github.com/growgami/nixi`) |
| `upstream` | Hermes repository (`github.com/NousResearch/hermes-agent.git`) |

---

## Sync Cadence

- **Recommended:** Sync weekly, or when a significant upstream release lands
- Subscribe to upstream release notifications (GitHub releases or watch the repo)
- Always test before deploying synced changes to production tenant Machines

## Sync Procedure (Normal — No Conflicts)

```bash
# 1. Fetch latest upstream
git fetch upstream

# 2. Update local main to match upstream
git checkout main
git merge upstream/main --ff-only

# 3. Rebase nixi branch onto updated main
git checkout nixi
git rebase main

# 4. Run full test suite
scripts/run_tests.sh

# 5. If tests pass, push
git push origin nixi
```

## Sync Procedure (With Conflicts)

```bash
# 1-3 same as above

# 4. If rebase halts with conflicts:
#    git status  — see conflicted files
#    Resolve conflicts in each file

# 5. For conflicts in nixi/ package files:
#    Always keep our version — nixi/ is not upstream

# 6. For conflicts in core files that nixi modifies:
#    CAREFULLY merge — keep upstream's changes
#    and re-apply nixi's additions on top.
#    The nixi/ package uses extension points, so
#    conflicts should be limited to:
#    - gateway/config.py (Platform enum — additive)
#    - gateway/run.py (_create_adapter, _is_user_authorized — additive)
#    - gateway/platforms/slack.py (NIXI_MODE check — small conditional)
#    - toolsets.py (hermes-nixi entry — additive)
#    - agent/prompt_builder.py (PLATFORM_HINTS — additive)
#    - cron/scheduler.py (platform_map — additive)

# 7. After resolving:
git rebase --continue

# 8. Run full test suite
scripts/run_tests.sh
```

## Conflict Resolution Strategy by File Type

| File Type | Strategy | Reason |
|-----------|----------|--------|
| `gateway/config.py` (Platform enum) | Accept both — keep upstream additions + add NIXI | Enum additions are always additive, no conflict expected |
| `gateway/run.py` (_create_adapter) | Accept both — keep upstream additions + add NIXI elif | Factory pattern additions are always additive |
| `gateway/run.py` (_is_user_authorized) | Accept both — keep upstream additions + add NIXI bypass | Auth map additions are additive |
| `gateway/platforms/slack.py` (NIXI_MODE) | Keep nixi changes, re-apply if upstream refactors connect() | Small conditional at start of method, easy to re-apply |
| `toolsets.py` (hermes-nixi) | Accept both — keep upstream additions + add hermes-nixi | Toolset additions are always additive |
| `agent/prompt_builder.py` (PLATFORM_HINTS) | Accept both — keep upstream additions + add nixi | Dict entries are additive |
| `cron/scheduler.py` (platform_map) | Accept both — keep upstream additions + add nixi | Dict entries are additive |
| `nixi/` directory | Always keep nixi version | Not in upstream, never conflicts |
| Any other core file | Keep upstream, re-apply nixi patches | Core changes should be minimal by design |

## Regression Testing Checklist

After every upstream sync, verify:

- [ ] `Platform.NIXI` still in Platform enum
- [ ] `_create_adapter(Platform.NIXI)` returns NixiAdapter
- [ ] NIXI auth bypass works (bearer token validated, team_id checked)
- [ ] Slack send-only mode still works (NIXI_MODE=1)
- [ ] Employee overlay loading works
- [ ] Path validator blocks traversal
- [ ] Config seeding produces valid config.yaml
- [ ] Full Hermes test suite passes (agents other than nixi are unaffected)

## Hot Syncs (Urgent Upstream Fixes)

If upstream releases a critical security fix:

```bash
# Cherry-pick specific commits
git fetch upstream
git checkout nixi
git cherry-pick <commit-sha>

# Or create a patch
git format-patch -1 <commit-sha> --stdout | git am
```

**Warning:** Cherry-picks create duplicate commits on the next rebase. After a hot sync, the next normal sync should use `git rebase main` which will handle the duplicates.

## Upstream PR Strategy

- Consider contributing non-Nixi-specific improvements back upstream
- Examples: Platform adapter improvements, bug fixes in gateway/run.py, security improvements
- Never contribute Nixi-specific code (tenant isolation, employee overlays) upstream
- Tag PRs with `[nixi-fork]` prefix if forking context is useful