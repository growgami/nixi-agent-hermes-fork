"""Tests for NIXI_ALLOWED_USERS enforcement.

Verifies that:
- NIXI_ALLOWED_USERS restricts access when NIXI_ALLOW_ALL_USERS=false
- NIXI_ALLOW_ALL_USERS=true (default) preserves open-access behavior
- Removing the Platform.NIXI bypass from _is_user_authorized() works correctly
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.session import SessionSource


# ─── Helpers ────────────────────────────────────────────────────────────────


def _clear_auth_env(monkeypatch) -> None:
    """Remove all auth-related env vars to start from a clean state."""
    for key in (
        "NIXI_ALLOWED_USERS",
        "NIXI_ALLOW_ALL_USERS",
        "GATEWAY_ALLOWED_USERS",
        "GATEWAY_ALLOW_ALL_USERS",
        # Clear other platform envs too to prevent cross-contamination
        "TELEGRAM_ALLOWED_USERS", "TELEGRAM_ALLOW_ALL_USERS",
        "DISCORD_ALLOWED_USERS", "DISCORD_ALLOW_ALL_USERS",
        "WHATSAPP_ALLOWED_USERS", "WHATSAPP_ALLOW_ALL_USERS",
        "SLACK_ALLOWED_USERS", "SLACK_ALLOW_ALL_USERS",
        "SIGNAL_ALLOWED_USERS", "SIGNAL_ALLOW_ALL_USERS",
        "EMAIL_ALLOWED_USERS", "EMAIL_ALLOW_ALL_USERS",
        "SMS_ALLOWED_USERS", "SMS_ALLOW_ALL_USERS",
        "MATTERMOST_ALLOWED_USERS", "MATTERMOST_ALLOW_ALL_USERS",
        "MATRIX_ALLOWED_USERS", "MATRIX_ALLOW_ALL_USERS",
        "DINGTALK_ALLOWED_USERS", "DINGTALK_ALLOW_ALL_USERS",
        "FEISHU_ALLOWED_USERS", "FEISHU_ALLOW_ALL_USERS",
        "WECOM_ALLOWED_USERS", "WECOM_ALLOW_ALL_USERS",
        "WECOM_CALLBACK_ALLOWED_USERS", "WECOM_CALLBACK_ALLOW_ALL_USERS",
        "WEIXIN_ALLOWED_USERS", "WEIXIN_ALLOW_ALL_USERS",
        "BLUEBUBBLES_ALLOWED_USERS", "BLUEBUBBLES_ALLOW_ALL_USERS",
        "QQ_ALLOWED_USERS", "QQ_ALLOW_ALL_USERS",
        "QQ_GROUP_ALLOWED_USERS",
    ):
        monkeypatch.delenv(key, raising=False)


def _make_nixi_runner(monkeypatch, tmp_path):
    """Create a GatewayRunner suitable for testing _is_user_authorized."""
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.NIXI: PlatformConfig(enabled=True)}
    )
    adapter = SimpleNamespace(send=AsyncMock())
    runner.adapters = {Platform.NIXI: adapter}
    runner.pairing_store = MagicMock()
    runner.pairing_store.is_approved.return_value = False
    runner.pairing_store._is_rate_limited.return_value = False
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._update_prompts = {}
    runner.hooks = SimpleNamespace(dispatch=AsyncMock(return_value=None))
    runner._sessions = {}
    return runner


def _nixi_source(user_id: str, chat_type: str = "dm") -> SessionSource:
    """Create a SessionSource for a NIXI user."""
    return SessionSource(
        platform=Platform.NIXI,
        user_id=user_id,
        chat_id=f"nixi-{user_id}",
        user_name=f"user_{user_id}",
        chat_type=chat_type,
    )


# ─── Test: NIXI_ALLOWED_USERS permit ────────────────────────────────────────


def test_nixi_allowed_users_permit(monkeypatch, tmp_path):
    """NIXI_ALLOWED_USERS=user123 should authorize user123."""
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv("NIXI_ALLOWED_USERS", "user123")
    monkeypatch.setenv("NIXI_ALLOW_ALL_USERS", "false")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    runner = _make_nixi_runner(monkeypatch, tmp_path)
    source = _nixi_source("user123")

    assert runner._is_user_authorized(source) is True


# ─── Test: NIXI_ALLOWED_USERS deny ──────────────────────────────────────────


def test_nixi_allowed_users_deny(monkeypatch, tmp_path):
    """NIXI_ALLOWED_USERS=user123 should deny user456."""
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv("NIXI_ALLOWED_USERS", "user123")
    monkeypatch.setenv("NIXI_ALLOW_ALL_USERS", "false")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    runner = _make_nixi_runner(monkeypatch, tmp_path)
    source = _nixi_source("user456")

    assert runner._is_user_authorized(source) is False


# ─── Test: NIXI_ALLOW_ALL_USERS default (no env vars) ──────────────────────


def test_nixi_allow_all_users_default(monkeypatch, tmp_path):
    """With no env vars set, NIXI users should be authorized (backward compat).

    NIXI_ALLOW_ALL_USERS defaults to 'true' in platform_allow_all_map,
    so all NIXI users remain authorized without any config change.
    """
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    runner = _make_nixi_runner(monkeypatch, tmp_path)
    source = _nixi_source("any_nixi_user")

    assert runner._is_user_authorized(source) is True


# ─── Test: NIXI_ALLOW_ALL_USERS explicit true ──────────────────────────────


def test_nixi_allow_all_users_explicit_true(monkeypatch, tmp_path):
    """NIXI_ALLOW_ALL_USERS=true should authorize any NIXI user."""
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv("NIXI_ALLOW_ALL_USERS", "true")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    runner = _make_nixi_runner(monkeypatch, tmp_path)
    source = _nixi_source("any_nixi_user")

    assert runner._is_user_authorized(source) is True


# ─── Test: NIXI_ALLOW_ALL_USERS=false with allowlist ────────────────────────


def test_nixi_allow_all_users_false_with_allowlist(monkeypatch, tmp_path):
    """NIXI_ALLOW_ALL_USERS=false + NIXI_ALLOWED_USERS=user123 authorizes only user123."""
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv("NIXI_ALLOW_ALL_USERS", "false")
    monkeypatch.setenv("NIXI_ALLOWED_USERS", "user123")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    runner = _make_nixi_runner(monkeypatch, tmp_path)

    allowed_source = _nixi_source("user123")
    denied_source = _nixi_source("user456")

    assert runner._is_user_authorized(allowed_source) is True
    assert runner._is_user_authorized(denied_source) is False


# ─── Test: NIXI no allowlist defaults to gateway allow-all ──────────────────


def test_nixi_no_allowlist_defaults_to_gateway_allow_all(monkeypatch, tmp_path):
    """When NIXI_ALLOW_ALL_USERS is unset (defaults true), NIXI users are authorized
    regardless of GATEWAY_ALLOW_ALL_USERS."""
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv("GATEWAY_ALLOW_ALL_USERS", "false")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    runner = _make_nixi_runner(monkeypatch, tmp_path)
    source = _nixi_source("any_nixi_user")

    # NIXI_ALLOW_ALL_USERS defaults to "true", so authorization passes
    # even when GATEWAY_ALLOW_ALL_USERS is explicitly false
    assert runner._is_user_authorized(source) is True