"""Tests for nixi.deploy — config seeding and gateway startup."""

import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml


# ─── validate_env tests ─────────────────────────────────────────────────


class TestValidateEnv:
    """Tests for start_nixi environment validation."""

    def test_missing_nixi_internal_secret_raises(self, tmp_path):
        """NIXI_INTERNAL_SECRET must be set."""
        from nixi.deploy import validate_env

        home = tmp_path / "tenant"
        home.mkdir()
        env = {
            "NIXI_TEAM_ID": "T_TEST",
            "SLACK_BOT_TOKEN": "xoxb-test",
            "HERMES_HOME": str(home),
            # NIXI_INTERNAL_SECRET intentionally omitted
        }
        with patch.dict(os.environ, env, clear=True):
            os.environ.pop("NIXI_INTERNAL_SECRET", None)
            with pytest.raises(EnvironmentError, match="NIXI_INTERNAL_SECRET"):
                validate_env()

    def test_missing_team_id_raises(self, tmp_path):
        """NIXI_TEAM_ID must be set."""
        from nixi.deploy import validate_env

        home = tmp_path / "tenant"
        home.mkdir()
        env = {
            "NIXI_INTERNAL_SECRET": "secret123",
            "SLACK_BOT_TOKEN": "xoxb-test",
            "HERMES_HOME": str(home),
            # NIXI_TEAM_ID intentionally omitted
        }
        with patch.dict(os.environ, env, clear=True):
            os.environ.pop("NIXI_TEAM_ID", None)
            with pytest.raises(EnvironmentError, match="NIXI_TEAM_ID"):
                validate_env()

    def test_missing_slack_bot_token_raises(self, tmp_path):
        """SLACK_BOT_TOKEN must be set."""
        from nixi.deploy import validate_env

        home = tmp_path / "tenant"
        home.mkdir()
        env = {
            "NIXI_INTERNAL_SECRET": "secret123",
            "NIXI_TEAM_ID": "T_TEST",
            "HERMES_HOME": str(home),
            # SLACK_BOT_TOKEN intentionally omitted
        }
        with patch.dict(os.environ, env, clear=True):
            os.environ.pop("SLACK_BOT_TOKEN", None)
            with pytest.raises(EnvironmentError, match="SLACK_BOT_TOKEN"):
                validate_env()

    def test_missing_hermes_home_raises(self, tmp_path):
        """HERMES_HOME must be set and point to existing directory."""
        from nixi.deploy import validate_env

        env = {
            "NIXI_INTERNAL_SECRET": "secret123",
            "NIXI_TEAM_ID": "T_TEST",
            "SLACK_BOT_TOKEN": "xoxb-test",
            # HERMES_HOME intentionally omitted
        }
        with patch.dict(os.environ, env, clear=True):
            os.environ.pop("HERMES_HOME", None)
            with pytest.raises(EnvironmentError, match="HERMES_HOME"):
                validate_env()

    def test_hermes_home_must_exist(self, tmp_path):
        """HERMES_HOME must point to an existing directory."""
        from nixi.deploy import validate_env

        fake_home = tmp_path / "nonexistent"
        env = {
            "NIXI_INTERNAL_SECRET": "secret123",
            "NIXI_TEAM_ID": "T_TEST",
            "SLACK_BOT_TOKEN": "xoxb-test",
            "HERMES_HOME": str(fake_home),
        }
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(EnvironmentError, match="HERMES_HOME"):
                validate_env()

    def test_valid_env_returns_home(self, tmp_path):
        """All required env vars set → returns HERMES_HOME path."""
        from nixi.deploy import validate_env

        home = tmp_path / "tenant"
        home.mkdir()
        env = {
            "NIXI_INTERNAL_SECRET": "secret123",
            "NIXI_TEAM_ID": "T_TEST",
            "SLACK_BOT_TOKEN": "xoxb-test",
            "HERMES_HOME": str(home),
        }
        with patch.dict(os.environ, env, clear=True):
            result = validate_env()
            assert result == home.resolve()

    def test_slack_app_token_not_required(self, tmp_path):
        """SLACK_APP_TOKEN should NOT be required — NIXI_MODE disables Socket Mode."""
        from nixi.deploy import validate_env

        home = tmp_path / "tenant"
        home.mkdir()
        env = {
            "NIXI_INTERNAL_SECRET": "secret123",
            "NIXI_TEAM_ID": "T_TEST",
            "SLACK_BOT_TOKEN": "xoxb-test",
            "HERMES_HOME": str(home),
        }
        with patch.dict(os.environ, env, clear=True):
            os.environ.pop("SLACK_APP_TOKEN", None)
            # Should NOT raise
            result = validate_env()
            assert result == home.resolve()


# ─── seed_if_needed tests ────────────────────────────────────────────────


class TestSeedIfNeeded:
    """Tests for start_nixi config seeding logic."""

    def test_seeds_config_when_missing(self, tmp_path):
        """seed_if_needed creates config.yaml when it doesn't exist."""
        from nixi.deploy import seed_if_needed

        home = tmp_path / "tenant"
        home.mkdir()

        seed_if_needed(home)

        config_path = home / "config.yaml"
        assert config_path.exists()

        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        # Verify key fields exist
        assert "_config_version" in config
        assert "model" in config
        assert "gateway" in config
        assert config["gateway"]["slack"]["enabled"] is True
        assert config["gateway"]["nixi"]["enabled"] is True

    def test_skips_seeding_when_config_exists(self, tmp_path):
        """seed_if_needed does NOT overwrite existing config.yaml."""
        from nixi.deploy import seed_if_needed

        home = tmp_path / "tenant"
        home.mkdir()

        # Write a pre-existing config
        config_path = home / "config.yaml"
        config_path.write_text("model: existing-model\n", encoding="utf-8")

        seed_if_needed(home)

        # Should not have been overwritten
        content = config_path.read_text(encoding="utf-8")
        assert "existing-model" in content
        assert "gateway" not in content

    def test_seeding_creates_soul_and_agents(self, tmp_path):
        """seed_if_needed creates SOUL.md and AGENTS.md."""
        from nixi.deploy import seed_if_needed

        home = tmp_path / "tenant"
        home.mkdir()

        seed_if_needed(home)

        assert (home / "SOUL.md").exists()
        assert (home / "AGENTS.md").exists()

    def test_seeding_creates_directory_structure(self, tmp_path):
        """seed_if_needed creates the required subdirectories."""
        from nixi.deploy import seed_if_needed

        home = tmp_path / "tenant"
        home.mkdir()

        seed_if_needed(home)

        assert (home / "employees").is_dir()
        assert (home / "skills").is_dir()
        assert (home / "skills" / "seeded").is_dir()

    def test_config_version_matches_live_default(self, tmp_path):
        """_config_version in seeded config must match the live DEFAULT_CONFIG, not hardcoded."""
        from nixi.deploy import seed_if_needed
        from hermes_cli.config import DEFAULT_CONFIG

        home = tmp_path / "tenant"
        home.mkdir()

        seed_if_needed(home)

        config_path = home / "config.yaml"
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        assert config["_config_version"] == DEFAULT_CONFIG.get("_config_version", 1)


# ─── start_nixi integration tests ─────────────────────────────────────


class TestStartNixi:
    """Tests for the start_nixi() orchestration function."""

    def test_sets_nixi_mode_env(self, tmp_path):
        """start_nixi sets NIXI_MODE=1 before importing gateway."""
        from nixi.deploy import start_nixi

        home = tmp_path / "tenant"
        home.mkdir()

        # Mock the gateway import to avoid side effects
        mock_gateway = MagicMock()
        mock_gateway.start_gateway = AsyncMock(return_value=True)

        with patch.dict(os.environ, {
            "NIXI_INTERNAL_SECRET": "secret123",
            "NIXI_TEAM_ID": "T_TEST",
            "SLACK_BOT_TOKEN": "xoxb-test",
            "HERMES_HOME": str(home),
        }, clear=True):
            with patch("nixi.deploy._import_gateway", return_value=mock_gateway):
                with patch("nixi.deploy._get_cache_size", return_value=128):
                    with patch("asyncio.run") as mock_asyncio_run:
                        mock_asyncio_run.return_value = True
                        start_nixi()

        # NIXI_MODE should have been set
        # (it was set within the function before gateway import)

    def test_does_not_overwrite_existing_config(self, tmp_path):
        """If config.yaml already exists, start_nixi should NOT re-seed."""
        from nixi.deploy import start_nixi

        home = tmp_path / "tenant"
        home.mkdir()
        (home / "config.yaml").write_text("model: my-model\n", encoding="utf-8")

        mock_gateway = MagicMock()

        with patch.dict(os.environ, {
            "NIXI_INTERNAL_SECRET": "secret123",
            "NIXI_TEAM_ID": "T_TEST",
            "SLACK_BOT_TOKEN": "xoxb-test",
            "HERMES_HOME": str(home),
        }, clear=True):
            with patch("nixi.deploy._import_gateway", return_value=mock_gateway):
                with patch("nixi.deploy._get_cache_size", return_value=128):
                    with patch("asyncio.run") as mock_asyncio_run:
                        mock_asyncio_run.return_value = True
                        start_nixi()

        # Config should not have been overwritten
        content = (home / "config.yaml").read_text(encoding="utf-8")
        assert "my-model" in content
        assert "gateway" not in content

    def test_nixi_mode_set_before_gateway_import(self, tmp_path):
        """NIXI_MODE env var must be set BEFORE the gateway module is imported."""
        import nixi.deploy

        home = tmp_path / "tenant"
        home.mkdir()

        call_order = []

        def mock_import_gateway():
            call_order.append(("gateway_import", os.environ.get("NIXI_MODE")))
            return MagicMock()

        with patch.dict(os.environ, {
            "NIXI_INTERNAL_SECRET": "secret123",
            "NIXI_TEAM_ID": "T_TEST",
            "SLACK_BOT_TOKEN": "xoxb-test",
            "HERMES_HOME": str(home),
        }, clear=True):
            with patch("nixi.deploy._import_gateway", side_effect=mock_import_gateway):
                with patch("nixi.deploy._get_cache_size", return_value=128):
                    with patch("asyncio.run", return_value=True):
                        nixi.deploy.start_nixi()

        # Gateway import should have seen NIXI_MODE=1
        assert call_order[0] == ("gateway_import", "1")

    def test_seeds_when_config_missing(self, tmp_path):
        """If config.yaml doesn't exist, start_nixi seeds the tenant home."""
        from nixi.deploy import start_nixi

        home = tmp_path / "tenant"
        home.mkdir()
        # No config.yaml yet

        mock_gateway = MagicMock()

        with patch.dict(os.environ, {
            "NIXI_INTERNAL_SECRET": "secret123",
            "NIXI_TEAM_ID": "T_TEST",
            "SLACK_BOT_TOKEN": "xoxb-test",
            "HERMES_HOME": str(home),
        }, clear=True):
            with patch("nixi.deploy._import_gateway", return_value=mock_gateway):
                with patch("nixi.deploy._get_cache_size", return_value=128):
                    with patch("asyncio.run", return_value=True):
                        start_nixi()

        # Config should have been seeded
        assert (home / "config.yaml").exists()
        assert (home / "SOUL.md").exists()
        assert (home / "AGENTS.md").exists()

    def test_logs_startup_message(self, tmp_path, caplog):
        """start_nixi logs team_id and port on startup."""
        from nixi.deploy import start_nixi

        home = tmp_path / "tenant"
        home.mkdir()

        mock_gateway = MagicMock()

        with patch.dict(os.environ, {
            "NIXI_INTERNAL_SECRET": "secret123",
            "NIXI_TEAM_ID": "T_TEST_TEAM",
            "SLACK_BOT_TOKEN": "xoxb-test",
            "HERMES_HOME": str(home),
            "NIXI_PORT": "9090",
        }, clear=True):
            with patch("nixi.deploy._import_gateway", return_value=mock_gateway):
                with patch("nixi.deploy._get_cache_size", return_value=128):
                    with patch("asyncio.run", return_value=True):
                        with caplog.at_level(logging.INFO):
                            start_nixi()

        # Should log team_id and port info
        logged_messages = [r.message for r in caplog.records]
        assert any("T_TEST_TEAM" in m for m in logged_messages)

    def test_default_port_is_8080(self, tmp_path):
        """Default NIXI_PORT should be 8080 when not specified."""
        from nixi.deploy import _get_port

        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("NIXI_PORT", None)
            assert _get_port() == 8080

    def test_custom_port_from_env(self, tmp_path):
        """NIXI_PORT env var should override default."""
        from nixi.deploy import _get_port

        with patch.dict(os.environ, {"NIXI_PORT": "9090"}, clear=True):
            assert _get_port() == 9090

    def test_raises_on_invalid_env(self, tmp_path):
        """start_nixi raises before touching anything if env vars are missing."""
        from nixi.deploy import start_nixi

        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(EnvironmentError):
                start_nixi()


class TestSeedIfNeededHomeChannel:
    """Tests for NIXI_HOME_CHANNEL pass-through in seed_if_needed."""

    def test_home_channel_passed_to_seed(self, tmp_path):
        """seed_if_needed reads NIXI_HOME_CHANNEL and passes it to seed_hermes_home."""
        from nixi.deploy import seed_if_needed

        home = tmp_path / "tenant"
        home.mkdir()

        with patch.dict(os.environ, {
            "NIXI_HOME_CHANNEL": "C0AE0QVNT1P",
        }):
            seed_if_needed(home)

        config_path = home / "config.yaml"
        assert config_path.exists()

        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        assert config["gateway"]["nixi"]["home_channel"] == "C0AE0QVNT1P"

    def test_home_channel_omitted_when_env_unset(self, tmp_path):
        """seed_if_needed omits home_channel when NIXI_HOME_CHANNEL is not set."""
        from nixi.deploy import seed_if_needed

        home = tmp_path / "tenant"
        home.mkdir()

        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("NIXI_HOME_CHANNEL", None)
            seed_if_needed(home)

        config_path = home / "config.yaml"
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        assert "home_channel" not in config["gateway"]["nixi"]


class TestStartNixiHomeChannel:
    """Tests for NIXI_HOME_CHANNEL in start_nixi startup banner and seeding."""

    def test_banner_shows_home_channel_when_set(self, tmp_path, capsys):
        """start_nixi banner shows the home channel ID when NIXI_HOME_CHANNEL is set."""
        from nixi.deploy import start_nixi

        home = tmp_path / "tenant"
        home.mkdir()

        mock_gateway = MagicMock()

        with patch.dict(os.environ, {
            "NIXI_INTERNAL_SECRET": "secret123",
            "NIXI_TEAM_ID": "T_TEST",
            "SLACK_BOT_TOKEN": "xoxb-test",
            "HERMES_HOME": str(home),
            "NIXI_HOME_CHANNEL": "C0AE0QVNT1P",
        }, clear=True):
            with patch("nixi.deploy._import_gateway", return_value=mock_gateway):
                with patch("nixi.deploy._get_cache_size", return_value=128):
                    with patch("asyncio.run", return_value=True):
                        start_nixi()

        captured = capsys.readouterr()
        assert "C0AE0QVNT1P" in captured.out

    def test_banner_shows_not_set_when_empty(self, tmp_path, capsys):
        """start_nixi banner shows '(not set)' when NIXI_HOME_CHANNEL is empty."""
        from nixi.deploy import start_nixi

        home = tmp_path / "tenant"
        home.mkdir()

        mock_gateway = MagicMock()

        with patch.dict(os.environ, {
            "NIXI_INTERNAL_SECRET": "secret123",
            "NIXI_TEAM_ID": "T_TEST",
            "SLACK_BOT_TOKEN": "xoxb-test",
            "HERMES_HOME": str(home),
        }, clear=True):
            os.environ.pop("NIXI_HOME_CHANNEL", None)
            with patch("nixi.deploy._import_gateway", return_value=mock_gateway):
                with patch("nixi.deploy._get_cache_size", return_value=128):
                    with patch("asyncio.run", return_value=True):
                        start_nixi()

        captured = capsys.readouterr()
        assert "not set" in captured.out


class TestStartNixiCacheSizeLogging:
    """Tests for cache size logging in start_nixi startup."""

    def test_cache_size_logged_in_banner(self, tmp_path, capsys):
        """start_nixi prints Cache Size in startup banner."""
        from nixi.deploy import start_nixi

        home = tmp_path / "tenant"
        home.mkdir()

        mock_gateway = MagicMock()

        with patch.dict(os.environ, {
            "NIXI_INTERNAL_SECRET": "secret123",
            "NIXI_TEAM_ID": "T_TEST",
            "SLACK_BOT_TOKEN": "xoxb-test",
            "HERMES_HOME": str(home),
        }, clear=True):
            with patch("nixi.deploy._import_gateway", return_value=mock_gateway):
                with patch("nixi.deploy._get_cache_size", return_value=256):
                    with patch("asyncio.run", return_value=True):
                        start_nixi()

        captured = capsys.readouterr()
        assert "Cache Size: 256" in captured.out

    def test_warns_on_cache_size_below_minimum(self, tmp_path, capsys, caplog):
        """start_nixi warns when cache size is below 16."""
        import logging as _logging

        from nixi.deploy import start_nixi

        home = tmp_path / "tenant"
        home.mkdir()

        mock_gateway = MagicMock()

        with patch.dict(os.environ, {
            "NIXI_INTERNAL_SECRET": "secret123",
            "NIXI_TEAM_ID": "T_TEST",
            "SLACK_BOT_TOKEN": "xoxb-test",
            "HERMES_HOME": str(home),
        }, clear=True):
            with patch("nixi.deploy._import_gateway", return_value=mock_gateway):
                with patch("nixi.deploy._get_cache_size", return_value=8):
                    with patch("asyncio.run", return_value=True):
                        with caplog.at_level(_logging.WARNING):
                            start_nixi()

        captured = capsys.readouterr()
        assert "[WARN] Cache size 8 is below recommended minimum (16)" in captured.out
        assert any("below recommended minimum" in r.message for r in caplog.records)

    def test_warns_on_cache_size_above_maximum(self, tmp_path, capsys, caplog):
        """start_nixi warns when cache size exceeds 1024."""
        import logging as _logging

        from nixi.deploy import start_nixi

        home = tmp_path / "tenant"
        home.mkdir()

        mock_gateway = MagicMock()

        with patch.dict(os.environ, {
            "NIXI_INTERNAL_SECRET": "secret123",
            "NIXI_TEAM_ID": "T_TEST",
            "SLACK_BOT_TOKEN": "xoxb-test",
            "HERMES_HOME": str(home),
        }, clear=True):
            with patch("nixi.deploy._import_gateway", return_value=mock_gateway):
                with patch("nixi.deploy._get_cache_size", return_value=2048):
                    with patch("asyncio.run", return_value=True):
                        with caplog.at_level(_logging.WARNING):
                            start_nixi()

        captured = capsys.readouterr()
        assert "[WARN] Cache size 2048 exceeds recommended maximum (1024)" in captured.out
        assert any("exceeds recommended maximum" in r.message for r in caplog.records)

    def test_no_warning_at_default_size(self, tmp_path, capsys):
        """start_nixi does not warn at default cache size 128."""
        from nixi.deploy import start_nixi

        home = tmp_path / "tenant"
        home.mkdir()

        mock_gateway = MagicMock()

        with patch.dict(os.environ, {
            "NIXI_INTERNAL_SECRET": "secret123",
            "NIXI_TEAM_ID": "T_TEST",
            "SLACK_BOT_TOKEN": "xoxb-test",
            "HERMES_HOME": str(home),
        }, clear=True):
            with patch("nixi.deploy._import_gateway", return_value=mock_gateway):
                with patch("nixi.deploy._get_cache_size", return_value=128):
                    with patch("asyncio.run", return_value=True):
                        start_nixi()

        captured = capsys.readouterr()
        assert "[WARN]" not in captured.out


import logging