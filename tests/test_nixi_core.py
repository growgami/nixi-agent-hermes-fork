"""Tests for nixi package core modules: path_validator, employee_provider, seed_config, config_seeder."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml


# ─── path_validator tests ────────────────────────────────────────────────


class TestSafePath:
    """Tests for nixi.path_validator.safe_path."""

    def test_simple_path_resolves_within_home(self, tmp_path):
        from nixi.path_validator import safe_path

        home = tmp_path / "hermes"
        home.mkdir()
        result = safe_path(home, "employees/user1/USER.md")
        assert str(result).startswith(str(home.resolve()))

    def test_dot_path_resolves_to_home(self, tmp_path):
        from nixi.path_validator import safe_path

        home = tmp_path / "hermes"
        home.mkdir()
        result = safe_path(home, ".")
        assert result.resolve() == home.resolve()

    def test_traversal_raises_error(self, tmp_path):
        from nixi.path_validator import PathTraversalError, safe_path

        home = tmp_path / "hermes"
        home.mkdir()
        with pytest.raises(PathTraversalError):
            safe_path(home, "../../../etc/passwd")

    def test_double_dot_traversal_raises_error(self, tmp_path):
        from nixi.path_validator import PathTraversalError, safe_path

        home = tmp_path / "hermes"
        home.mkdir()
        with pytest.raises(PathTraversalError):
            safe_path(home, "..")

    def test_symlink_escape_raises_error(self, tmp_path):
        from nixi.path_validator import PathTraversalError, safe_path

        home = tmp_path / "hermes"
        home.mkdir()
        # Create a symlink pointing outside the home dir
        outside = tmp_path / "outside"
        outside.mkdir()
        link = home / "escape_link"
        try:
            link.symlink_to(outside)
        except OSError:
            pytest.skip("Symlinks not supported on this platform")

        with pytest.raises(PathTraversalError):
            safe_path(home, "escape_link/secret.txt")

    def test_absolute_path_outside_home_raises_error(self, tmp_path):
        from nixi.path_validator import PathTraversalError, safe_path

        home = tmp_path / "hermes"
        home.mkdir()
        with pytest.raises(PathTraversalError):
            safe_path(home, "/etc/passwd")

    def test_mixed_traversal_raises_error(self, tmp_path):
        from nixi.path_validator import PathTraversalError, safe_path

        home = tmp_path / "hermes"
        home.mkdir()
        with pytest.raises(PathTraversalError):
            safe_path(home, "skills/../../etc/shadow")

    def test_valid_nested_path(self, tmp_path):
        from nixi.path_validator import safe_path

        home = tmp_path / "hermes"
        home.mkdir()
        result = safe_path(home, "skills/seeded/example.md")
        assert result.resolve().is_relative_to(home.resolve())


class TestValidateHermesHome:
    """Tests for nixi.path_validator.validate_hermes_home."""

    def test_valid_home_returns_path(self, tmp_path):
        from nixi.path_validator import validate_hermes_home

        home = tmp_path / "hermes"
        home.mkdir()
        with patch.dict(os.environ, {"HERMES_HOME": str(home)}):
            result = validate_hermes_home()
            assert result == home.resolve()

    def test_missing_env_var_raises_error(self):
        from nixi.path_validator import validate_hermes_home

        with patch.dict(os.environ, {}, clear=True):
            # Remove HERMES_HOME if it exists
            os.environ.pop("HERMES_HOME", None)
            with pytest.raises(EnvironmentError):
                validate_hermes_home()

    def test_nonexistent_dir_raises_error(self, tmp_path):
        from nixi.path_validator import validate_hermes_home

        fake_home = tmp_path / "nonexistent"
        with patch.dict(os.environ, {"HERMES_HOME": str(fake_home)}):
            with pytest.raises(EnvironmentError):
                validate_hermes_home()


# ─── employee_provider tests ─────────────────────────────────────────────


class TestLoadOverlay:
    """Tests for nixi.employee_provider.load_overlay."""

    def test_returns_content_for_existing_employee(self, tmp_path):
        from nixi.employee_provider import load_overlay

        home = tmp_path / "hermes"
        emp_dir = home / "employees" / "user123"
        emp_dir.mkdir(parents=True)
        (emp_dir / "USER.md").write_text("# Employee Info\nSome details", encoding="utf-8")

        with patch.dict(os.environ, {"HERMES_HOME": str(home)}):
            result = load_overlay("user123")
            assert result == "# Employee Info\nSome details"

    def test_returns_empty_string_for_missing_employee(self, tmp_path):
        from nixi.employee_provider import load_overlay

        home = tmp_path / "hermes"
        home.mkdir()

        with patch.dict(os.environ, {"HERMES_HOME": str(home)}):
            result = load_overlay("nonexistent_user")
            assert result == ""

    def test_returns_empty_string_for_missing_employees_dir(self, tmp_path):
        from nixi.employee_provider import load_overlay

        home = tmp_path / "hermes"
        home.mkdir()
        # No employees/ directory at all

        with patch.dict(os.environ, {"HERMES_HOME": str(home)}):
            result = load_overlay("any_user")
            assert result == ""


class TestGetOrCreateEmployeeDir:
    """Tests for nixi.employee_provider.get_or_create_employee_dir."""

    def test_creates_directory_if_missing(self, tmp_path):
        from nixi.employee_provider import get_or_create_employee_dir

        home = tmp_path / "hermes"
        home.mkdir()

        with patch.dict(os.environ, {"HERMES_HOME": str(home)}):
            emp_dir = get_or_create_employee_dir("user456")
            assert emp_dir.exists()
            assert emp_dir.name == "user456"

    def test_returns_existing_directory(self, tmp_path):
        from nixi.employee_provider import get_or_create_employee_dir

        home = tmp_path / "hermes"
        existing = home / "employees" / "user789"
        existing.mkdir(parents=True)

        with patch.dict(os.environ, {"HERMES_HOME": str(home)}):
            emp_dir = get_or_create_employee_dir("user789")
            assert emp_dir == existing


# ─── seed_config tests ──────────────────────────────────────────────────


class TestGenerateSeedConfig:
    """Tests for nixi.seed_config.generate_seed_config."""

    def test_produces_valid_config_dict(self):
        from nixi.seed_config import generate_seed_config

        config = generate_seed_config(
            company_name="TestCorp",
            slack_workspace_id="T12345",
            model_provider="openai",
            model="gpt-4o",
        )

        assert isinstance(config, dict)
        assert "_config_version" in config
        assert "model" in config

    def test_config_version_read_dynamically(self):
        """_config_version must come from DEFAULT_CONFIG, not hardcoded."""
        from nixi.seed_config import generate_seed_config

        from hermes_cli.config import DEFAULT_CONFIG

        config = generate_seed_config(
            company_name="TestCorp",
            slack_workspace_id="T12345",
            model_provider="openai",
            model="gpt-4o",
        )

        assert config["_config_version"] == DEFAULT_CONFIG.get("_config_version", 1)

    def test_config_includes_model(self):
        from nixi.seed_config import generate_seed_config

        config = generate_seed_config(
            company_name="TestCorp",
            slack_workspace_id="T12345",
            model_provider="openai",
            model="gpt-4o",
        )

        assert config["model"] == "gpt-4o"

    def test_config_includes_gateway_settings(self):
        from nixi.seed_config import generate_seed_config

        config = generate_seed_config(
            company_name="TestCorp",
            slack_workspace_id="T12345",
            model_provider="openai",
            model="gpt-4o",
        )

        assert "gateway" in config
        assert config["gateway"]["slack"]["enabled"] is True
        assert config["gateway"]["nixi"]["enabled"] is True

    def test_config_includes_memory_scope(self):
        from nixi.seed_config import generate_seed_config

        config = generate_seed_config(
            company_name="TestCorp",
            slack_workspace_id="T12345",
            model_provider="openai",
            model="gpt-4o",
        )

        assert config["memory"]["scope"] == "organization"


# ─── config_seeder tests ────────────────────────────────────────────────


class TestSeedHermesHome:
    """Tests for nixi.config_seeder.seed_hermes_home."""

    def test_creates_directory_structure(self, tmp_path):
        from nixi.config_seeder import seed_hermes_home

        home = tmp_path / "tenants" / "testcorp"
        seed_hermes_home(
            home=home,
            company_name="TestCorp",
            slack_workspace_id="T12345",
            model_provider="openai",
            model="gpt-4o",
            soul_content="# Be helpful",
            agents_content="# Agents config",
        )

        assert home.exists()
        assert (home / "employees").is_dir()
        assert (home / "skills").is_dir()
        assert (home / "skills" / "seeded").is_dir()
        assert (home / "skills" / "channel").is_dir()
        assert (home / "skills" / "event").is_dir()
        assert (home / "skills" / "learned").is_dir()
        assert (home / "skills" / "archive").is_dir()

    def test_writes_config_yaml(self, tmp_path):
        from nixi.config_seeder import seed_hermes_home

        home = tmp_path / "tenants" / "testcorp"
        seed_hermes_home(
            home=home,
            company_name="TestCorp",
            slack_workspace_id="T12345",
            model_provider="openai",
            model="gpt-4o",
            soul_content="# Be helpful",
            agents_content="# Agents config",
        )

        config_path = home / "config.yaml"
        assert config_path.exists()

        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        assert config["model"] == "gpt-4o"
        assert config["gateway"]["slack"]["enabled"] is True
        assert config["gateway"]["nixi"]["enabled"] is True

    def test_writes_soul_md(self, tmp_path):
        from nixi.config_seeder import seed_hermes_home

        home = tmp_path / "tenants" / "testcorp"
        seed_hermes_home(
            home=home,
            company_name="TestCorp",
            slack_workspace_id="T12345",
            model_provider="openai",
            model="gpt-4o",
            soul_content="# Custom Soul",
            agents_content="# Agents",
        )

        soul_path = home / "SOUL.md"
        assert soul_path.exists()
        assert soul_path.read_text(encoding="utf-8") == "# Custom Soul"

    def test_writes_agents_md(self, tmp_path):
        from nixi.config_seeder import seed_hermes_home

        home = tmp_path / "tenants" / "testcorp"
        seed_hermes_home(
            home=home,
            company_name="TestCorp",
            slack_workspace_id="T12345",
            model_provider="openai",
            model="gpt-4o",
            soul_content="# Soul",
            agents_content="# Custom Agents Config",
        )

        agents_path = home / "AGENTS.md"
        assert agents_path.exists()
        assert agents_path.read_text(encoding="utf-8") == "# Custom Agents Config"

    def test_config_yaml_has_dynamic_version(self, tmp_path):
        from nixi.config_seeder import seed_hermes_home

        from hermes_cli.config import DEFAULT_CONFIG

        home = tmp_path / "tenants" / "testcorp"
        seed_hermes_home(
            home=home,
            company_name="TestCorp",
            slack_workspace_id="T12345",
            model_provider="openai",
            model="gpt-4o",
            soul_content="# Soul",
            agents_content="# Agents",
        )

        config_path = home / "config.yaml"
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        assert config["_config_version"] == DEFAULT_CONFIG.get("_config_version", 1)