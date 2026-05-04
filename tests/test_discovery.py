"""Tests for nixi.discovery — HERMES_HOME CWD-walk discovery."""

import os
from contextlib import contextmanager
from pathlib import Path

import pytest

from nixi.discovery import discover_hermes_home


@pytest.fixture
def tmp_project(tmp_path):
    """Provide a temp directory factory and chdir context manager."""

    class Factory:
        def create_tenant(self, root: Path, tenant_id: str, config_body: str) -> Path:
            """Create data/tenant/{id}/config.yaml under root, return root."""
            tenant_dir = root / "data" / "tenant" / tenant_id
            tenant_dir.mkdir(parents=True)
            config_file = tenant_dir / "config.yaml"
            config_file.write_text(config_body)
            return root

        @contextmanager
        def chdir(self, target: Path):
            """Context manager to change CWD and restore on exit."""
            old = os.getcwd()
            os.chdir(target)
            try:
                yield
            finally:
                os.chdir(old)

    return Factory()


class TestDiscoverHermesHome:
    """discover_hermes_home() walks CWD upward to find valid hermes_home."""

    def test_finds_valid_hermes_home_in_cwd(self, tmp_path, tmp_project):
        """Returns hermes_home dir (parent of data/) when valid config in CWD."""
        root = tmp_project.create_tenant(tmp_path, "company", "nixi:\n  extraction_model: gpt-4o\n")
        with tmp_project.chdir(root):
            result = discover_hermes_home()
        assert result == root

    def test_finds_valid_hermes_home_in_parent(self, tmp_path, tmp_project):
        """Returns hermes_home when config found in parent directory."""
        root = tmp_project.create_tenant(tmp_path, "company", "model: gpt-4o\n")
        subdir = root / "subdir"
        subdir.mkdir()
        with tmp_project.chdir(subdir):
            result = discover_hermes_home()
        assert result == root

    def test_returns_none_when_no_tenant_found(self, tmp_path, tmp_project):
        """Returns None when no data/tenant/ exists anywhere in CWD ancestry."""
        with tmp_project.chdir(tmp_path):
            result = discover_hermes_home()
        assert result is None

    def test_returns_none_when_multiple_tenants_at_same_level(self, tmp_path, tmp_project):
        """Returns None (ambiguous) when multiple valid tenants at same depth."""
        tmp_project.create_tenant(tmp_path, "alpha", "nixi: {}\n")
        tmp_project.create_tenant(tmp_path, "beta", "nixi: {}\n")
        with tmp_project.chdir(tmp_path):
            result = discover_hermes_home()
        assert result is None

    def test_returns_none_for_config_without_nixi_or_model_keys(self, tmp_path, tmp_project):
        """Returns None when config.yaml exists but lacks nixi: or model: keys."""
        tenant_dir = tmp_path / "data" / "tenant" / "company"
        tenant_dir.mkdir(parents=True)
        config_file = tenant_dir / "config.yaml"
        config_file.write_text("other_key: value\n")
        with tmp_project.chdir(tmp_path):
            result = discover_hermes_home()
        assert result is None

    def test_respects_max_depth(self, tmp_path, tmp_project):
        """Stops search after max_depth levels even if match exists higher up."""
        # Build a deep nesting; put the tenant config 10 levels up from leaf.
        root = tmp_path
        tmp_project.create_tenant(root, "deep", "nixi: {}\n")
        leaf = root
        for i in range(10):
            leaf = leaf / f"level{i}"
        leaf.mkdir(parents=True)
        with tmp_project.chdir(leaf):
            # max_depth=3 should not reach root (10 levels up)
            result = discover_hermes_home(start_dir=leaf, max_depth=3)
        assert result is None

    def test_finds_hermes_home_with_explicit_start_dir(self, tmp_path, tmp_project):
        """Finds hermes_home using explicit start_dir without chdir."""
        root = tmp_project.create_tenant(tmp_path, "explicit", "model: claude-3\n")
        subdir = root / "nested" / "deep"
        subdir.mkdir(parents=True)
        result = discover_hermes_home(start_dir=subdir)
        assert result == root

    def test_returns_parent_of_data_as_hermes_home(self, tmp_path, tmp_project):
        """Returns the directory containing data/, NOT the tenant dir or data/ itself."""
        root = tmp_project.create_tenant(tmp_path, "acme", "nixi:\n  extraction_model: gpt-4o\n")
        with tmp_project.chdir(root):
            result = discover_hermes_home()
        # Must be root (parent of data/), not data/ or data/tenant/acme
        assert result == root
        assert result != root / "data"
        assert result != root / "data" / "tenant" / "acme"

    def test_finds_deeply_nested_cwd(self, tmp_path, tmp_project):
        """Finds hermes_home when CWD is several levels below the project root."""
        root = tmp_project.create_tenant(tmp_path, "corp", "nixi:\n  extraction_model: gpt-4o\n")
        deep = root / "a" / "b" / "c"
        deep.mkdir(parents=True)
        with tmp_project.chdir(deep):
            result = discover_hermes_home()
        assert result == root

    def test_returns_none_at_filesystem_root_with_no_match(self, tmp_path, tmp_project, monkeypatch):
        """Returns None when CWD is filesystem root with no match."""
        # Use start_dir=/ to test root boundary without actually chdir to /
        result = discover_hermes_home(start_dir=Path("/"), max_depth=2)
        assert result is None

    def test_yaml_with_nixi_section_is_valid(self, tmp_path, tmp_project):
        """Config with nixi: section (even empty) is treated as valid."""
        root = tmp_project.create_tenant(tmp_path, "minimal", "nixi: {}\n")
        with tmp_project.chdir(root):
            result = discover_hermes_home()
        assert result == root

    def test_yaml_with_model_key_is_valid(self, tmp_path, tmp_project):
        """Config with model: key at top level is treated as valid."""
        root = tmp_project.create_tenant(tmp_path, "modelonly", "model: gpt-4o\n")
        with tmp_project.chdir(root):
            result = discover_hermes_home()
        assert result == root

    def test_unparseable_yaml_treated_as_invalid(self, tmp_path, tmp_project):
        """Returns None when config.yaml exists but is not valid YAML."""
        tenant_dir = tmp_path / "data" / "tenant" / "broken"
        tenant_dir.mkdir(parents=True)
        config_file = tenant_dir / "config.yaml"
        config_file.write_text("{{invalid yaml::\n")
        with tmp_project.chdir(tmp_path):
            result = discover_hermes_home()
        assert result is None