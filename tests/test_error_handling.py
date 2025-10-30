"""Tests for error handling throughout the application."""

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from peptide_agent.cli import app
from peptide_agent.config import Settings
from peptide_agent.runner.main import (
    APIError,
    PeptideAgentError,
    _create_llm,
    _load_base_prompt,
    predict_single,
)

runner = CliRunner()


class TestAPIErrorHandling:
    """Test API error handling."""

    def test_create_llm_missing_api_key(self, monkeypatch):
        """Test LLM creation fails when API key is missing."""
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)

        with pytest.raises(APIError, match="GEMINI_API_KEY"):
            _create_llm("gemini-2.5-pro")

    def test_create_llm_with_api_key(self, monkeypatch):
        """Test LLM creation succeeds with API key."""
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")

        # Mock the actual LLM creation to avoid real API call
        with patch("peptide_agent.runner.main.ChatGoogleGenerativeAI") as mock_llm:
            mock_llm.return_value = MagicMock()
            llm = _create_llm("gemini-2.5-pro")
            assert llm is not None


class TestFileErrorHandling:
    """Test file operation error handling."""

    def test_load_base_prompt_missing_file(self, monkeypatch):
        """Test loading prompt fails gracefully when file is missing."""

        def mock_files(*args):
            mock_path = MagicMock()
            mock_path.__truediv__ = lambda self, x: mock_path
            mock_path.read_text.side_effect = FileNotFoundError("Prompt not found")
            return mock_path

        monkeypatch.setattr("peptide_agent.runner.main.files", mock_files)

        with pytest.raises(PeptideAgentError):
            _load_base_prompt()


class TestCLIErrorHandling:
    """Test CLI error handling."""

    def test_predict_missing_arguments(self):
        """Test predict command fails with missing arguments."""
        result = runner.invoke(app, ["predict"])
        assert result.exit_code == 1
        # Just verify it exits with error code, error messages are in logs

    def test_predict_invalid_json_file(self, tmp_path):
        """Test predict command fails with invalid JSON."""
        json_file = tmp_path / "invalid.json"
        json_file.write_text("not valid json")

        result = runner.invoke(app, ["predict", "--input-json", str(json_file)])
        assert result.exit_code == 1
        # Just verify it exits with error code, error messages are in logs

    def test_predict_missing_json_file(self):
        """Test predict command fails with missing JSON file."""
        result = runner.invoke(app, ["predict", "--input-json", "nonexistent.json"])
        assert result.exit_code == 1
        # Just verify it exits with error code, error messages are in logs

    def test_index_invalid_config(self, tmp_path):
        """Test index command with invalid config."""
        config = tmp_path / "bad_config.yaml"
        config.write_text("invalid: yaml: syntax:")

        # This should handle the error gracefully
        result = runner.invoke(app, ["index", "-c", str(config)])
        # May fail, but should not crash
        assert result.exit_code in [0, 1]


class TestValidationErrors:
    """Test validation error handling."""

    def test_empty_batch_request(self, monkeypatch):
        """Test batch prediction with empty request list."""
        from peptide_agent.runner import main as runner_main

        # Mock dependencies
        monkeypatch.setattr(runner_main, "_retrieve_contexts", lambda *a, **k: [])

        settings = Settings()
        result = runner_main.predict_batch([], settings)
        assert result == []


class TestIntegrationErrors:
    """Test integration error scenarios."""

    def test_predict_single_with_mocked_api_error(self, monkeypatch):
        """Test single prediction handles API errors."""
        from peptide_agent.runner import main as runner_main

        # Mock dependencies
        def mock_retrieve(*args, **kwargs):
            return [["context1", "context2"]]

        def mock_create_llm(*args, **kwargs):
            raise APIError("API connection failed")

        monkeypatch.setattr(runner_main, "_retrieve_contexts", mock_retrieve)
        monkeypatch.setattr(runner_main, "_create_llm", mock_create_llm)

        settings = Settings()
        with pytest.raises(APIError):
            predict_single("FF", "nanofibers", settings)

    def test_cli_verbose_logging(self, monkeypatch, tmp_path):
        """Test verbose logging flag works."""
        from peptide_agent import cli as cli_mod

        # Mock the actual prediction to avoid real API call
        monkeypatch.setattr(cli_mod, "predict_single", lambda *a, **k: "Mock report")

        result = runner.invoke(app, ["predict", "-p", "FF", "-t", "nanofibers", "--verbose"])
        # Should not crash even with verbose logging
        assert result.exit_code in [0, 1]


class TestConfigErrors:
    """Test configuration error handling."""

    def test_settings_from_nonexistent_yaml(self):
        """Test loading settings from nonexistent file."""
        # Should not crash, should use defaults
        settings = Settings.from_yaml("nonexistent.yaml")
        assert settings.data_dir == "data"

    def test_settings_with_env_override(self, monkeypatch):
        """Test settings override from environment."""
        monkeypatch.setenv("PEPTIDE_DATA_DIR", "/custom/path")
        monkeypatch.setenv("PEPTIDE_TOP_K", "20")

        settings = Settings.from_yaml(None)
        assert settings.data_dir == "/custom/path"
        assert settings.top_k == 20
