"""Integration tests for the peptide agent."""

import json
from pathlib import Path

import pandas as pd
import pytest

from peptide_agent.config import Settings
from peptide_agent.indexing.faiss_store import _load_context_documents, build_or_load_vectorstore


class TestIndexingIntegration:
    """Test indexing integration with real file operations."""

    def test_load_context_documents_valid_csv(self, tmp_path):
        """Test loading valid training data."""
        # Create a test CSV file
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        csv_path = data_dir / "train_split.csv"
        df = pd.DataFrame(
            {
                "PEPTIDE_CODE": ["FF", "YY"],
                "Morphology": ["nanofibers", "hydrogel"],
                "PH": [6.5, 7.0],
                "CONCENTRATION_mg ml": [1.0, 2.0],
                "TEMPERATURE_C": [25, 25],
                "SOLVENT": ["Water", "PBS"],
                "Time (min)": [60, 120],
            }
        )
        df.to_csv(csv_path, index=False)

        # Load documents
        docs = _load_context_documents(data_dir)
        assert len(docs) == 2
        assert all(doc.page_content for doc in docs)
        assert all(doc.metadata["type"] == "train_example" for doc in docs)

    def test_load_context_documents_empty_dir(self, tmp_path):
        """Test loading from empty directory."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        docs = _load_context_documents(empty_dir)
        assert len(docs) == 0

    def test_build_vectorstore_from_scratch(self, tmp_path):
        """Test building vectorstore from scratch."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        cache_dir = tmp_path / "cache"

        # Create minimal training data
        csv_path = data_dir / "train_split.csv"
        df = pd.DataFrame(
            {
                "PEPTIDE_CODE": ["FF"],
                "Morphology": ["nanofibers"],
                "PH": [6.5],
                "CONCENTRATION_mg ml": [1.0],
                "TEMPERATURE_C": [25],
                "SOLVENT": ["Water"],
                "Time (min)": [60],
            }
        )
        df.to_csv(csv_path, index=False)

        # Build vectorstore
        vectorstore = build_or_load_vectorstore(
            data_dir=data_dir,
            cache_dir=cache_dir,
            embed_model_name="sentence-transformers/all-MiniLM-L6-v2",
            refresh=False,
        )

        assert vectorstore is not None
        assert (cache_dir / "index.faiss").exists()
        assert (cache_dir / "index.pkl").exists()

    def test_load_existing_vectorstore(self, tmp_path):
        """Test loading existing vectorstore from cache."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        cache_dir = tmp_path / "cache"

        # Create minimal training data
        csv_path = data_dir / "train_split.csv"
        df = pd.DataFrame(
            {
                "PEPTIDE_CODE": ["FF"],
                "Morphology": ["nanofibers"],
            }
        )
        df.to_csv(csv_path, index=False)

        # Build first time
        build_or_load_vectorstore(
            data_dir=data_dir,
            cache_dir=cache_dir,
            embed_model_name="sentence-transformers/all-MiniLM-L6-v2",
            refresh=False,
        )

        # Load second time (should use cache)
        vectorstore2 = build_or_load_vectorstore(
            data_dir=data_dir,
            cache_dir=cache_dir,
            embed_model_name="sentence-transformers/all-MiniLM-L6-v2",
            refresh=False,
        )

        assert vectorstore2 is not None

    def test_refresh_vectorstore(self, tmp_path):
        """Test refreshing vectorstore."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        cache_dir = tmp_path / "cache"

        # Create minimal training data
        csv_path = data_dir / "train_split.csv"
        df = pd.DataFrame(
            {
                "PEPTIDE_CODE": ["FF"],
                "Morphology": ["nanofibers"],
            }
        )
        df.to_csv(csv_path, index=False)

        # Build first time
        build_or_load_vectorstore(
            data_dir=data_dir,
            cache_dir=cache_dir,
            embed_model_name="sentence-transformers/all-MiniLM-L6-v2",
            refresh=False,
        )

        # Force refresh
        vectorstore = build_or_load_vectorstore(
            data_dir=data_dir,
            cache_dir=cache_dir,
            embed_model_name="sentence-transformers/all-MiniLM-L6-v2",
            refresh=True,
        )

        assert vectorstore is not None


class TestConfigIntegration:
    """Test configuration integration."""

    def test_settings_from_yaml_file(self, tmp_path):
        """Test loading settings from YAML file."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
data_dir: /custom/data
faiss_cache_dir: /custom/cache
llm_model: gemini-2.5-flash
top_k: 20
""")

        settings = Settings.from_yaml(str(config_file))
        assert settings.data_dir == "/custom/data"
        assert settings.faiss_cache_dir == "/custom/cache"
        assert settings.llm_model == "gemini-2.5-flash"
        assert settings.top_k == 20

    def test_settings_env_overrides_yaml(self, tmp_path, monkeypatch):
        """Test that environment variables override YAML config."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
data_dir: /yaml/data
top_k: 10
""")

        monkeypatch.setenv("PEPTIDE_DATA_DIR", "/env/data")
        monkeypatch.setenv("PEPTIDE_TOP_K", "30")

        settings = Settings.from_yaml(str(config_file))
        assert settings.data_dir == "/env/data"
        assert settings.top_k == 30


class TestEndToEndScenarios:
    """Test end-to-end scenarios."""

    def test_batch_json_format(self, tmp_path):
        """Test batch request JSON format validation."""
        json_file = tmp_path / "requests.json"
        requests = [
            {"peptide_code": "FF", "target_structural_assembly": "nanofibers"},
            {"peptide_code": "YY", "target_structural_assembly": "hydrogel"},
        ]
        json_file.write_text(json.dumps(requests))

        # Load and validate
        loaded = json.loads(json_file.read_text())
        assert len(loaded) == 2
        assert all("peptide_code" in req for req in loaded)
        assert all("target_structural_assembly" in req for req in loaded)

    def test_malformed_csv_handling(self, tmp_path):
        """Test handling of malformed CSV data."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        # Create CSV with missing columns
        csv_path = data_dir / "train_split.csv"
        csv_path.write_text("PEPTIDE_CODE\nFF\n")  # Missing other columns

        # Should not crash
        docs = _load_context_documents(data_dir)
        # May have fewer docs due to errors, but should complete
        assert isinstance(docs, list)
