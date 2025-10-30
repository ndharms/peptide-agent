import os
from pathlib import Path

import yaml
from pydantic import BaseModel


class Settings(BaseModel):
    data_dir: str = "data"
    faiss_cache_dir: str = "data/index/faiss"
    embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_model: str = "gemini-2.5-pro"
    top_k: int = 10
    batch_size: int = 40

    @classmethod
    def from_yaml(cls, path: str | None = None) -> "Settings":
        raw = {}
        if path and Path(path).exists():
            with open(path, encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}

        # Environment variables override YAML config
        # Priority: env var > yaml > default
        if "PEPTIDE_DATA_DIR" in os.environ:
            raw["data_dir"] = os.environ["PEPTIDE_DATA_DIR"]
        elif "data_dir" not in raw:
            raw["data_dir"] = cls().data_dir

        if "PEPTIDE_FAISS_DIR" in os.environ:
            raw["faiss_cache_dir"] = os.environ["PEPTIDE_FAISS_DIR"]
        elif "faiss_cache_dir" not in raw:
            raw["faiss_cache_dir"] = cls().faiss_cache_dir

        if "PEPTIDE_EMBED_MODEL" in os.environ:
            raw["embed_model_name"] = os.environ["PEPTIDE_EMBED_MODEL"]
        elif "embed_model_name" not in raw:
            raw["embed_model_name"] = cls().embed_model_name

        if "PEPTIDE_LLM_MODEL" in os.environ:
            raw["llm_model"] = os.environ["PEPTIDE_LLM_MODEL"]
        elif "llm_model" not in raw:
            raw["llm_model"] = cls().llm_model

        # Prefer PEPTIDE_TOP_K, but accept legacy env var "top_k" for backwards compatibility
        if "PEPTIDE_TOP_K" in os.environ:
            raw["top_k"] = int(os.environ["PEPTIDE_TOP_K"])
        elif "top_k" in os.environ:
            raw["top_k"] = int(os.environ["top_k"])
        elif "top_k" not in raw:
            raw["top_k"] = cls().top_k

        if "PEPTIDE_BATCH_SIZE" in os.environ:
            raw["batch_size"] = int(os.environ["PEPTIDE_BATCH_SIZE"])
        elif "batch_size" not in raw:
            raw["batch_size"] = cls().batch_size

        return cls(**raw)
