# tests/conftest.py
import json
import types
import pytest
from pathlib import Path

# --- Minimal fake Doc type (avoids importing LangChain Document) ---
class FakeDoc:
    def __init__(self, content, metadata=None):
        self.page_content = content
        self.metadata = metadata or {}

# --- Fake vectorstore/retriever ---
class FakeRetriever:
    def __init__(self, docs):
        self._docs = docs
    def invoke(self, query):
        return list(self._docs)

class FakeVectorStore:
    def __init__(self, docs):
        self._docs = docs
    def as_retriever(self, search_kwargs=None):
        return FakeRetriever(self._docs)

@pytest.fixture
def tmp_repo(tmp_path: Path):
    """
    Creates a minimal repo layout in a temp dir:
    data/context_examples.json
    data/relevant_papers/PaperA/{paper.pdf, metadata.json}
    """
    root = tmp_path
    data = root / "data"
    papers = data / "relevant_papers" / "PaperA"
    papers.mkdir(parents=True)

    # Context examples (make sure they contain "SUCCESS" for the agent filter)
    examples = [
        {
            "PEPTIDE_CODE": "FF",
            "MORPHOLOGY": "Fiber",
            "PH": "(6.5,7.5)",
            "CONCENTRATION_LOG_MGML": "(-1,0)",
            "TEMPERATURE_C": "25",
            "SOLVENT": "Water",
            "OUTCOME": "SUCCESS",
        }
    ]
    (data / "context_examples.json").write_text(
        json.dumps(examples, indent=2), encoding="utf-8"
    )

    # metadata.json (only DOI/PAPER_TYPE are required by your current settings)
    (papers / "metadata.json").write_text(
        json.dumps({"DOI": "10.1234/example", "PAPER_TYPE": "RESEARCH"}), encoding="utf-8"
    )

    # A placeholder PDF file; we won't actually parse it because we'll stub PyPDFLoader
    (papers / "paper.pdf").write_bytes(b"%PDF-1.4\n%dummy")

    return root

@pytest.fixture
def patch_sys_path(tmp_repo, monkeypatch):
    """
    Adds the temp repo to sys.path so imports like src.pipeline.agent work under tmp.
    """
    import sys
    monkeypatch.chdir(tmp_repo)
    sys.path.insert(0, str(tmp_repo))
    yield
    sys.path.remove(str(tmp_repo))

@pytest.fixture
def stub_heavy_deps(monkeypatch):
    """
    Stubs out heavy dependencies: PyPDFLoader, CharacterTextSplitter, HF Embeddings, FAISS, Gemini.
    Returns a holder so tests can customize returned docs.
    """
    holder = types.SimpleNamespace()

    # --- stub PyPDFLoader.load -> list[FakeDoc] ---
    class StubLoader:
        def __init__(self, _):
            pass
        def load(self):
            # one page with simple text (no "SUCCESS" here; retrieval test will use vectorstore stub)
            return [FakeDoc("Methods: peptide FF assembled into fibers at pH 7.0 in water.")]

    monkeypatch.setattr(
        "src.pipeline.agent.PyPDFLoader", StubLoader, raising=True
    )

    # --- stub CharacterTextSplitter.split_documents -> identity or small split ---
    class StubSplitter:
        def __init__(self, chunk_size=1000, chunk_overlap=200):
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap
        def split_documents(self, docs):
            # Minimal: return exactly what we got (keep interface)
            return docs

    monkeypatch.setattr(
        "src.pipeline.agent.CharacterTextSplitter", StubSplitter, raising=True
    )

    # --- stub HuggingFaceEmbeddings (no network/model download) ---
    class StubEmbeddings:
        def __init__(self, model_name=None):
            self.model_name = model_name

    monkeypatch.setattr(
        "src.pipeline.agent.HuggingFaceEmbeddings", StubEmbeddings, raising=True
    )

    # --- FAISS.from_documents -> FakeVectorStore capturing docs ---
    def fake_from_documents(docs, _emb):
        # Allow tests to override the docs returned at retrieval time via holder.retrieval_docs
        return FakeVectorStore(holder.retrieval_docs if hasattr(holder, "retrieval_docs") else docs)

    monkeypatch.setattr(
        "src.pipeline.agent.FAISS.from_documents", staticmethod(fake_from_documents), raising=True
    )

    # --- Stub Gemini LLM to return deterministic content ---
    class StubLLM:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
        def __or__(self, other):
            # allow chaining; return the parser in tests
            return other

    # StrOutputParser simply returns input; we fake the chain ourselves below.
    # But to keep your `prompt | llm | StrOutputParser()` line working, we’ll stub the
    # whole pipeline by intercepting `generate_report`’s .invoke call via a tiny wrapper later.

    monkeypatch.setattr(
        "src.pipeline.agent.ChatGoogleGenerativeAI", StubLLM, raising=True
    )

    holder.FakeDoc = FakeDoc
    holder.FakeVectorStore = FakeVectorStore
    holder.FakeRetriever = FakeRetriever
    return holder
