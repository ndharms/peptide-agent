# tests/test_agent.py
import json
import types
import pytest

def test_load_data_builds_vectorstore(tmp_repo, patch_sys_path, stub_heavy_deps, monkeypatch):
    # Arrange: ensure FAISS returns vectorstore with the same docs passed in
    # Prepare that the VectorStore will hold these docs:
    stub_heavy_deps.retrieval_docs = [
        stub_heavy_deps.FakeDoc('{"PEPTIDE_CODE":"FF","OUTCOME":"SUCCESS"}'),
        stub_heavy_deps.FakeDoc("control doc without success"),
    ]

    # Act
    from src.pipeline.agent import load_data
    vs = load_data()

    # Assert
    # FakeVectorStore has .as_retriever; at least ensure it exists.
    assert hasattr(vs, "as_retriever")

def test_retrieve_docs_filters_success(tmp_repo, patch_sys_path, stub_heavy_deps, monkeypatch):
    # Arrange: Set retrieval docs with a mix of success/non-success
    stub_heavy_deps.retrieval_docs = [
        stub_heavy_deps.FakeDoc("... OUTCOME: SUCCESS ..."),
        stub_heavy_deps.FakeDoc("... failed attempt ..."),
        stub_heavy_deps.FakeDoc('{"OUTCOME":"SUCCESS","note":"ok"}'),
    ]

    # Act
    from src.pipeline.agent import retrieve_docs
    state = {"peptide_code": "FF", "retrieved_docs": [], "report": ""}
    new_state = retrieve_docs(state)

    # Assert: only docs containing "SUCCESS" string are retained
    assert len(new_state["retrieved_docs"]) == 2
    assert all("SUCCESS" in c for c in new_state["retrieved_docs"])

def test_generate_report_uses_schema_and_context(tmp_repo, patch_sys_path, stub_heavy_deps, monkeypatch):
    # Arrange: ensure some retrieved docs are present
    contexts = ["CONTEXT A SUCCESS", "CONTEXT B SUCCESS"]
    state = {"peptide_code": "FF", "retrieved_docs": contexts, "report": ""}

    # Intercept the prompt | llm | parser chain by stubbing StrOutputParser
    class StubParser:
        def __ror__(self, left):
            # Left is the LLM; we don't need it. `invoke` will be called on this parser.
            return self
        def invoke(self, variables):
            # variables contain: peptide_code, contexts, schema
            assert "peptide_code" in variables
            assert "contexts" in variables and all(s in variables["contexts"] for s in ["CONTEXT A", "CONTEXT B"])
            assert "schema" in variables and "TEMPERATURE_C" in variables["schema"]
            # Return a deterministic "report" string (your code expects a string)
            return (
                "Report for FF\n"
                "Suggested pH bucket: (6.5,7.5)\n"
                "Solvent: Water\n"
                "Temperature: 25"
            )

    monkeypatch.setattr("src.pipeline.agent.StrOutputParser", StubParser, raising=True)

    # Act
    from src.pipeline.agent import generate_report
    out_state = generate_report(state)

    # Assert
    assert "report" in out_state and "Report for FF" in out_state["report"]

def test_run_agent_integration(tmp_repo, patch_sys_path, stub_heavy_deps, monkeypatch):
    # Arrange: retrieval returns one SUCCESS doc so the flow continues
    stub_heavy_deps.retrieval_docs = [stub_heavy_deps.FakeDoc("SUCCESS: FF assembled")]

    # Stub StrOutputParser similarly to previous test
    class StubParser:
        def __ror__(self, left):  # allow llm | parser
            return self
        def invoke(self, variables):
            return "OK"

    monkeypatch.setattr("src.pipeline.agent.StrOutputParser", StubParser, raising=True)

    # Act
    from src.pipeline.agent import run_agent
    result = run_agent("FF")

    # Assert
    assert result == "OK"
