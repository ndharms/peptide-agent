import json

import pytest
from langchain_core.runnables import RunnableLambda

import peptide_agent.runner.main as runner
from peptide_agent.config import Settings


class FakeDoc:
    def __init__(self, content: str):
        self.page_content = content


class FakeRetriever:
    def __init__(self, docs_per_query):
        self._docs_per_query = docs_per_query

    def invoke(self, _query):
        return [FakeDoc("ctx-1"), FakeDoc("ctx-2")]

    def batch(self, _queries):
        return [[FakeDoc("ctx-a"), FakeDoc("ctx-b")]] * len(_queries)


class FakeVectorStore:
    def as_retriever(self, search_kwargs=None):  # noqa: ARG002
        return FakeRetriever(docs_per_query=["dummy"])


@pytest.fixture(autouse=True)
def patch_vectorstore(monkeypatch):
    monkeypatch.setattr(
        runner,
        "build_or_load_vectorstore",
        lambda *a, **k: FakeVectorStore(),  # noqa: ANN001, ANN002
    )

    class DummyPrompt:
        @staticmethod
        def from_template(template):  # noqa: ARG004
            return RunnableLambda(lambda inputs: "PROMPT")

    monkeypatch.setattr(runner, "ChatPromptTemplate", DummyPrompt)


def test_predict_single(monkeypatch):
    monkeypatch.setattr(runner, "_create_llm", lambda model: RunnableLambda(lambda _: "REPORT"))
    settings = Settings()
    out = runner.predict_single("FF", "nanofibers", settings)
    assert isinstance(out, str)
    assert out  # non-empty


def test_predict_batch(monkeypatch):
    json_array = json.dumps(
        [
            {
                "id": 0,
                "report": "PH: (5,6)\nConcentration (log M): (0,1)\nTemperature (C): (20,25)\nSolvent: Water\nEstimated Time (minutes): (60,120)",
            },
            {
                "id": 1,
                "report": "PH: (6,7)\nConcentration (log M): (1,2)\nTemperature (C): (20,25)\nSolvent: PBS\nEstimated Time (minutes): (0,30)",
            },
        ]
    )
    monkeypatch.setattr(
        runner,
        "_create_llm",
        lambda model: RunnableLambda(lambda _: json_array),
    )
    settings = Settings()
    requests = [
        {"peptide_code": "FF", "target_structural_assembly": "nanofibers"},
        {"peptide_code": "YY", "target_structural_assembly": "hydrogel"},
    ]
    reports = runner.predict_batch(requests, settings)
    assert len(reports) == 2
    assert all(isinstance(r, str) and r for r in reports)
