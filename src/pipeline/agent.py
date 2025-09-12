from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
from typing import TypedDict, List
from langchain_core.documents import Document
from src.pipeline.schemas import peptide_output_schema
import json
import os
from pathlib import Path
import pandas as pd


TOP_K_DEFAULT = 10
EMBED_MODEL_DEFAULT = "sentence-transformers/all-MiniLM-L6-v2"
FAISS_CACHE_DIR = Path("data/index/faiss")
LLM_MODEL_DEFAULT = "gemini-1.5-pro"


def create_llm(model: str = LLM_MODEL_DEFAULT):
    return ChatGoogleGenerativeAI(
        model=model,
        temperature=0,
        google_api_key=os.getenv("GEMINI_API_KEY"),
    )


def load_json_examples(data_dir: Path) -> List[Document]:
    docs: List[Document] = []
    json_path = data_dir / "context_examples.json"
    if json_path.exists():
        with open(json_path, "r") as f:
            examples = json.load(f)
        for ex in examples:
            content = json.dumps(ex)
            docs.append(
                Document(
                    page_content=content,
                    metadata={"source": "context_examples", "type": "example"},
                )
            )
    return docs


def load_csv_examples(data_dir: Path) -> List[Document]:
    docs: List[Document] = []
    # Support both spellings if present
    candidates = [data_dir / "train_split.csv", data_dir / "train_splilt.csv"]
    csv_path = next((p for p in candidates if p.exists()), None)
    if csv_path is not None:
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            rec = dict(row)
            rec["OUTCOME"] = "SUCCESS"
            content = json.dumps(rec)
            docs.append(
                Document(
                    page_content=content,
                    metadata={"source": csv_path.name, "type": "train_example"},
                )
            )
    return docs


def load_papers(data_dir: Path) -> List[Document]:
    docs: List[Document] = []
    papers_dir = data_dir / "relevant_papers"
    if papers_dir.exists():
        for paper_dir in papers_dir.iterdir():
            if paper_dir.is_dir():
                metadata_path = paper_dir / "metadata.json"
                pdf_path = paper_dir / "paper.pdf"
                if metadata_path.exists() and pdf_path.exists():
                    with open(metadata_path, "r") as f:
                        metadata = json.load(f)
                    loader = PyPDFLoader(str(pdf_path))
                    pages = loader.load()
                    for page in pages:
                        peptide_codes = metadata.get("PEPTIDE_CODE", [])
                        doi = metadata.get("DOI", "")
                        page.metadata.update(
                            {
                                "peptide_codes": peptide_codes,
                                "doi": doi,
                                "source": str(pdf_path),
                            }
                        )
                    docs.extend(pages)
    return docs


def get_vectorstore(
    data_dir: Path,
    embed_model_name: str = EMBED_MODEL_DEFAULT,
    cache_dir: Path = FAISS_CACHE_DIR,
    refresh: bool = False,
) -> FAISS:
    cache_dir.mkdir(parents=True, exist_ok=True)
    embeddings = HuggingFaceEmbeddings(model_name=embed_model_name)
    index_faiss = cache_dir / "index.faiss"
    index_pkl = cache_dir / "index.pkl"
    if not refresh and index_faiss.exists() and index_pkl.exists():
        return FAISS.load_local(
            str(cache_dir), embeddings, allow_dangerous_deserialization=True
        )
    # Build from scratch
    docs: List[Document] = []
    docs.extend(load_csv_examples(data_dir))
    docs.extend(load_json_examples(data_dir))
    docs.extend(load_papers(data_dir))
    if not docs:
        docs = [Document(page_content="No data", metadata={"source": "generated"})]
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents(docs)
    vectorstore = FAISS.from_documents(splits, embeddings)
    vectorstore.save_local(str(cache_dir))
    return vectorstore


class AgentState(TypedDict):
    peptide_code: str
    target_structural_assembly: str
    top_k: int
    llm_model: str
    refresh_index: bool
    retrieved_docs: List[str]
    report: str


# Deprecated: use get_vectorstore() instead.


def retrieve_docs(state: AgentState):
    """Retrieve relevant documents for the peptide code and target structural assembly."""
    vectorstore = get_vectorstore(Path("data"), refresh=state.get("refresh_index", False))
    k = state.get("top_k", TOP_K_DEFAULT)
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    query = f"{state['peptide_code']} {state['target_structural_assembly']}"
    docs = retriever.invoke(query)

    contents: List[str] = [doc.page_content for doc in docs]
    state["retrieved_docs"] = contents
    return state


def generate_report(state: AgentState):
    """Generate report using the configured LLM."""
    llm = create_llm(model=state.get("llm_model", LLM_MODEL_DEFAULT))

    prompt = ChatPromptTemplate.from_template(
        """
    Given:
    - PEPTIDE_CODE: {peptide_code}
    - TARGET_STRUCTURAL_ASSEMBLY: {target_structural_assembly}

    Relevant contexts (papers and successful examples):
    {contexts}

    Produce a concise, evidence-grounded report recommending optimal experimental
    conditions for synthesis toward the target structural assembly:
    - pH
    - Concentration (log M)
    - Temperature (C)
    - Solvent
    - Estimated Time (minutes)

    Use this schema as a guide: {schema}
    """
    )

    chain = prompt | llm | StrOutputParser()

    contexts = "\n\n".join(state.get("retrieved_docs", []))
    schema_str = str(peptide_output_schema.schema)

    report = chain.invoke(
        {
            "peptide_code": state["peptide_code"],
            "target_structural_assembly": state["target_structural_assembly"],
            "contexts": contexts,
            "schema": schema_str,
        }
    )
    state["report"] = report
    return state


# Build the graph
workflow = StateGraph(AgentState)
workflow.add_node("retrieve", retrieve_docs)
workflow.add_node("generate", generate_report)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

app = workflow.compile()


def run_agent(
    peptide_code: str,
    target_structural_assembly: str,
    top_k: int = TOP_K_DEFAULT,
    llm_model: str = LLM_MODEL_DEFAULT,
    refresh_index: bool = False,
) -> str:
    """Run the agent to generate report."""
    initial_state = {
        "peptide_code": peptide_code,
        "target_structural_assembly": target_structural_assembly,
        "top_k": top_k,
        "llm_model": llm_model,
        "refresh_index": refresh_index,
        "retrieved_docs": [],
        "report": "",
    }
    result = app.invoke(initial_state)
    return result["report"]
