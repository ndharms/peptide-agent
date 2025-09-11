from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
from typing import TypedDict, List
from src.pipeline.schema import peptide_output_schema
import json
import os
from pathlib import Path


class AgentState(TypedDict):
    peptide_code: str
    retrieved_docs: List[str]
    report: str


def load_data():
    """Load curated data from data/ folder."""
    data_dir = Path("data")

    # Load JSON examples
    json_docs = []
    with open(data_dir / "context_examples.json", "r") as f:
        examples = json.load(f)
    for ex in examples:
        content = json.dumps(ex)
        json_docs.append({"page_content": content, "metadata": {"source": "context_examples"}})

    # Load relevant papers
    paper_docs = []
    papers_dir = data_dir / "relevant_papers"
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
                    page.metadata.update({"peptide_codes": peptide_codes, "doi": doi})
                paper_docs.extend(pages)

    all_docs = json_docs + paper_docs

    # Split and embed
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(all_docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(splits, embeddings)
    return vectorstore


def retrieve_docs(state: AgentState):
    """Retrieve relevant documents for the peptide code."""
    vectorstore = load_data()  # Load once, but for simplicity reload
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    docs = retriever.invoke(state["peptide_code"])
    # Filter for successful outcomes where possible
    relevant_content = []
    for doc in docs:
        content = doc.page_content
        if "SUCCESS" in content:
            relevant_content.append(content)
    state["retrieved_docs"] = relevant_content
    return state


def generate_report(state: AgentState):
    """Generate report using Gemini."""
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro", temperature=0, google_api_key=os.getenv("GEMINI_API_KEY")
    )

    prompt = ChatPromptTemplate.from_template(
        """
    Based on the following successful experimental contexts for PEPTIDE_CODE: {peptide_code}

    Contexts:
    {contexts}

    Generate a report on optimal experimental conditions for synthesis:
    - pH
    - Concentration (log M)
    - Temperature (C)
    - Solvent
    - Estimated Time (hours)

    Structure the report using this schema: {schema}
    Focus on conditions from successful outcomes.
    """
    )

    chain = prompt | llm | StrOutputParser()

    contexts = "\n\n".join(state["retrieved_docs"])
    schema_str = str(peptide_output_schema.schema)

    report = chain.invoke(
        {"peptide_code": state["peptide_code"], "contexts": contexts, "schema": schema_str}
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


def run_agent(peptide_code: str) -> str:
    """Run the agent to generate report."""
    initial_state = {"peptide_code": peptide_code, "retrieved_docs": [], "report": ""}
    result = app.invoke(initial_state)
    return result["report"]
