import logging
from pathlib import Path

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)


def build_or_load_vectorstore(
    data_dir: Path,
    cache_dir: Path,
    embed_model_name: str,
    refresh: bool = False,
) -> FAISS:
    """Build or load FAISS vectorstore with error handling and logging.

    Args:
        data_dir: Directory containing training data
        cache_dir: Directory for FAISS index cache
        embed_model_name: Name of embedding model to use
        refresh: Whether to force rebuild of index

    Returns:
        FAISS vectorstore instance

    Raises:
        Exception: If vectorstore build/load fails
    """
    logger.info(f"Building/loading vectorstore (refresh={refresh})")
    logger.info(f"Data dir: {data_dir}")
    logger.info(f"Cache dir: {cache_dir}")

    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error(f"Failed to create cache directory: {e}")
        raise

    try:
        logger.info(f"Loading embedding model: {embed_model_name}")
        embeddings = HuggingFaceEmbeddings(model_name=embed_model_name)
    except Exception as e:
        logger.error(f"Failed to load embedding model: {e}")
        raise

    index_faiss = cache_dir / "index.faiss"
    index_pkl = cache_dir / "index.pkl"

    if not refresh and index_faiss.exists() and index_pkl.exists():
        logger.info("Loading existing FAISS index from cache")
        try:
            # NOTE: allow_dangerous_deserialization is needed for pickle files
            # Only use this with trusted data sources
            vectorstore = FAISS.load_local(
                str(cache_dir), embeddings, allow_dangerous_deserialization=True
            )
            logger.info("Successfully loaded cached index")
            return vectorstore
        except Exception as e:
            logger.warning(f"Failed to load cached index: {e}, rebuilding...")
            # Fall through to rebuild

    logger.info("Building new FAISS index")
    try:
        docs: list[Document] = _load_context_documents(data_dir)
        if not docs:
            logger.warning("No documents loaded, creating placeholder")
            docs = [Document(page_content="No data", metadata={"source": "generated"})]

        logger.info(f"Loaded {len(docs)} document(s)")

        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = splitter.split_documents(docs)
        logger.info(f"Split into {len(splits)} chunk(s)")

        vectorstore = FAISS.from_documents(splits, embeddings)
        logger.info("Created FAISS vectorstore")

        vectorstore.save_local(str(cache_dir))
        logger.info(f"Saved index to {cache_dir}")

        return vectorstore
    except Exception as e:
        logger.error(f"Failed to build vectorstore: {e}")
        raise


def _load_context_documents(data_dir: Path) -> list[Document]:
    """Load context documents from data directory with error handling.

    Args:
        data_dir: Directory containing training data

    Returns:
        List of Document objects
    """
    import json

    import pandas as pd

    docs: list[Document] = []

    # CSV examples
    candidates = [data_dir / "train_split.csv"]
    csv_path = next((p for p in candidates if p.exists()), None)

    if csv_path is None:
        logger.warning(f"No training CSV found in {data_dir}")
        return docs

    logger.info(f"Loading training data from: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded CSV with {len(df)} rows")

        for idx, row in df.iterrows():
            try:
                rec = dict(row)
                rec["OUTCOME"] = "SUCCESS"
                content = json.dumps(rec)
                docs.append(
                    Document(
                        page_content=content,
                        metadata={"source": csv_path.name, "type": "train_example"},
                    )
                )
            except Exception as e:
                logger.warning(f"Failed to process row {idx}: {e}")
                continue

        logger.info(f"Successfully loaded {len(docs)} document(s)")
    except Exception as e:
        logger.error(f"Failed to read CSV file {csv_path}: {e}")
        raise

    return docs
