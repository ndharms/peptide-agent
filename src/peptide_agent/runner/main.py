import json
import logging
import os
from importlib.resources import files
from pathlib import Path
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from ..config import Settings
from ..indexing.faiss_store import build_or_load_vectorstore
from ..schemas import PeptideSynthesisConditions, peptide_output_schema

# Configure logger
logger = logging.getLogger(__name__)


class PeptideAgentError(Exception):
    """Base exception for peptide agent errors."""

    pass


class APIError(PeptideAgentError):
    """Error related to API calls."""

    pass


class ValidationError(PeptideAgentError):
    """Error related to data validation."""

    pass


def _create_llm(model: str) -> ChatGoogleGenerativeAI:
    """Create LLM instance with error checking."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.error("GEMINI_API_KEY environment variable not set")
        raise APIError("GEMINI_API_KEY environment variable is required")

    logger.info(f"Creating LLM with model: {model}")
    try:
        return ChatGoogleGenerativeAI(
            model=model,
            temperature=0,
            google_api_key=api_key,
        )
    except Exception as e:
        logger.error(f"Failed to create LLM: {e}")
        raise APIError(f"Failed to initialize LLM: {e}") from e


def _escape_braces(text: str) -> str:
    # Escape all braces in the base prompt so they are treated as literals,
    # not variables by LangChain's f-string-style templates.
    return text.replace("{", "{{").replace("}", "}}")


def _load_base_prompt() -> str:
    """Load base prompt from package resources with error handling."""
    try:
        prompt_path = files("peptide_agent.prompts") / "prompt.md"
        prompt_text = prompt_path.read_text(encoding="utf-8")
        logger.debug(f"Loaded base prompt ({len(prompt_text)} characters)")
        return prompt_text
    except Exception as e:
        logger.error(f"Failed to load base prompt: {e}")
        raise PeptideAgentError(f"Failed to load base prompt: {e}") from e


def _retrieve_contexts(items: list[dict[str, str]], settings: Settings) -> list[list[str]]:
    """Retrieve relevant contexts from vectorstore with error handling."""
    logger.info(f"Retrieving contexts for {len(items)} item(s)")

    try:
        vectorstore = build_or_load_vectorstore(
            data_dir=Path(settings.data_dir),
            cache_dir=Path(settings.faiss_cache_dir),
            embed_model_name=settings.embed_model_name,
            refresh=False,
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": settings.top_k})
    except Exception as e:
        logger.error(f"Failed to build/load vectorstore: {e}")
        raise PeptideAgentError(f"Failed to initialize retrieval system: {e}") from e

    queries = [f"{it['peptide_code']} {it['target_structural_assembly']}" for it in items]
    logger.debug(f"Generated {len(queries)} query/queries")

    contents_per_query: list[list[str]] = []
    try:
        if hasattr(retriever, "batch"):
            logger.debug("Using batch retrieval")
            docs_per_query = retriever.batch(queries)  # type: ignore[attr-defined]
            for docs in docs_per_query:
                contents_per_query.append([d.page_content for d in docs])
        else:
            logger.debug("Using sequential retrieval")
            for q in queries:
                docs = retriever.invoke(q)
                contents_per_query.append([d.page_content for d in docs])

        logger.info(f"Retrieved contexts: {[len(c) for c in contents_per_query]}")
        return contents_per_query
    except Exception as e:
        logger.error(f"Failed to retrieve contexts: {e}")
        raise PeptideAgentError(f"Failed to retrieve contexts: {e}") from e


def predict_single(peptide_code: str, target_structural_assembly: str, settings: Settings) -> str:
    """Generate synthesis prediction for a single peptide with error handling.

    Args:
        peptide_code: Peptide sequence code (e.g., 'FF')
        target_structural_assembly: Target morphology (e.g., 'nanofibers')
        settings: Configuration settings

    Returns:
        Prediction report as string

    Raises:
        PeptideAgentError: If prediction fails
    """
    logger.info(f"Predicting single: {peptide_code} -> {target_structural_assembly}")

    try:
        contexts = _retrieve_contexts(
            items=[
                {
                    "peptide_code": peptide_code,
                    "target_structural_assembly": target_structural_assembly,
                }
            ],
            settings=settings,
        )[0]
    except Exception as e:
        logger.error(f"Context retrieval failed: {e}")
        raise

    try:
        llm = _create_llm(settings.llm_model)
    except Exception as e:
        logger.error(f"LLM creation failed: {e}")
        raise

    base_prompt = _escape_braces(_load_base_prompt())
    single_output_block = """
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

    try:
        prompt = ChatPromptTemplate.from_template(base_prompt + "\n\n" + single_output_block)
        chain = prompt | llm | StrOutputParser()

        schema_str = str(peptide_output_schema.schema)
        logger.debug("Invoking LLM chain")
        report = chain.invoke(
            {
                "peptide_code": peptide_code,
                "target_structural_assembly": target_structural_assembly,
                "contexts": "\n\n".join(contexts),
                "schema": schema_str,
            }
        )
        logger.info(f"Generated report ({len(report)} characters)")

        # Try to validate the report
        try:
            PeptideSynthesisConditions.from_report_string(report)
            logger.debug("Report validation successful")
        except Exception as e:
            logger.warning(f"Report validation failed (non-critical): {e}")

        return report
    except Exception as e:
        logger.error(f"LLM invocation failed: {e}")
        raise APIError(f"Failed to generate prediction: {e}") from e


essential_output_block = """
# Output formatting

STRICT OUTPUT FORMAT FOR "report" (exactly 5 lines, in this order; no extra text):
PH: (a,b) or (a,b] or [a,b) or [a,b]
Concentration (log M): (a,b) or (a,b] or [a,b) or [a,b]
Temperature (C): (a,b) or (a,b] or [a,b) or [a,b]
Solvent: <single word or phrase>
Estimated Time (minutes): (a,b) or (a,b] or [a,b) or [a,b]

Formatting rules:
- Use only numeric endpoints for intervals (integers or decimals).
- Use parentheses/brackets to indicate open/closed bounds as shown.
- Do not use symbols like < or >; always provide two numeric endpoints.
- Do not include any extra commentary, bullet points, headers, or blank lines.
- The labels must match exactly.
"""


def predict_batch(requests: list[dict[str, str]], settings: Settings) -> list[str]:
    """Generate synthesis predictions for multiple peptides with error handling.

    Args:
        requests: List of dicts with 'peptide_code' and 'target_structural_assembly'
        settings: Configuration settings

    Returns:
        List of prediction reports as strings

    Raises:
        PeptideAgentError: If batch prediction fails
    """
    logger.info(f"Predicting batch of {len(requests)} items")

    if not requests:
        logger.warning("Empty batch request")
        return []

    try:
        contexts_lists = _retrieve_contexts(items=requests, settings=settings)
    except Exception as e:
        logger.error(f"Batch context retrieval failed: {e}")
        raise

    items_for_llm: list[dict[str, Any]] = []
    for idx, (req, ctxs) in enumerate(zip(requests, contexts_lists)):
        items_for_llm.append(
            {
                "id": idx,
                "peptide_code": req["peptide_code"],
                "target_structural_assembly": req["target_structural_assembly"],
                "contexts": "\n\n".join(ctxs),
            }
        )

    logger.debug(f"Prepared {len(items_for_llm)} items for LLM")

    try:
        llm = _create_llm(settings.llm_model)
    except Exception as e:
        logger.error(f"Batch LLM creation failed: {e}")
        raise

    base_prompt = _escape_braces(_load_base_prompt())
    batch_appendix = """
Return ONLY a valid JSON array of objects with:
[
  {"id": <int>, "report": "<string>"},
  ...
]

Do not include any explanations outside the JSON.

You will be provided a list of items to predict on.
For each item, generate a self-assembly report using this schema as a guide: {schema}

Evaluate these items (as JSON):
{items_json}
"""

    try:
        prompt = ChatPromptTemplate.from_template(
            base_prompt + "\n\n" + essential_output_block + "\n\n" + batch_appendix
        )
        chain = prompt | llm | StrOutputParser()
        schema_str = str(peptide_output_schema.schema)
        items_json = json.dumps(items_for_llm, ensure_ascii=False)

        logger.debug("Invoking LLM chain for batch")
        output = chain.invoke({"schema": schema_str, "items_json": items_json})
        logger.info(f"Received batch response ({len(output)} characters)")
    except Exception as e:
        logger.error(f"Batch LLM invocation failed: {e}")
        raise APIError(f"Failed to generate batch predictions: {e}") from e

    def _extract_json_array(text: str) -> Any:
        """Extract JSON array from LLM output with fallback parsing."""
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            logger.warning(f"Initial JSON parse failed: {e}, trying regex extraction")
            import re as _re

            m = _re.search(r"\[\s*{.*}\s*\]", text, flags=_re.DOTALL)
            if not m:
                logger.error("Could not extract JSON array from response")
                raise ValidationError("Failed to parse JSON from LLM response") from e
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError as e2:
                logger.error(f"Regex-extracted JSON also invalid: {e2}")
                raise ValidationError("Failed to parse extracted JSON") from e2

    try:
        data = _extract_json_array(output)
    except Exception as e:
        logger.error(f"JSON extraction failed: {e}")
        # Return empty reports on parse failure
        return [""] * len(requests)

    by_id: dict[int, str] = {}
    for obj in data:
        if isinstance(obj, dict) and "id" in obj and "report" in obj:
            try:
                by_id[int(obj["id"])] = str(obj["report"])
            except Exception as e:
                logger.warning(f"Failed to process response object: {e}")
                continue

    logger.debug(f"Extracted {len(by_id)} reports from response")

    reports: list[str] = [by_id.get(i, "") for i in range(len(requests))]
    logger.info(f"Returning {len(reports)} reports")
    return reports
