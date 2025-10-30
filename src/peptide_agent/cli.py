import json
import logging
import sys
from pathlib import Path

import typer

from .config import Settings
from .indexing.faiss_store import build_or_load_vectorstore
from .runner.main import PeptideAgentError, predict_batch, predict_single


# Configure logging
def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the application."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stderr)],
    )


app = typer.Typer(help="Peptide agent CLI")
logger = logging.getLogger(__name__)


@app.command()
def index(
    config: str | None = typer.Option(None, "--config", "-c", help="Path to YAML config"),
    refresh: bool = typer.Option(False, "--refresh", help="Force rebuild of FAISS index"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
):
    """Build or load the FAISS vector index for retrieval."""
    setup_logging(verbose)
    logger.info("Starting index command")

    try:
        settings = Settings.from_yaml(config)
        logger.info(f"Using data_dir: {settings.data_dir}")
        logger.info(f"Using faiss_cache_dir: {settings.faiss_cache_dir}")

        build_or_load_vectorstore(
            data_dir=Path(settings.data_dir),
            cache_dir=Path(settings.faiss_cache_dir),
            embed_model_name=settings.embed_model_name,
            refresh=refresh,
        )
        typer.echo("✓ Index ready.")
        logger.info("Index command completed successfully")
    except Exception as e:
        logger.error(f"Index command failed: {e}", exc_info=verbose)
        typer.echo(f"✗ Error: {e}", err=True)
        raise typer.Exit(code=1) from None


@app.command()
def predict(
    peptide_code: str | None = typer.Option(
        None, "--peptide-code", "-p", help="Peptide code (e.g., FF)"
    ),
    target_structural_assembly: str | None = typer.Option(
        None, "--target-structural-assembly", "-t", help="Target morphology/assembly"
    ),
    input_json: str | None = typer.Option(
        None, "--input-json", help="Path to JSON list for batch predictions"
    ),
    config: str | None = typer.Option(None, "--config", "-c", help="Path to YAML config"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
):
    """Generate peptide synthesis predictions."""
    setup_logging(verbose)
    logger.info("Starting predict command")

    try:
        settings = Settings.from_yaml(config)

        if input_json:
            logger.info(f"Running batch prediction from: {input_json}")
            try:
                input_path = Path(input_json)
                if not input_path.exists():
                    raise FileNotFoundError(f"Input file not found: {input_json}")

                items = json.loads(input_path.read_text(encoding="utf-8"))
                logger.info(f"Loaded {len(items)} items for prediction")
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in input file: {e}")
                typer.echo(f"✗ Error: Invalid JSON in {input_json}: {e}", err=True)
                raise typer.Exit(code=1) from None
            except FileNotFoundError as e:
                logger.error(str(e))
                typer.echo(f"✗ Error: {e}", err=True)
                raise typer.Exit(code=1) from None

            try:
                reports = predict_batch(items, settings)
                typer.echo(json.dumps(reports, ensure_ascii=False, indent=2))
                logger.info(f"Batch prediction completed: {len(reports)} reports")
            except PeptideAgentError as e:
                logger.error(f"Batch prediction failed: {e}", exc_info=verbose)
                typer.echo(f"✗ Error: {e}", err=True)
                raise typer.Exit(code=1) from None
            return

        if not peptide_code or not target_structural_assembly:
            msg = "Provide --peptide-code and --target-structural-assembly, or --input-json for batch mode."
            logger.error(msg)
            typer.echo(f"✗ Error: {msg}", err=True)
            raise typer.Exit(code=1) from None

        logger.info(f"Running single prediction: {peptide_code} -> {target_structural_assembly}")
        try:
            report = predict_single(peptide_code, target_structural_assembly, settings)
            typer.echo(report)
            logger.info("Single prediction completed")
        except PeptideAgentError as e:
            logger.error(f"Single prediction failed: {e}", exc_info=verbose)
            typer.echo(f"✗ Error: {e}", err=True)
            raise typer.Exit(code=1) from None

    except Exception as e:
        logger.error(f"Predict command failed: {e}", exc_info=verbose)
        typer.echo(f"✗ Unexpected error: {e}", err=True)
        raise typer.Exit(code=1) from None


if __name__ == "__main__":
    app()
