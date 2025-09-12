import typer
from .agent import run_agent


app = typer.Typer()


@app.command(name="generate-report")
def generate_report(
    peptide_code: str = typer.Option(
        ...,
        "--peptide-code",
        "--peptide_code",
        "-p",
        help="Peptide code identifier (e.g., 'FF')",
    ),
    target_structural_assembly: str = typer.Option(
        ...,
        "--target-structural-assembly",
        "--target_structural_assembly",
        "-t",
        help="Target structural assembly (e.g., 'beta-sheet')",
    ),
    top_k: int = typer.Option(10, "--top-k", help="Number of docs to retrieve"),
    llm_model: str = typer.Option("gemini-1.5-pro", "--llm-model", help="LLM model id"),
    refresh_index: bool = typer.Option(
        False, "--refresh-index", help="Rebuild FAISS index cache"
    ),
):
    """Generate report for peptide_code and target_structural_assembly."""
    try:
        report = run_agent(
            peptide_code=peptide_code,
            target_structural_assembly=target_structural_assembly,
            top_k=top_k,
            llm_model=llm_model,
            refresh_index=refresh_index,
        )
        print(report)
    except Exception as e:
        typer.echo(f"Error generating report: {e}", err=True)
        raise


# Command alias to support underscore variant: `generate_report`
@app.command(name="generate_report")
def generate_report_alias(
    peptide_code: str = typer.Option(
        ...,
        "--peptide-code",
        "--peptide_code",
        "-p",
        help="Peptide code identifier (e.g., 'FF')",
    ),
    target_structural_assembly: str = typer.Option(
        ...,
        "--target-structural-assembly",
        "--target_structural_assembly",
        "-t",
        help="Target structural assembly (e.g., 'beta-sheet')",
    ),
    top_k: int = typer.Option(10, "--top-k", help="Number of docs to retrieve"),
    llm_model: str = typer.Option("gemini-1.5-pro", "--llm-model", help="LLM model id"),
    refresh_index: bool = typer.Option(
        False, "--refresh-index", help="Rebuild FAISS index cache"
    ),
):
    return generate_report(
        peptide_code=peptide_code,
        target_structural_assembly=target_structural_assembly,
        top_k=top_k,
        llm_model=llm_model,
        refresh_index=refresh_index,
    )


def main():
    app()


if __name__ == "__main__":
    main()

