import typer
from .agent import run_agent


app = typer.Typer()


@app.command()
def generate_report(
    peptide_code: str,
    target_structural_assembly: str,
    top_k: int = typer.Option(10, help="Number of docs to retrieve"),
    llm_model: str = typer.Option("gemini-1.5-pro", help="LLM model id"),
    refresh_index: bool = typer.Option(False, help="Rebuild FAISS index cache"),
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


def main():
    app()


if __name__ == "__main__":
    main()
