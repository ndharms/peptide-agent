import typer
from .agent import run_agent


app = typer.Typer()


@app.command()
def generate_report(peptide_code: str):
    """Generate report on optimal experimental conditions for the given peptide code."""
    try:
        report = run_agent(peptide_code)
        print(report)
    except Exception as e:
        typer.echo(f"Error generating report: {e}", err=True)
        raise


if __name__ == "__main__":
    app()
