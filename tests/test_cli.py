import json

from typer.testing import CliRunner

from peptide_agent.cli import app

runner = CliRunner()


def test_cli_index(monkeypatch, tmp_path):
    # Avoid real indexing
    from peptide_agent import cli as cli_mod

    monkeypatch.setattr(cli_mod, "build_or_load_vectorstore", lambda *a, **k: object())
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("data_dir: ./data\nfaiss_cache_dir: ./data/index/faiss\n")

    result = runner.invoke(app, ["index", "-c", str(cfg)])
    assert result.exit_code == 0
    assert "Index ready" in result.stdout


def test_cli_predict_single(monkeypatch):
    from peptide_agent import cli as cli_mod

    monkeypatch.setattr(cli_mod, "predict_single", lambda *a, **k: "REPORT")
    result = runner.invoke(app, ["predict", "-p", "FF", "-t", "nanofibers"])
    assert result.exit_code == 0
    assert "REPORT" in result.stdout


def test_cli_predict_batch(monkeypatch, tmp_path):
    from peptide_agent import cli as cli_mod

    monkeypatch.setattr(cli_mod, "predict_batch", lambda items, settings: ["R1", "R2"])  # noqa: ARG005
    payload = [
        {"peptide_code": "FF", "target_structural_assembly": "nanofibers"},
        {"peptide_code": "YY", "target_structural_assembly": "hydrogel"},
    ]
    f = tmp_path / "in.json"
    f.write_text(json.dumps(payload))

    result = runner.invoke(app, ["predict", "--input-json", str(f)])
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert data == ["R1", "R2"]
