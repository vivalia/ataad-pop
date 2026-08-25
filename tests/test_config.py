from pathlib import Path

from ataad_pop.config import AnalysisConfig, repository_root


def test_repository_root_contains_pyproject():
    assert (repository_root() / "pyproject.toml").is_file()


def test_environment_round_trip(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("ATAAD_POP_DATA", str(tmp_path / "input.csv"))
    monkeypatch.setenv("ATAAD_POP_OUTPUT_ROOT", str(tmp_path / "out"))
    config = AnalysisConfig.from_environment()
    assert config.data_path == tmp_path / "input.csv"
    assert config.output_root == tmp_path / "out"
    assert config.seed == 20260814

