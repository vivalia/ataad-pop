from pathlib import Path

from ataad_pop.privacy import scan_public_tree


def test_restricted_workbook_is_blocked(tmp_path: Path):
    path = tmp_path / "raw.xlsx"
    path.write_bytes(b"not a real workbook")
    findings = scan_public_tree(tmp_path)
    assert any(finding.path == Path("raw.xlsx") for finding in findings)


def test_patient_prediction_csv_is_blocked(tmp_path: Path):
    path = tmp_path / "patient_outputs.csv"
    path.write_text("patient_id,prediction\n1,0.5\n", encoding="utf-8")
    findings = scan_public_tree(tmp_path)
    assert findings


def test_aggregate_metrics_are_allowed(tmp_path: Path):
    path = tmp_path / "metrics.csv"
    path.write_text("model,auroc,brier\nL2,0.81,0.15\n", encoding="utf-8")
    assert scan_public_tree(tmp_path) == []

