from pathlib import Path

from ataad_pop.schema import derive_bentall, load_feature_metadata, validate_input
from ataad_pop.synthetic import generate_synthetic


ROOT = Path(__file__).resolve().parents[1]
METADATA = ROOT / "config" / "feature_metadata.csv"


def test_synthetic_data_satisfy_public_schema():
    metadata = load_feature_metadata(METADATA)
    frame = generate_synthetic(METADATA, n=180, seed=7)
    assert validate_input(frame, metadata) == []
    outcome = derive_bentall(frame)
    assert len(outcome) == 180
    assert set(outcome.unique()).issubset({0, 1})
    assert 0 < int(outcome.sum()) < len(outcome)


def test_synthetic_data_are_deterministic():
    left = generate_synthetic(METADATA, n=120, seed=31)
    right = generate_synthetic(METADATA, n=120, seed=31)
    assert left.equals(right)

