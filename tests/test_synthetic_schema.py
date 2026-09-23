from pathlib import Path

import pandas as pd

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


def test_validation_ignores_metadata_and_input_column_whitespace(tmp_path: Path):
    metadata = pd.DataFrame(
        {
            " Variable ": [" Age "],
            " Domain ": ["phenotype"],
            " Imputation_type ": ["continuous"],
            " Missing_indicator_for_sensitivity ": [False],
        }
    )
    metadata_path = tmp_path / "metadata.csv"
    metadata.to_csv(metadata_path, index=False)

    loaded = load_feature_metadata(metadata_path)
    frame = pd.DataFrame(
        {
            " Date_of_surgery ": ["2024-01-01", "2024-01-02"],
            " Surgeon ": [1, 2],
            " Missingctimage ": [0, 0],
            " Bentall_mechanic_valve ": [1, 0],
            " Bentall_bio_valve ": [0, 1],
            " Age ": [50, 60],
        }
    )

    assert validate_input(frame, loaded) == []
