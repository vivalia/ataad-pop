"""Runtime configuration without machine-specific paths."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class AnalysisConfig:
    """Resolved paths and frozen analysis constants.

    Environment variables are optional conveniences; explicit CLI values should
    be preferred for controlled clinical-data runs.
    """

    data_path: Path
    feature_metadata: Path
    output_root: Path
    sheet_name: str = "main"
    seed: int = 20260814
    missingness_threshold: float = 0.30
    n_nearest_features: int = 40
    fixed_c: float = 0.01

    @classmethod
    def from_environment(cls) -> AnalysisConfig:
        root = repository_root()
        return cls(
            data_path=Path(os.getenv("ATAAD_POP_DATA", root / "data" / "TAAD_new1.xlsx")),
            feature_metadata=Path(
                os.getenv(
                    "ATAAD_POP_FEATURE_METADATA",
                    root / "config" / "feature_metadata.csv",
                )
            ),
            output_root=Path(os.getenv("ATAAD_POP_OUTPUT_ROOT", root)),
            sheet_name=os.getenv("ATAAD_POP_SHEET", "main"),
            seed=int(os.getenv("ATAAD_POP_SEED", "20260814")),
        )

    def as_environment(self) -> dict[str, str]:
        return {
            "ATAAD_POP_DATA": str(self.data_path),
            "ATAAD_POP_FEATURE_METADATA": str(self.feature_metadata),
            "ATAAD_POP_OUTPUT_ROOT": str(self.output_root),
            "ATAAD_POP_SHEET": self.sheet_name,
            "ATAAD_POP_SEED": str(self.seed),
        }

