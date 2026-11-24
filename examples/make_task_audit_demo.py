"""Create a fictional three-task dataset exercising the POP audit."""

import argparse
from pathlib import Path

import numpy as np

from ataad_pop.synthetic import generate_synthetic


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--metadata", type=Path, default=Path("config/feature_metadata.csv"))
    args = parser.parse_args()
    frame = generate_synthetic(args.metadata, n=360, seed=42)
    rng = np.random.default_rng(43)
    surgeon = frame["Surgeon"].to_numpy()
    eligible = rng.random(len(frame)) < 0.78
    branch = rng.random(len(frame)) < np.where(surgeon <= 3, 0.85, 0.22)
    frame["Totalarch_fourbranch"] = (eligible & branch).astype(int)
    frame["Totalarch_island"] = (eligible & ~branch).astype(int)
    indication = rng.random(len(frame)) < 0.16
    dissection = rng.random(len(frame)) < 0.55
    salvage = rng.random(len(frame)) < 0.04
    frame["CABG_for_dissection"] = (indication & dissection).astype(int)
    frame["CABG_for_CAD"] = (indication & ~dissection).astype(int)
    frame["Salvage_CABG_after_declamp"] = salvage.astype(int)
    frame["CABG"] = (indication | salvage).astype(int)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
