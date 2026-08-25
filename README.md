# ATAAD-POP

[![CI](https://github.com/vivalia/ataad-pop/actions/workflows/ci.yml/badge.svg)](https://github.com/vivalia/ataad-pop/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)

Transportability-aware preoperative operative-decision modeling for acute type A aortic dissection (ATAAD) using the **Phenotype–Observation–Practice (POP)** framework.

This repository contains the leakage-controlled analysis code supporting a retrospective study of recorded Bentall operative decisions. POP separates strictly preoperative phenotype, observation-process indicators, and practice context before model development; evaluates transport across held-out surgeon and calendar-year environments; nests stochastic multiple imputation and training-only recalibration inside every outer fold; and exposes missing-data uncertainty as an explicit defer signal.

## What this repository is—and is not

- It is a reproducible research implementation for auditing signal provenance, environment transportability, calibration, uncertainty, missingness shift, and clinical-domain dependence.
- It predicts concordance with a **recorded historical operation**. It does not estimate counterfactual treatment benefit and must not be used as an autonomous surgical recommendation system.
- It contains no raw clinical records and no patient-level predictions. The source cohort is controlled-access because it contains sensitive health information.

## Main analyses

1. Phenotype-only and phenotype-plus-observation L2-logistic models.
2. Leave-one-surgeon-out and rolling-year validation.
3. Twenty stochastic training-fold imputations with prediction-level pooling.
4. Training-only probability recalibration.
5. Environment-stratified paired bootstrap confidence intervals.
6. Multiple-imputation-aware selective prediction.
7. Missingness-shift stress tests, including complete loss of echocardiography.
8. Clinical-domain ablation and domain-only information analysis.
9. Fixed-hyperparameter elastic-net, random-forest, and LightGBM benchmark.
10. Prespecified subgroup and small-environment sensitivity analyses.

## Repository layout

```text
analysis/                 Frozen manuscript analysis scripts
config/                   Non-patient feature/channel metadata
docs/                     Methods, data-access, and reproducibility notes
examples/                 Synthetic-data workflow
results/aggregated/       Publishable aggregate tables only
src/ataad_pop/            CLI, configuration, schema, privacy, synthetic data
tests/                    Unit and privacy tests
```

## Installation

Python 3.11 or newer is recommended.

```bash
git clone https://github.com/vivalia/ataad-pop.git
cd ataad-pop
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev,benchmark]'
```

## Quick start with synthetic data

The synthetic example validates the public pipeline without exposing or approximating any real patient record.

```bash
ataad-pop make-synthetic --output data/synthetic_ataad.csv --n 360
ataad-pop validate-data --data data/synthetic_ataad.csv
pytest
```

To run a fast end-to-end analysis:

```bash
ATAAD_POP_DATA=data/synthetic_ataad.csv \
python analysis/v2_jbhi_formal_validation.py \
  --mode quick --schemes loso \
  --feature-metadata config/feature_metadata.csv \
  --output-root .
```

Formal runs use 20 imputations and 2,000 bootstrap replicates and can require substantial CPU time.

## Controlled clinical-data run

Place an institutionally authorized workbook outside Git version control and pass it explicitly. The expected endpoint, environment, and metadata fields are documented in [docs/DATA_SCHEMA.md](docs/DATA_SCHEMA.md).

```bash
python analysis/v2_jbhi_formal_validation.py \
  --data /secure/path/TAAD.xlsx \
  --sheet main \
  --feature-metadata config/feature_metadata.csv \
  --output-root /secure/path/analysis-output \
  --mode formal --schemes both
```

Never open a pull request containing raw records, row identifiers, free text, dates of birth, medical-record numbers, or patient-level prediction files.

## Reproducibility

The confirmatory seed is `20260814`. The full run order, leakage-control contract, expected artifacts, and frozen interpretation boundaries are documented in [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md). Aggregate manuscript results are retained under `results/aggregated/`; patient-level predictions are intentionally excluded.

## Citation

If you use this software, cite the repository using [CITATION.cff](CITATION.cff). The associated manuscript citation will be added after publication.

## License and clinical disclaimer

The code is released under the [MIT License](LICENSE). It is research software supplied without warranty. It has not been validated for clinical deployment and must not delay or replace specialist assessment.
