# Auditing Phenotype, Observation, and Practice Signals in Preoperative Operative-Decision Models for Acute Type A Aortic Dissection

ATAAD-POP audits which signals predict recorded surgical decisions and whether
that performance persists across surgeons and years. It does not recommend
treatment or estimate treatment benefit.

## Installation

Use Python 3.11 or newer. Dependencies and optional groups are defined in
[`pyproject.toml`](pyproject.toml).

```bash
git clone https://github.com/vivalia/ataad-pop.git
cd ataad-pop
python -m pip install -e .
```

## Reproduce the Figure 2 audit

Run all three tasks on fully fictional data:

```bash
demo=$(mktemp -d)
python examples/make_task_audit_demo.py "$demo/input.csv"
ATAAD_POP_DATA="$demo/input.csv" ATAAD_POP_OUTPUT_ROOT="$demo" \
  python analysis/v2_jbhi_task_channel_audit.py
```

Render the manuscript figure from public aggregate results:

```bash
python analysis/plot_common_pipeline.py \
  results/aggregated/primary/task_triage_common_pipeline.csv figure2.pdf
```

See the [Figure 2 guide](docs/FIGURE2_TASK_AUDIT.md) for analysis details and the
[reproducibility guide](docs/REPRODUCIBILITY.md) for the separate frozen Bentall
workflow and supporting analyses.

## Data access

See the [input schema](docs/DATA_SCHEMA.md) and [data access policy](docs/DATA_ACCESS.md).
Patient records and case-level outputs are not public. Keep authorized clinical
inputs and outputs outside this repository. The synthetic demo does not reproduce
clinical estimates.

## Citation and license

Cite the software using [`CITATION.cff`](CITATION.cff). Code is released under the
[MIT License](LICENSE).
