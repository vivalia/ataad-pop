# Reproduce the common-pipeline POP task audit (Figure 2)

The revised figure compares random-split and leave-one-surgeon-out results using
the same exploratory learner for all three tasks. Bentall LOSO AUROC is 0.793 in
this audit; 0.811 is the separate, frozen 20-imputation result and is not a point
in the revised figure. Use `task_triage_common_pipeline.csv` for Figure 2.
CABG is an indication-coded proxy, not adjudicated timing.

## Render the manuscript figure without patient data

```bash
python -m pip install -e .
python analysis/plot_common_pipeline.py \
  results/aggregated/primary/task_triage_common_pipeline.csv figure2.pdf
```

## Run the task-channel audit on fictional data

```bash
python examples/make_task_audit_demo.py /tmp/ataad-pop-demo/input.csv
ATAAD_POP_DATA=/tmp/ataad-pop-demo/input.csv \
ATAAD_POP_OUTPUT_ROOT=/tmp/ataad-pop-demo \
python analysis/v2_jbhi_task_channel_audit.py
```

The demo is generated from scratch and is not intended to reproduce any clinical
results. The same audit accepts an authorized local workbook or CSV with the
expected clinical fields, with free-text exceptions resolved during private data
curation before analysis. The historical three-task audit uses the previously
retained phenotype set; the separate formal Bentall pipeline screens candidates
inside each outer training fold. Exploratory penalty selection reuses transforms
fitted in the outer training set. Neither evaluation fits on outer test data.

Case-level predictions are not written by default. Enable them only for an
authorized private run by setting `ATAAD_POP_SAVE_CASE_PREDICTIONS=1`. Keep
clinical input, diagnostic plots, cohort and fold tables, and run outputs outside
Git; review disclosure risk before releasing any aggregates.
