# Reproducibility guide

## Frozen settings

- master random seed: `20260814`;
- primary endpoint: recorded Bentall decision;
- primary environment validation: leave-one-surgeon-out;
- secondary environment validation: rolling calendar year;
- feature missingness threshold: 30% in outer training data;
- stochastic imputations: 20 formal, 2 quick;
- MICE iterations: 20 formal, 5 quick;
- maximum neighboring features: 40;
- L2-logistic inverse regularization strength: `C=0.01`;
- environment-stratified bootstrap: 2,000 formal, 200 quick.

## Recommended run order

```bash
python analysis/v2_jbhi_formal_validation.py --mode formal --schemes both \
  --data /secure/path/TAAD.xlsx --feature-metadata config/feature_metadata.csv \
  --output-root /secure/path/analysis-output

ATAAD_POP_DATA=/secure/path/TAAD.xlsx \
ATAAD_POP_FEATURE_METADATA=config/feature_metadata.csv \
ATAAD_POP_OUTPUT_ROOT=/secure/path/analysis-output \
python analysis/v2_jbhi_missingness_stress.py

ATAAD_POP_DATA=/secure/path/TAAD.xlsx \
ATAAD_POP_OUTPUT_ROOT=/secure/path/analysis-output \
python analysis/v2_jbhi_paired_subgroup.py

ATAAD_POP_DATA=/secure/path/TAAD.xlsx \
ATAAD_POP_OUTPUT_ROOT=/secure/path/analysis-output \
python analysis/v2_jbhi_algorithm_benchmark.py --mode formal

ATAAD_POP_DATA=/secure/path/TAAD.xlsx \
ATAAD_POP_OUTPUT_ROOT=/secure/path/analysis-output \
python analysis/v2_jbhi_domain_ablation.py
```

Generated artifacts live under the selected output root and are ignored by Git. The public `results/aggregated/` directory contains only summary tables suitable for disclosure.

## Expected-data boundary

The original study used a corrected institutional workbook and controlled feature metadata. Exact reproduction of manuscript numbers requires authorized access to that version. The synthetic example tests code paths and schema contracts but is not designed to reproduce clinical performance.

## Analysis status

The main phenotype model and its environment-validation strategy are confirmatory within the existing single-center dataset. Algorithm benchmarking, subgroups, small-environment sensitivity, and clinical-domain ablation are post-freeze supportive analyses. Independent-center or untouched prospective temporal validation remains required before clinical use.

