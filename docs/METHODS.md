# POP methods overview

## Signal provenance

For patient `i`, available information is separated conceptually into phenotype, observation, and practice channels:

`x_i = (x_i^P, x_i^O, x_i^R)`.

- **Phenotype:** strictly preoperative patient measurements and anatomy.
- **Observation:** indicators of which tests or values are missing.
- **Practice:** surgeon and calendar context.

The confirmatory model uses phenotype only. Observation indicators are a sensitivity and drift-audit channel. Practice variables define environments and shortcut audits; they are not inputs to the proposed patient score.

## Leakage-control contract

For every outer held-out environment:

1. feature missingness and variance filtering are learned from outer-training data;
2. each stochastic imputer is fitted on outer-training data;
3. held-out data are transformed without refitting;
4. observed values are restored and discrete values projected to training support;
5. scaling and L2-logistic fitting use outer-training data only;
6. calibration is learned from inner out-of-fold training predictions;
7. held-out labels are used only for final evaluation.

## Validation and inference

The main scheme is leave-one-surgeon-out. Rolling-year validation trains on earlier years and tests later years. Performance includes AUROC, AUPRC, Brier score, log loss, calibration intercept, and calibration slope. Confidence intervals and model contrasts use environment-stratified paired bootstrap resampling.

## Missing-data uncertainty

Each outer fold uses multiple stochastic completed datasets. Predictions are pooled at patient level; their between-imputation standard deviation is retained as a missing-data sensitivity measure. Coverage analyses select low-uncertainty cases within each environment and compare them with random retention at identical coverage. Deferred cases return to ordinary expert review.

## Interpretation boundary

The estimand is concordance with a recorded historical operation. Threshold analyses are descriptive historical-decision concordance analyses, not causal decision-curve evidence for treatment benefit.

