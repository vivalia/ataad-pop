# Synthetic example

Generate and validate a schema-compatible dataset:

```bash
ataad-pop make-synthetic --output data/synthetic_ataad.csv --n 360
ataad-pop validate-data --data data/synthetic_ataad.csv
```

Then run the quick leave-one-surgeon-out analysis shown in the main README. Synthetic results test execution only; they do not reproduce or validate any clinical claim.

