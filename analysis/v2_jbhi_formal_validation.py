#!/usr/bin/env python3
"""Formal internal validation for the ATAAD POP-JBHI manuscript.

Primary endpoint: recorded Bentall operative decision.
Primary transportability test: leave-one-surgeon-out.
Secondary transportability test: rolling calendar year.

Every outer fold independently performs feature filtering, stochastic multiple
imputation, scaling, and model fitting. The source workbook and all previous
analysis outputs are read-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/ataad_jbhi_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import minimize
from sklearn.exceptions import ConvergenceWarning
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge, LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


REPOSITORY = Path(__file__).resolve().parents[1]
RAW_XLSX = Path(os.environ.get("ATAAD_POP_DATA", REPOSITORY / "data" / "TAAD_new1.xlsx"))
FEATURE_METADATA = Path(
    os.environ.get("ATAAD_POP_FEATURE_METADATA", REPOSITORY / "config" / "feature_metadata.csv")
)
PROJECT = Path(os.environ.get("ATAAD_POP_OUTPUT_ROOT", REPOSITORY))
SHEET_NAME = os.environ.get("ATAAD_POP_SHEET", "main")
SEED = int(os.environ.get("ATAAD_POP_SEED", "20260814"))
MISSINGNESS_THRESHOLD = 0.30
N_NEAREST_FEATURES = 40
FIXED_C = 0.01
TEST_YEARS = (2020, 2021, 2022, 2023, 2024)
HARD_BOUNDS = {"Age": (0.0, 120.0), "LVEF": (0.0, 100.0)}
KNOWN_REPLACEMENTS = {
    "CTD": {"1（类马方综合症）": 1},
    "First_tear_root": {"0（左锁骨下动脉开口）": 0},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("quick", "formal"), default="formal")
    parser.add_argument("--schemes", choices=("loso", "temporal", "both"), default="both")
    parser.add_argument("--data", type=Path, default=RAW_XLSX)
    parser.add_argument("--sheet", default=SHEET_NAME)
    parser.add_argument("--feature-metadata", type=Path, default=FEATURE_METADATA)
    parser.add_argument("--output-root", type=Path, default=PROJECT)
    return parser.parse_args()


def setup_dirs(mode: str) -> dict[str, Path]:
    base = PROJECT / "artifacts" / "formal_validation" / mode
    paths = {
        "base": base,
        "tables": base / "tables",
        "predictions": base / "predictions",
        "plots": base / "plots",
        "reports": base / "reports",
        "logs": base / "logs",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def setup_logging(log_path: Path) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()],
        force=True,
    )


def numeric(frame: pd.DataFrame, col: str) -> pd.Series:
    if col not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[col], errors="coerce")


def derive_bentall(frame: pd.DataFrame) -> pd.Series:
    mech = numeric(frame, "Bentall_mechanic_valve").fillna(0)
    bio = numeric(frame, "Bentall_bio_valve").fillna(0)
    return ((mech == 1) | (bio == 1)).astype(int)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*Conditional Formatting extension.*")
        if RAW_XLSX.suffix.lower() == ".csv":
            raw = pd.read_csv(RAW_XLSX)
        else:
            raw = pd.read_excel(RAW_XLSX, sheet_name=SHEET_NAME)
    raw.columns = raw.columns.astype(str).str.strip()
    raw.insert(0, "Row_ID", np.arange(1, len(raw) + 1, dtype=int))
    for col, mapping in KNOWN_REPLACEMENTS.items():
        if col in raw:
            raw[col] = raw[col].replace(mapping)
    raw["D1_Bentall"] = derive_bentall(raw)
    raw["Date_of_surgery"] = pd.to_datetime(raw["Date_of_surgery"], errors="coerce")
    raw["Year"] = raw["Date_of_surgery"].dt.year.astype("Int64")
    raw["Surgeon"] = numeric(raw, "Surgeon").astype("Int64")
    raw = raw.loc[numeric(raw, "Missingctimage").fillna(0).ne(1)].copy()
    raw = raw.loc[raw["Year"].notna() & raw["Surgeon"].notna()].reset_index(drop=True)

    metadata = pd.read_csv(FEATURE_METADATA)
    required = {"Variable", "Domain", "Missing_indicator_for_sensitivity"}
    missing_metadata = sorted(required.difference(metadata.columns))
    if missing_metadata:
        raise ValueError(f"Feature metadata missing columns: {missing_metadata}")
    candidate_meta = metadata[["Variable", "Domain"]].drop_duplicates().reset_index(drop=True)
    candidates = candidate_meta["Variable"].tolist()
    missing = [c for c in candidates if c not in raw.columns]
    if missing:
        raise ValueError(f"Candidate features absent from source workbook: {missing}")
    for col in candidates:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")

    observation_features = metadata.loc[
        metadata["Missing_indicator_for_sensitivity"].astype(str).str.lower().eq("true"),
        "Variable",
    ].tolist()
    return raw, candidate_meta, observation_features


def select_fold_features(
    train: pd.DataFrame,
    candidate_meta: pd.DataFrame,
    scheme: str,
    fold: int,
    environment: str,
) -> tuple[list[str], list[dict[str, object]]]:
    retained: list[str] = []
    rows: list[dict[str, object]] = []
    for item in candidate_meta.itertuples(index=False):
        series = pd.to_numeric(train[item.Variable], errors="coerce")
        n_observed = int(series.notna().sum())
        n_unique = int(series.dropna().nunique())
        missing_rate = float(series.isna().mean())
        keep = n_observed > 0 and n_unique > 1 and missing_rate <= MISSINGNESS_THRESHOLD
        if keep:
            retained.append(item.Variable)
        reason = ""
        if n_observed == 0:
            reason = "all missing in outer training"
        elif n_unique <= 1:
            reason = "constant in outer training"
        elif missing_rate > MISSINGNESS_THRESHOLD:
            reason = ">30% missing in outer training"
        rows.append({
            "Scheme": scheme,
            "Fold": fold,
            "Environment": environment,
            "Variable": item.Variable,
            "Domain": item.Domain,
            "Train_N": len(train),
            "Observed_N": n_observed,
            "Missing_rate": missing_rate,
            "Unique_observed": n_unique,
            "Retained": keep,
            "Drop_reason": reason,
        })
    return retained, rows


def infer_variable_types(
    train_x: pd.DataFrame,
) -> tuple[list[str], list[str], dict[str, np.ndarray]]:
    discrete: list[str] = []
    continuous: list[str] = []
    support: dict[str, np.ndarray] = {}
    for col in train_x.columns:
        values = pd.to_numeric(train_x[col], errors="coerce").dropna().to_numpy(dtype=float)
        unique = np.unique(values)
        is_binary = len(unique) <= 2 and set(unique.tolist()).issubset({0.0, 1.0})
        is_low_integer = 2 <= len(unique) <= 8 and np.all(np.isclose(unique, np.round(unique)))
        if is_binary or is_low_integer:
            discrete.append(col)
            support[col] = np.sort(unique)
        else:
            continuous.append(col)
    return discrete, continuous, support


def build_bounds(
    train_x: pd.DataFrame,
    discrete: Sequence[str],
    support: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    discrete_set = set(discrete)
    lower: list[float] = []
    upper: list[float] = []
    for col in train_x.columns:
        observed = pd.to_numeric(train_x[col], errors="coerce").dropna().astype(float)
        if col in discrete_set:
            lo = float(support[col].min())
            hi = float(support[col].max())
        else:
            lo = 0.0 if len(observed) and float(observed.min()) >= 0 else -np.inf
            hi = np.inf
            if col in HARD_BOUNDS:
                hard_lo, hard_hi = HARD_BOUNDS[col]
                lo = hard_lo if hard_lo is not None else lo
                hi = hard_hi if hard_hi is not None else hi
        lower.append(lo)
        upper.append(hi)
    return np.asarray(lower), np.asarray(upper)


def nearest_level(values: np.ndarray, levels: np.ndarray) -> np.ndarray:
    return levels[np.argmin(np.abs(values[:, None] - levels[None, :]), axis=1)]


def restore_observed_and_project(
    completed: np.ndarray,
    original: pd.DataFrame,
    discrete: Sequence[str],
    support: dict[str, np.ndarray],
) -> pd.DataFrame:
    out = pd.DataFrame(completed, columns=original.columns, index=original.index)
    discrete_set = set(discrete)
    for col in original.columns:
        source = pd.to_numeric(original[col], errors="coerce")
        observed = source.notna()
        out.loc[observed, col] = source.loc[observed]
        if col in discrete_set:
            missing = ~observed
            if missing.any():
                out.loc[missing, col] = nearest_level(
                    out.loc[missing, col].to_numpy(dtype=float), support[col]
                )
    return out.reset_index(drop=True)


def safe_metrics(y: Iterable[int], p: Iterable[float]) -> dict[str, float | int]:
    y_array = np.asarray(y, dtype=int)
    p_array = np.clip(np.asarray(p, dtype=float), 1e-8, 1 - 1e-8)
    two_classes = len(np.unique(y_array)) == 2
    return {
        "N": int(len(y_array)),
        "Events": int(y_array.sum()),
        "Prevalence": float(y_array.mean()),
        "AUROC": float(roc_auc_score(y_array, p_array)) if two_classes else np.nan,
        "AUPRC": float(average_precision_score(y_array, p_array)) if two_classes else np.nan,
        "Brier": float(brier_score_loss(y_array, p_array)),
        "LogLoss": float(log_loss(y_array, p_array, labels=[0, 1])),
    }


def calibration_intercept_slope(y: Iterable[int], p: Iterable[float]) -> tuple[float, float]:
    y_array = np.asarray(y, dtype=float)
    p_array = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    logit = np.log(p_array / (1 - p_array))

    def objective(params: np.ndarray) -> float:
        eta = params[0] + params[1] * logit
        probability = 1.0 / (1.0 + np.exp(-np.clip(eta, -35, 35)))
        return -float(np.sum(y_array * np.log(probability + 1e-12) + (1 - y_array) * np.log(1 - probability + 1e-12)))

    result = minimize(objective, x0=np.array([0.0, 1.0]), method="BFGS")
    if not result.success and not np.all(np.isfinite(result.x)):
        return np.nan, np.nan
    return float(result.x[0]), float(result.x[1])


def extended_metrics(y: Iterable[int], p: Iterable[float]) -> dict[str, float | int]:
    output = safe_metrics(y, p)
    intercept, slope = calibration_intercept_slope(y, p)
    output["Calibration_intercept"] = intercept
    output["Calibration_slope"] = slope
    return output


def make_outer_splits(raw: pd.DataFrame, schemes: str) -> list[dict[str, object]]:
    splits: list[dict[str, object]] = []
    fold = 0
    if schemes in ("loso", "both"):
        for surgeon in sorted(raw["Surgeon"].astype(int).unique()):
            fold += 1
            splits.append({
                "Scheme": "LEAVE_ONE_SURGEON_OUT",
                "Fold": fold,
                "Environment": f"Surgeon_{surgeon}",
                "Train_index": np.where(raw["Surgeon"].astype(int).to_numpy() != surgeon)[0],
                "Test_index": np.where(raw["Surgeon"].astype(int).to_numpy() == surgeon)[0],
            })
    if schemes in ("temporal", "both"):
        for temporal_fold, year in enumerate(TEST_YEARS, start=1):
            splits.append({
                "Scheme": "ROLLING_YEAR",
                "Fold": temporal_fold,
                "Environment": f"Year_{year}",
                "Train_index": np.where(raw["Year"].astype(int).to_numpy() < year)[0],
                "Test_index": np.where(raw["Year"].astype(int).to_numpy() == year)[0],
            })
    return splits


def fit_scaled_logistic(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(x_train)
    test_scaled = scaler.transform(x_test)
    model = LogisticRegression(
        C=FIXED_C,
        penalty="l2",
        solver="liblinear",
        max_iter=5000,
        random_state=SEED,
    )
    model.fit(train_scaled, y_train)
    return model.predict_proba(test_scaled)[:, 1], model.coef_.ravel()


def inner_oof_probabilities(x_train: np.ndarray, y_train: np.ndarray, seed: int) -> np.ndarray:
    """Generate training-only OOF probabilities for outer-fold recalibration."""
    probabilities = np.full(len(y_train), np.nan, dtype=float)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for inner_train, inner_validation in splitter.split(x_train, y_train):
        fold_probability, _ = fit_scaled_logistic(
            x_train[inner_train], y_train[inner_train], x_train[inner_validation]
        )
        probabilities[inner_validation] = fold_probability
    if np.isnan(probabilities).any():
        raise RuntimeError("Inner OOF calibration probabilities contain missing values")
    return probabilities


def apply_logistic_recalibration(p: np.ndarray, intercept: float, slope: float) -> np.ndarray:
    clipped = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    logit = np.log(clipped / (1 - clipped))
    eta = intercept + slope * logit
    return 1.0 / (1.0 + np.exp(-np.clip(eta, -35, 35)))


def run_outer_validation(
    raw: pd.DataFrame,
    candidate_meta: pd.DataFrame,
    observation_features: list[str],
    outer_splits: list[dict[str, object]],
    n_imputations: int,
    mice_rounds: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prediction_rows: list[dict[str, object]] = []
    mi_prediction_rows: list[dict[str, object]] = []
    fold_metric_rows: list[dict[str, object]] = []
    filter_rows: list[dict[str, object]] = []
    diagnostic_rows: list[dict[str, object]] = []
    coefficient_rows: list[dict[str, object]] = []

    for split_number, split in enumerate(outer_splits, start=1):
        scheme = str(split["Scheme"])
        fold = int(split["Fold"])
        environment = str(split["Environment"])
        train_idx = np.asarray(split["Train_index"], dtype=int)
        test_idx = np.asarray(split["Test_index"], dtype=int)
        train = raw.iloc[train_idx].copy().reset_index(drop=True)
        test = raw.iloc[test_idx].copy().reset_index(drop=True)
        y_train = train["D1_Bentall"].astype(int).to_numpy()
        y_test = test["D1_Bentall"].astype(int).to_numpy()
        if len(train) == 0 or len(test) == 0 or len(np.unique(y_train)) < 2:
            logging.warning("Skipping invalid outer split %s", environment)
            continue

        retained, fold_audit = select_fold_features(train, candidate_meta, scheme, fold, environment)
        filter_rows.extend(fold_audit)
        train_x = train[retained].apply(pd.to_numeric, errors="coerce")
        test_x = test[retained].apply(pd.to_numeric, errors="coerce")
        discrete, _, support = infer_variable_types(train_x)
        lower, upper = build_bounds(train_x, discrete, support)
        nearest = min(N_NEAREST_FEATURES, len(retained))

        usable_observation = [
            col for col in observation_features
            if train[col].isna().astype(int).nunique() > 1
        ]
        observation_train = train[usable_observation].isna().astype(float).to_numpy()
        observation_test = test[usable_observation].isna().astype(float).to_numpy()
        model_predictions = {
            "Phenotype_only": [],
            "Phenotype_plus_observation": [],
        }
        model_train_oof_predictions = {
            "Phenotype_only": [],
            "Phenotype_plus_observation": [],
        }

        logging.info(
            "Outer %d/%d | %s | train=%d test=%d events=%d/%d retained=%d obs_indicators=%d",
            split_number,
            len(outer_splits),
            environment,
            len(train),
            len(test),
            int(y_train.sum()),
            int(y_test.sum()),
            len(retained),
            len(usable_observation),
        )

        for imputation in range(1, n_imputations + 1):
            start = time.time()
            seed = SEED + split_number * 100_000 + imputation * 1009
            imputer = IterativeImputer(
                estimator=BayesianRidge(),
                sample_posterior=True,
                max_iter=mice_rounds,
                tol=1e-3,
                n_nearest_features=nearest,
                initial_strategy="median",
                imputation_order="ascending",
                skip_complete=False,
                min_value=lower,
                max_value=upper,
                random_state=seed,
            )
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                train_array = imputer.fit_transform(train_x)
                test_array = imputer.transform(test_x)
            train_completed = restore_observed_and_project(train_array, train_x, discrete, support)
            test_completed = restore_observed_and_project(test_array, test_x, discrete, support)

            variants = {
                "Phenotype_only": (
                    train_completed.to_numpy(dtype=float),
                    test_completed.to_numpy(dtype=float),
                    retained,
                ),
                "Phenotype_plus_observation": (
                    np.column_stack([train_completed.to_numpy(dtype=float), observation_train]),
                    np.column_stack([test_completed.to_numpy(dtype=float), observation_test]),
                    retained + [f"OBS__{col}" for col in usable_observation],
                ),
            }
            for model_name, (x_train_model, x_test_model, feature_names) in variants.items():
                probabilities, coefficients = fit_scaled_logistic(x_train_model, y_train, x_test_model)
                model_predictions[model_name].append(probabilities)
                model_train_oof_predictions[model_name].append(
                    inner_oof_probabilities(x_train_model, y_train, seed + 31)
                )
                for row_id, truth, probability in zip(test["Row_ID"], y_test, probabilities):
                    mi_prediction_rows.append({
                        "Scheme": scheme,
                        "Fold": fold,
                        "Environment": environment,
                        "Model": model_name,
                        "Imputation": imputation,
                        "Row_ID": int(row_id),
                        "Y": int(truth),
                        "P": float(probability),
                    })
                for feature, coefficient in zip(feature_names, coefficients):
                    coefficient_rows.append({
                        "Scheme": scheme,
                        "Fold": fold,
                        "Environment": environment,
                        "Model": model_name,
                        "Imputation": imputation,
                        "Feature": feature,
                        "Standardized_coefficient": float(coefficient),
                    })

            diagnostic_rows.append({
                "Scheme": scheme,
                "Fold": fold,
                "Environment": environment,
                "Imputation": imputation,
                "Seed": seed,
                "MICE_rounds_completed": int(imputer.n_iter_),
                "Retained_features": len(retained),
                "Train_residual_missing": int(train_completed.isna().sum().sum()),
                "Test_residual_missing": int(test_completed.isna().sum().sum()),
                "Runtime_seconds": time.time() - start,
            })
            logging.info("  MI %02d/%02d completed in %.1fs", imputation, n_imputations, time.time() - start)

        for model_name, probability_list in model_predictions.items():
            matrix = np.vstack(probability_list)
            pooled_probability_raw = matrix.mean(axis=0)
            pooled_train_oof = np.vstack(model_train_oof_predictions[model_name]).mean(axis=0)
            calibration_intercept, calibration_slope = calibration_intercept_slope(y_train, pooled_train_oof)
            if not np.isfinite(calibration_intercept) or not np.isfinite(calibration_slope) or calibration_slope <= 0:
                logging.warning("Invalid training-only calibration for %s/%s; using identity", environment, model_name)
                calibration_intercept, calibration_slope = 0.0, 1.0
            pooled_probability = apply_logistic_recalibration(
                pooled_probability_raw, calibration_intercept, calibration_slope
            )
            between_mi_sd = matrix.std(axis=0, ddof=1) if n_imputations > 1 else np.zeros(len(test))
            metrics = extended_metrics(y_test, pooled_probability)
            fold_metric_rows.append({
                "Scheme": scheme,
                "Fold": fold,
                "Environment": environment,
                "Model": model_name,
                "Train_N": len(train),
                "Retained_features": len(retained),
                "Observation_indicators": len(usable_observation),
                "Training_calibration_intercept": calibration_intercept,
                "Training_calibration_slope": calibration_slope,
                **metrics,
            })
            for local_row, probability, probability_raw, mi_sd in zip(
                test.itertuples(index=False), pooled_probability, pooled_probability_raw, between_mi_sd
            ):
                prediction_rows.append({
                    "Scheme": scheme,
                    "Fold": fold,
                    "Environment": environment,
                    "Model": model_name,
                    "Row_ID": int(local_row.Row_ID),
                    "Year": int(local_row.Year),
                    "Surgeon": int(local_row.Surgeon),
                    "Y": int(local_row.D1_Bentall),
                    "P": float(probability),
                    "P_raw_before_training_only_calibration": float(probability_raw),
                    "Between_MI_SD": float(mi_sd),
                    "Training_calibration_intercept": calibration_intercept,
                    "Training_calibration_slope": calibration_slope,
                })

    return (
        pd.DataFrame(prediction_rows),
        pd.DataFrame(mi_prediction_rows),
        pd.DataFrame(fold_metric_rows),
        pd.DataFrame(filter_rows),
        pd.DataFrame(diagnostic_rows),
        pd.DataFrame(coefficient_rows),
    )


def environment_stratified_bootstrap(
    group: pd.DataFrame,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    point = extended_metrics(group["Y"], group["P"])
    metric_names = ["AUROC", "AUPRC", "Brier", "LogLoss", "Calibration_intercept", "Calibration_slope"]
    samples = {name: [] for name in metric_names}
    environments = [d.index.to_numpy() for _, d in group.groupby("Environment")]
    for _ in range(n_bootstrap):
        sampled_indices = np.concatenate([
            rng.choice(indices, size=len(indices), replace=True) for indices in environments
        ])
        sampled = group.loc[sampled_indices]
        if sampled["Y"].nunique() < 2:
            continue
        result = extended_metrics(sampled["Y"], sampled["P"])
        for name in metric_names:
            if np.isfinite(result[name]):
                samples[name].append(float(result[name]))
    rows = []
    for name in metric_names:
        values = np.asarray(samples[name], dtype=float)
        rows.append({
            "Metric": name,
            "Estimate": float(point[name]),
            "CI_lower": float(np.percentile(values, 2.5)) if len(values) else np.nan,
            "CI_upper": float(np.percentile(values, 97.5)) if len(values) else np.nan,
            "Bootstrap_valid": int(len(values)),
        })
    return pd.DataFrame(rows)


def pooled_metrics_with_ci(
    predictions: pd.DataFrame,
    n_bootstrap: int,
) -> pd.DataFrame:
    all_rows = []
    for group_number, ((scheme, model), group) in enumerate(
        predictions.groupby(["Scheme", "Model"], sort=False), start=1
    ):
        interval = environment_stratified_bootstrap(group, n_bootstrap, SEED + group_number * 1000)
        interval.insert(0, "Model", model)
        interval.insert(0, "Scheme", scheme)
        interval.insert(2, "N", len(group))
        interval.insert(3, "Events", int(group["Y"].sum()))
        all_rows.append(interval)
    return pd.concat(all_rows, ignore_index=True)


def selective_performance(
    predictions: pd.DataFrame,
    random_repeats: int,
) -> pd.DataFrame:
    loso = predictions.loc[predictions["Scheme"].eq("LEAVE_ONE_SURGEON_OUT")].copy()
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(SEED + 777)
    for model_name, model_data in loso.groupby("Model"):
        model_data = model_data.copy()
        p = np.clip(model_data["P"].to_numpy(), 1e-8, 1 - 1e-8)
        model_data["Predictive_entropy"] = -(p * np.log(p) + (1 - p) * np.log(1 - p))
        for coverage in (1.0, 0.8, 0.6, 0.4):
            for rule in ("Predictive_entropy", "Between_MI_SD"):
                selected_parts = []
                selected_counts: dict[str, int] = {}
                for environment, environment_data in model_data.groupby("Environment"):
                    n_selected = max(1, int(math.floor(len(environment_data) * coverage)))
                    selected_counts[environment] = n_selected
                    selected_parts.append(environment_data.nsmallest(n_selected, rule))
                selected = pd.concat(selected_parts, ignore_index=True)
                metrics = extended_metrics(selected["Y"], selected["P"])
                rows.append({
                    "Model": model_name,
                    "Coverage": coverage,
                    "Selection_rule": rule,
                    **metrics,
                    "Random_AUROC_mean": np.nan,
                    "Random_AUROC_Q025": np.nan,
                    "Random_AUROC_Q975": np.nan,
                })

            random_aurocs = []
            for _ in range(random_repeats):
                selected_parts = []
                for environment, environment_data in model_data.groupby("Environment"):
                    n_selected = max(1, int(math.floor(len(environment_data) * coverage)))
                    choices = rng.choice(environment_data.index.to_numpy(), size=n_selected, replace=False)
                    selected_parts.append(environment_data.loc[choices])
                selected = pd.concat(selected_parts, ignore_index=True)
                if selected["Y"].nunique() == 2:
                    random_aurocs.append(roc_auc_score(selected["Y"], selected["P"]))
            rows.append({
                "Model": model_name,
                "Coverage": coverage,
                "Selection_rule": "Random_selection_reference",
                "N": int(sum(max(1, math.floor(len(d) * coverage)) for _, d in model_data.groupby("Environment"))),
                "Events": np.nan,
                "Prevalence": np.nan,
                "AUROC": np.nan,
                "AUPRC": np.nan,
                "Brier": np.nan,
                "LogLoss": np.nan,
                "Calibration_intercept": np.nan,
                "Calibration_slope": np.nan,
                "Random_AUROC_mean": float(np.mean(random_aurocs)),
                "Random_AUROC_Q025": float(np.percentile(random_aurocs, 2.5)),
                "Random_AUROC_Q975": float(np.percentile(random_aurocs, 97.5)),
            })
    return pd.DataFrame(rows)


def decision_curve(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    loso = predictions.loc[predictions["Scheme"].eq("LEAVE_ONE_SURGEON_OUT")]
    for model_name, group in loso.groupby("Model"):
        y = group["Y"].to_numpy(dtype=int)
        p = group["P"].to_numpy(dtype=float)
        n = len(y)
        prevalence = y.mean()
        for threshold in np.arange(0.05, 0.701, 0.025):
            positive = p >= threshold
            tp = int(np.sum(positive & (y == 1)))
            fp = int(np.sum(positive & (y == 0)))
            weight = threshold / (1 - threshold)
            rows.append({
                "Model": model_name,
                "Threshold": threshold,
                "Net_benefit_historical_decision": tp / n - fp / n * weight,
                "Treat_all": prevalence - (1 - prevalence) * weight,
                "Treat_none": 0.0,
                "Interpretation_boundary": "historical decision identification; not causal treatment benefit",
            })
    return pd.DataFrame(rows)


def calibration_bins(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (scheme, model), group in predictions.groupby(["Scheme", "Model"]):
        local = group.copy()
        local["Bin"] = pd.qcut(local["P"], q=10, labels=False, duplicates="drop")
        for bin_number, bin_data in local.groupby("Bin"):
            rows.append({
                "Scheme": scheme,
                "Model": model,
                "Bin": int(bin_number),
                "N": len(bin_data),
                "Predicted_mean": float(bin_data["P"].mean()),
                "Observed_rate": float(bin_data["Y"].mean()),
            })
    return pd.DataFrame(rows)


def coefficient_stability(coefficients: pd.DataFrame) -> pd.DataFrame:
    primary = coefficients.loc[
        coefficients["Scheme"].eq("LEAVE_ONE_SURGEON_OUT")
        & coefficients["Model"].eq("Phenotype_only")
    ].copy()
    rows = []
    total_fits = primary[["Fold", "Imputation"]].drop_duplicates().shape[0]
    for feature, group in primary.groupby("Feature"):
        values = group["Standardized_coefficient"].to_numpy(dtype=float)
        positive = float(np.mean(values > 0))
        negative = float(np.mean(values < 0))
        rows.append({
            "Feature": feature,
            "Fits_present": len(values),
            "Total_primary_fits": total_fits,
            "Presence_fraction": len(values) / total_fits if total_fits else np.nan,
            "Coefficient_median": float(np.median(values)),
            "Coefficient_Q25": float(np.percentile(values, 25)),
            "Coefficient_Q75": float(np.percentile(values, 75)),
            "Median_absolute_coefficient": float(np.median(np.abs(values))),
            "Sign_consistency": max(positive, negative),
        })
    return pd.DataFrame(rows).sort_values(
        ["Presence_fraction", "Median_absolute_coefficient"], ascending=[False, False]
    )


def make_plots(
    paths: dict[str, Path],
    predictions: pd.DataFrame,
    pooled_ci: pd.DataFrame,
    calibration: pd.DataFrame,
    selective: pd.DataFrame,
    dca: pd.DataFrame,
    stability: pd.DataFrame,
) -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    colors = {"Phenotype_only": "#2563EB", "Phenotype_plus_observation": "#0F766E"}

    auc = pooled_ci.loc[pooled_ci["Metric"].eq("AUROC")].copy()
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    for i, row in enumerate(auc.itertuples(index=False)):
        ax.errorbar(
            row.Estimate,
            i,
            xerr=[[row.Estimate - row.CI_lower], [row.CI_upper - row.Estimate]],
            fmt="o",
            color=colors.get(row.Model, "#374151"),
            capsize=4,
        )
    ax.set_yticks(range(len(auc)))
    ax.set_yticklabels([f"{r.Scheme}\n{r.Model}" for r in auc.itertuples(index=False)])
    ax.axvline(0.5, ls="--", color="#9CA3AF")
    ax.set_xlim(0.45, 1.0)
    ax.set_xlabel("AUROC with environment-stratified bootstrap 95% CI")
    ax.set_title("Formal 20-MI internal transportability validation")
    fig.tight_layout()
    fig.savefig(paths["plots"] / "formal_transportability_forest.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    loso_calibration = calibration.loc[calibration["Scheme"].eq("LEAVE_ONE_SURGEON_OUT")]
    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    for model_name, group in loso_calibration.groupby("Model"):
        ax.plot(group["Predicted_mean"], group["Observed_rate"], marker="o", label=model_name, color=colors[model_name])
    ax.plot([0, 1], [0, 1], ls="--", color="#6B7280", label="Ideal")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted probability", ylabel="Observed Bentall rate")
    ax.set_title("Cross-surgeon calibration")
    ax.legend()
    fig.tight_layout()
    fig.savefig(paths["plots"] / "formal_cross_surgeon_calibration.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    selected = selective.loc[selective["Selection_rule"].isin(["Predictive_entropy", "Between_MI_SD"])]
    for (model_name, rule), group in selected.groupby(["Model", "Selection_rule"]):
        ax.plot(group["Coverage"], group["AUROC"], marker="o", label=f"{model_name}: {rule}")
    ax.set(xlim=(0.35, 1.02), ylim=(0.5, 1.0), xlabel="Coverage retained within each surgeon", ylabel="Pooled AUROC")
    ax.set_title("MI-aware cross-surgeon selective prediction")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(paths["plots"] / "formal_selective_prediction.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    for model_name, group in dca.groupby("Model"):
        ax.plot(group["Threshold"], group["Net_benefit_historical_decision"], label=model_name, color=colors[model_name])
    reference = dca.loc[dca["Model"].eq(dca["Model"].iloc[0])]
    ax.plot(reference["Threshold"], reference["Treat_all"], ls="--", color="#6B7280", label="Classify all as Bentall")
    ax.axhline(0, color="#111827", lw=1, label="Classify none")
    ax.set(xlabel="Decision threshold", ylabel="Net benefit for historical decision identification")
    ax.set_title("Decision-concordance curve (not causal treatment benefit)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(paths["plots"] / "formal_decision_concordance_curve.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    top = stability.loc[stability["Presence_fraction"].ge(0.8)].head(20).sort_values("Coefficient_median")
    if len(top):
        fig, ax = plt.subplots(figsize=(8.0, 7.2))
        ax.errorbar(
            top["Coefficient_median"],
            np.arange(len(top)),
            xerr=[top["Coefficient_median"] - top["Coefficient_Q25"], top["Coefficient_Q75"] - top["Coefficient_median"]],
            fmt="o",
            color="#2563EB",
            capsize=3,
        )
        ax.axvline(0, color="#6B7280", ls="--")
        ax.set_yticks(np.arange(len(top)))
        ax.set_yticklabels(top["Feature"])
        ax.set_xlabel("Standardized coefficient median and IQR")
        ax.set_title("Most stable phenotype coefficients across surgeon folds and MI")
        fig.tight_layout()
        fig.savefig(paths["plots"] / "formal_coefficient_stability.png", dpi=240, bbox_inches="tight")
        plt.close(fig)


def write_report(
    paths: dict[str, Path],
    mode: str,
    raw: pd.DataFrame,
    pooled_ci: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    selective: pd.DataFrame,
    diagnostics: pd.DataFrame,
) -> None:
    def metric_row(scheme: str, model: str, metric: str) -> pd.Series:
        return pooled_ci.loc[
            pooled_ci["Scheme"].eq(scheme)
            & pooled_ci["Model"].eq(model)
            & pooled_ci["Metric"].eq(metric)
        ].iloc[0]

    lines = [
        "# ATAAD POP-JBHI 正式内部验证报告",
        "",
        f"- 运行模式：{mode}。",
        f"- 可用 CTA 病例：{len(raw):,}；Bentall 事件：{int(raw['D1_Bentall'].sum()):,}。",
        f"- 术者环境：{raw['Surgeon'].nunique()}；日期范围：{raw['Date_of_surgery'].min().date()}–{raw['Date_of_surgery'].max().date()}。",
        f"- 每个外层折插补次数范围：{int(diagnostics.groupby(['Scheme', 'Fold'])['Imputation'].nunique().min())}–{int(diagnostics.groupby(['Scheme', 'Fold'])['Imputation'].nunique().max())}。",
        "- 外层测试病例未参与特征筛选、插补器拟合、标准化或模型拟合。",
        "",
        "## Pooled transportability",
        "",
    ]
    for scheme in pooled_ci["Scheme"].drop_duplicates():
        lines.append(f"### {scheme}")
        lines.append("")
        for model in pooled_ci.loc[pooled_ci["Scheme"].eq(scheme), "Model"].drop_duplicates():
            auc = metric_row(scheme, model, "AUROC")
            auprc = metric_row(scheme, model, "AUPRC")
            brier = metric_row(scheme, model, "Brier")
            intercept = metric_row(scheme, model, "Calibration_intercept")
            slope = metric_row(scheme, model, "Calibration_slope")
            lines.append(
                f"- {model}: AUROC {auc.Estimate:.3f} (95% CI {auc.CI_lower:.3f}–{auc.CI_upper:.3f}); "
                f"AUPRC {auprc.Estimate:.3f}; Brier {brier.Estimate:.3f}; "
                f"calibration intercept {intercept.Estimate:.3f}, slope {slope.Estimate:.3f}."
            )
        lines.append("")

    loso_primary = fold_metrics.loc[
        fold_metrics["Scheme"].eq("LEAVE_ONE_SURGEON_OUT")
        & fold_metrics["Model"].eq("Phenotype_only")
    ]
    if len(loso_primary):
        lines.extend([
            "## 逐术者稳定性",
            "",
            f"- 可估计折：{loso_primary['AUROC'].notna().sum()}/{len(loso_primary)}。",
            f"- AUROC 中位数 {loso_primary['AUROC'].median():.3f}，范围 {loso_primary['AUROC'].min():.3f}–{loso_primary['AUROC'].max():.3f}。",
            "",
        ])

    selective_primary = selective.loc[
        selective["Model"].eq("Phenotype_only")
        & selective["Coverage"].eq(0.6)
        & selective["Selection_rule"].isin(["Predictive_entropy", "Between_MI_SD"])
    ]
    if len(selective_primary):
        lines.extend(["## 预设 60% coverage", ""])
        for row in selective_primary.itertuples(index=False):
            lines.append(
                f"- {row.Selection_rule}: N={row.N}, events={row.Events}, AUROC {row.AUROC:.3f}, "
                f"AUPRC {row.AUPRC:.3f}, Brier {row.Brier:.3f}."
            )
        lines.append("")

    lines.extend([
        "## 解释边界",
        "",
        "- 这些结果估计模型复现记录术式决策的能力，不估计 Bentall 相对于其他术式的因果治疗获益。",
        "- 当前 2016–2024 队列已参与 v1/v2 方向发现，因此属于严格内部验证，而非独立外部确认。",
        "- observation channel 的任何增益必须结合缺失流程压力测试解释，不能直接视作患者生物学信息。",
    ])
    (paths["reports"] / "JBHI_formal_internal_validation_report_zh.md").write_text("\n".join(lines), encoding="utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    global RAW_XLSX, FEATURE_METADATA, PROJECT, SHEET_NAME
    args = parse_args()
    RAW_XLSX = args.data.expanduser().resolve()
    FEATURE_METADATA = args.feature_metadata.expanduser().resolve()
    PROJECT = args.output_root.expanduser().resolve()
    SHEET_NAME = args.sheet
    paths = setup_dirs(args.mode)
    setup_logging(paths["logs"] / "formal_validation.log")
    n_imputations = 2 if args.mode == "quick" else 20
    mice_rounds = 5 if args.mode == "quick" else 20
    n_bootstrap = 200 if args.mode == "quick" else 2000
    random_repeats = 100 if args.mode == "quick" else 500

    raw, candidate_meta, observation_features = load_data()
    outer_splits = make_outer_splits(raw, args.schemes)
    logging.info(
        "Starting %s validation: N=%d, outer_splits=%d, MI=%d, rounds=%d",
        args.mode,
        len(raw),
        len(outer_splits),
        n_imputations,
        mice_rounds,
    )
    (
        predictions,
        mi_predictions,
        fold_metrics,
        feature_audit,
        diagnostics,
        coefficients,
    ) = run_outer_validation(
        raw,
        candidate_meta,
        observation_features,
        outer_splits,
        n_imputations,
        mice_rounds,
    )

    pooled_ci = pooled_metrics_with_ci(predictions, n_bootstrap)
    calibration = calibration_bins(predictions)
    stability = coefficient_stability(coefficients) if args.schemes in ("loso", "both") else pd.DataFrame()
    if args.schemes in ("loso", "both"):
        selective = selective_performance(predictions, random_repeats)
        dca = decision_curve(predictions)
    else:
        selective = pd.DataFrame()
        dca = pd.DataFrame()

    predictions.to_csv(paths["predictions"] / "pooled_outer_predictions.csv", index=False)
    mi_predictions.to_csv(paths["predictions"] / "per_imputation_outer_predictions.csv", index=False)
    fold_metrics.to_csv(paths["tables"] / "outer_fold_metrics.csv", index=False)
    feature_audit.to_csv(paths["tables"] / "outer_fold_feature_filtering.csv", index=False)
    diagnostics.to_csv(paths["tables"] / "outer_fold_mice_diagnostics.csv", index=False)
    coefficients.to_csv(paths["tables"] / "standardized_coefficients_all_fits.csv", index=False)
    pooled_ci.to_csv(paths["tables"] / "pooled_metrics_environment_bootstrap_ci.csv", index=False)
    calibration.to_csv(paths["tables"] / "calibration_bins.csv", index=False)
    if len(stability):
        stability.to_csv(paths["tables"] / "coefficient_stability.csv", index=False)
    if len(selective):
        selective.to_csv(paths["tables"] / "selective_performance.csv", index=False)
    if len(dca):
        dca.to_csv(paths["tables"] / "decision_concordance_curve.csv", index=False)
        make_plots(paths, predictions, pooled_ci, calibration, selective, dca, stability)

    write_report(paths, args.mode, raw, pooled_ci, fold_metrics, selective, diagnostics)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "schemes": args.schemes,
        "source_file": str(RAW_XLSX),
        "source_sha256": file_sha256(RAW_XLSX),
        "cohort_N": len(raw),
        "events": int(raw["D1_Bentall"].sum()),
        "surgeons": sorted(raw["Surgeon"].astype(int).unique().tolist()),
        "test_years": list(TEST_YEARS),
        "candidate_features": len(candidate_meta),
        "prespecified_observation_indicators": len(observation_features),
        "n_imputations": n_imputations,
        "mice_rounds": mice_rounds,
        "mice_estimator": "BayesianRidge with posterior sampling",
        "regularization_C": FIXED_C,
        "bootstrap_repetitions": n_bootstrap,
        "interpretation": "strict internal validation; not independent external confirmation",
    }
    (paths["base"] / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logging.info("Completed. Output directory: %s", paths["base"])


if __name__ == "__main__":
    main()
