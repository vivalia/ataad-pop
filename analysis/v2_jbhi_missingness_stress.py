#!/usr/bin/env python3
"""Evidence-derived missingness-shift stress tests for the JBHI primary model."""

from __future__ import annotations

import logging
import math
import os
import time
import warnings
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/ataad_jbhi_matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.exceptions import ConvergenceWarning
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge, LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

from v2_jbhi_formal_validation import (
    FIXED_C,
    N_NEAREST_FEATURES,
    SEED,
    apply_logistic_recalibration,
    build_bounds,
    calibration_intercept_slope,
    extended_metrics,
    infer_variable_types,
    inner_oof_probabilities,
    load_data,
    make_outer_splits,
    restore_observed_and_project,
    select_fold_features,
)


REPOSITORY = Path(__file__).resolve().parents[1]
PROJECT = Path(os.environ.get("ATAAD_POP_OUTPUT_ROOT", REPOSITORY))
FORMAL = PROJECT / "artifacts" / "formal_validation" / "formal"
TABLES = FORMAL / "tables"
PREDICTIONS = FORMAL / "predictions"
PLOTS = FORMAL / "plots"
REPORTS = FORMAL / "reports"
LOGS = FORMAL / "logs"
QUICK_MODE = os.environ.get("ATAAD_STRESS_QUICK", "0") == "1"
N_IMPUTATIONS = 2 if QUICK_MODE else 20
MICE_ROUNDS = 5 if QUICK_MODE else 20
BOOTSTRAP_REPETITIONS = 200 if QUICK_MODE else 2000


def mask_to_target_rates(
    frame: pd.DataFrame,
    target_rates: pd.Series,
    features: list[str],
    seed: int,
) -> tuple[pd.DataFrame, int]:
    out = frame.copy()
    rng = np.random.default_rng(seed)
    added = 0
    for col in features:
        target = float(np.clip(target_rates.get(col, 0.0), 0.0, 1.0))
        current_missing = out[col].isna()
        target_missing_n = int(math.ceil(target * len(out)))
        need = max(0, target_missing_n - int(current_missing.sum()))
        observed_index = out.index[~current_missing].to_numpy()
        if need > 0 and len(observed_index):
            chosen = rng.choice(observed_index, size=min(need, len(observed_index)), replace=False)
            out.loc[chosen, col] = np.nan
            added += len(chosen)
    return out, added


def make_scenarios(
    train_test: pd.DataFrame,
    observation_features: list[str],
    late_rates: pd.Series,
    worst_year_rates: pd.Series,
    echo_features: list[str],
    split_seed: int,
) -> tuple[dict[str, pd.DataFrame], dict[str, int]]:
    scenarios = {"Observed_workflow": train_test.copy()}
    counts = {"Observed_workflow": 0}
    late, added_late = mask_to_target_rates(
        train_test, late_rates, observation_features, split_seed + 101
    )
    scenarios["Emulate_2023_2024_missingness"] = late
    counts["Emulate_2023_2024_missingness"] = added_late
    worst, added_worst = mask_to_target_rates(
        train_test, worst_year_rates, observation_features, split_seed + 202
    )
    scenarios["Worst_observed_year_missingness"] = worst
    counts["Worst_observed_year_missingness"] = added_worst
    no_echo = train_test.copy()
    echo_existing = [col for col in echo_features if col in no_echo.columns]
    no_echo[echo_existing] = no_echo[echo_existing].astype(float)
    before = int(no_echo[echo_existing].isna().sum().sum())
    no_echo.loc[:, echo_existing] = np.nan
    after = int(no_echo[echo_existing].isna().sum().sum())
    scenarios["No_preoperative_echo_available"] = no_echo
    counts["No_preoperative_echo_available"] = after - before
    return scenarios, counts


def fit_model_and_predict_scenarios(
    train_completed: pd.DataFrame,
    y_train: np.ndarray,
    scenario_completed: dict[str, pd.DataFrame],
) -> dict[str, np.ndarray]:
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_completed.to_numpy(dtype=float))
    model = LogisticRegression(
        C=FIXED_C,
        penalty="l2",
        solver="liblinear",
        max_iter=5000,
        random_state=SEED,
    )
    model.fit(train_scaled, y_train)
    return {
        scenario: model.predict_proba(scaler.transform(completed.to_numpy(dtype=float)))[:, 1]
        for scenario, completed in scenario_completed.items()
    }


def run_stress_tests() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw, candidate_meta, observation_features = load_data()
    splits = make_outer_splits(raw, "loso")
    echo_features = candidate_meta.loc[
        candidate_meta["Domain"].eq("Echocardiography_and_valve_status"), "Variable"
    ].tolist()
    late_rates = raw.loc[raw["Year"].astype(int).ge(2023), observation_features].isna().mean()
    worst_year_rates = (
        raw.assign(_Year=raw["Year"].astype(int))
        .groupby("_Year")[observation_features]
        .apply(lambda d: d.isna().mean())
        .max(axis=0)
    )

    prediction_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    scenario_audit_rows: list[dict[str, object]] = []

    for split_number, split in enumerate(splits, start=1):
        fold = int(split["Fold"])
        environment = str(split["Environment"])
        train = raw.iloc[np.asarray(split["Train_index"], dtype=int)].copy().reset_index(drop=True)
        test = raw.iloc[np.asarray(split["Test_index"], dtype=int)].copy().reset_index(drop=True)
        y_train = train["D1_Bentall"].astype(int).to_numpy()
        y_test = test["D1_Bentall"].astype(int).to_numpy()
        retained, _ = select_fold_features(
            train, candidate_meta, "LEAVE_ONE_SURGEON_OUT", fold, environment
        )
        train_x = train[retained].apply(pd.to_numeric, errors="coerce")
        discrete, _, support = infer_variable_types(train_x)
        lower, upper = build_bounds(train_x, discrete, support)
        scenarios, added_counts = make_scenarios(
            test,
            observation_features,
            late_rates,
            worst_year_rates,
            echo_features,
            SEED + split_number * 1000,
        )
        for scenario, scenario_frame in scenarios.items():
            scenario_audit_rows.append({
                "Fold": fold,
                "Environment": environment,
                "Scenario": scenario,
                "Test_N": len(test),
                "Added_missing_cells": added_counts[scenario],
                "Total_missing_cells_in_retained_features": int(scenario_frame[retained].isna().sum().sum()),
            })

        scenario_probability_lists = {scenario: [] for scenario in scenarios}
        train_oof_lists: list[np.ndarray] = []
        logging.info(
            "Stress fold %d/%d %s | train=%d test=%d retained=%d",
            split_number,
            len(splits),
            environment,
            len(train),
            len(test),
            len(retained),
        )
        for imputation in range(1, N_IMPUTATIONS + 1):
            start = time.time()
            seed = SEED + split_number * 100_000 + imputation * 1009
            imputer = IterativeImputer(
                estimator=BayesianRidge(),
                sample_posterior=True,
                max_iter=MICE_ROUNDS,
                tol=1e-3,
                n_nearest_features=min(N_NEAREST_FEATURES, len(retained)),
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
                scenario_arrays = {
                    scenario: imputer.transform(frame[retained].apply(pd.to_numeric, errors="coerce"))
                    for scenario, frame in scenarios.items()
                }
            train_completed = restore_observed_and_project(
                train_array, train_x, discrete, support
            )
            scenario_completed = {
                scenario: restore_observed_and_project(
                    array,
                    scenarios[scenario][retained].apply(pd.to_numeric, errors="coerce"),
                    discrete,
                    support,
                )
                for scenario, array in scenario_arrays.items()
            }
            probabilities = fit_model_and_predict_scenarios(
                train_completed, y_train, scenario_completed
            )
            for scenario, p in probabilities.items():
                scenario_probability_lists[scenario].append(p)
            train_oof_lists.append(
                inner_oof_probabilities(
                    train_completed.to_numpy(dtype=float), y_train, seed + 31
                )
            )
            logging.info(
                "  %s MI %02d/%02d completed in %.1fs",
                environment,
                imputation,
                N_IMPUTATIONS,
                time.time() - start,
            )

        training_oof = np.vstack(train_oof_lists).mean(axis=0)
        calibration_intercept, calibration_slope = calibration_intercept_slope(
            y_train, training_oof
        )
        for scenario, probability_list in scenario_probability_lists.items():
            raw_probability = np.vstack(probability_list).mean(axis=0)
            probability = apply_logistic_recalibration(
                raw_probability, calibration_intercept, calibration_slope
            )
            metrics = extended_metrics(y_test, probability)
            fold_rows.append({
                "Fold": fold,
                "Environment": environment,
                "Scenario": scenario,
                **metrics,
            })
            for row_id, truth, p in zip(test["Row_ID"], y_test, probability):
                prediction_rows.append({
                    "Fold": fold,
                    "Environment": environment,
                    "Scenario": scenario,
                    "Row_ID": int(row_id),
                    "Y": int(truth),
                    "P": float(p),
                })
    return pd.DataFrame(prediction_rows), pd.DataFrame(fold_rows), pd.DataFrame(scenario_audit_rows)


def paired_environment_bootstrap(predictions: pd.DataFrame) -> pd.DataFrame:
    wide = predictions.pivot_table(
        index=["Environment", "Row_ID", "Y"], columns="Scenario", values="P"
    ).reset_index()
    scenarios = [col for col in predictions["Scenario"].unique() if col != "Observed_workflow"]
    environments = [d.index.to_numpy() for _, d in wide.groupby("Environment")]
    rng = np.random.default_rng(SEED + 919)
    rows = []
    baseline_metrics = extended_metrics(wide["Y"], wide["Observed_workflow"])
    rows.append({
        "Scenario": "Observed_workflow",
        **baseline_metrics,
        "Delta_AUROC_vs_observed": 0.0,
        "Delta_AUROC_CI_lower": 0.0,
        "Delta_AUROC_CI_upper": 0.0,
        "Delta_Brier_vs_observed": 0.0,
        "Delta_Brier_CI_lower": 0.0,
        "Delta_Brier_CI_upper": 0.0,
        "Mean_absolute_probability_shift": 0.0,
        "Probability_shift_gt_010": 0.0,
    })
    for scenario in scenarios:
        scenario_metrics = extended_metrics(wide["Y"], wide[scenario])
        delta_auc_samples = []
        delta_brier_samples = []
        for _ in range(BOOTSTRAP_REPETITIONS):
            sampled_idx = np.concatenate([
                rng.choice(indices, size=len(indices), replace=True) for indices in environments
            ])
            sample = wide.loc[sampled_idx]
            if sample["Y"].nunique() < 2:
                continue
            delta_auc_samples.append(
                roc_auc_score(sample["Y"], sample[scenario])
                - roc_auc_score(sample["Y"], sample["Observed_workflow"])
            )
            delta_brier_samples.append(
                brier_score_loss(sample["Y"], sample[scenario])
                - brier_score_loss(sample["Y"], sample["Observed_workflow"])
            )
        probability_shift = np.abs(wide[scenario] - wide["Observed_workflow"])
        rows.append({
            "Scenario": scenario,
            **scenario_metrics,
            "Delta_AUROC_vs_observed": scenario_metrics["AUROC"] - baseline_metrics["AUROC"],
            "Delta_AUROC_CI_lower": float(np.percentile(delta_auc_samples, 2.5)),
            "Delta_AUROC_CI_upper": float(np.percentile(delta_auc_samples, 97.5)),
            "Delta_Brier_vs_observed": scenario_metrics["Brier"] - baseline_metrics["Brier"],
            "Delta_Brier_CI_lower": float(np.percentile(delta_brier_samples, 2.5)),
            "Delta_Brier_CI_upper": float(np.percentile(delta_brier_samples, 97.5)),
            "Mean_absolute_probability_shift": float(probability_shift.mean()),
            "Probability_shift_gt_010": float(np.mean(probability_shift > 0.10)),
        })
    return pd.DataFrame(rows)


def verify_baseline(predictions: pd.DataFrame) -> float:
    formal = pd.read_csv(PREDICTIONS / "pooled_outer_predictions.csv")
    formal = formal.loc[
        formal["Scheme"].eq("LEAVE_ONE_SURGEON_OUT")
        & formal["Model"].eq("Phenotype_only"),
        ["Row_ID", "P"],
    ].rename(columns={"P": "P_formal"})
    baseline = predictions.loc[
        predictions["Scenario"].eq("Observed_workflow"), ["Row_ID", "P"]
    ].rename(columns={"P": "P_stress"})
    merged = formal.merge(baseline, on="Row_ID", validate="one_to_one")
    return float(np.max(np.abs(merged["P_formal"] - merged["P_stress"])))


def make_plot(summary: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plot_data = summary.loc[summary["Scenario"].ne("Observed_workflow")].copy()
    plot_data = plot_data.sort_values("Delta_AUROC_vs_observed")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    axes[0].errorbar(
        plot_data["Delta_AUROC_vs_observed"],
        np.arange(len(plot_data)),
        xerr=[
            plot_data["Delta_AUROC_vs_observed"] - plot_data["Delta_AUROC_CI_lower"],
            plot_data["Delta_AUROC_CI_upper"] - plot_data["Delta_AUROC_vs_observed"],
        ],
        fmt="o",
        color="#2563EB",
        capsize=4,
    )
    axes[0].axvline(0, color="#6B7280", ls="--")
    axes[0].set_yticks(np.arange(len(plot_data)))
    axes[0].set_yticklabels(plot_data["Scenario"])
    axes[0].set_xlabel("Change in cross-surgeon AUROC vs observed workflow")
    axes[0].set_title("Discrimination under observation-process shifts")
    sns.barplot(
        data=plot_data,
        x="Mean_absolute_probability_shift",
        y="Scenario",
        color="#0F766E",
        ax=axes[1],
    )
    axes[1].set_xlabel("Mean absolute probability shift")
    axes[1].set_ylabel("")
    axes[1].set_title("Individual prediction sensitivity")
    fig.tight_layout()
    fig.savefig(PLOTS / "formal_missingness_shift_stress.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def write_report(summary: pd.DataFrame, max_baseline_difference: float) -> None:
    lines = [
        "# JBHI 主模型缺失流程压力测试",
        "",
        "## 设计",
        "",
        "- 仅评估冻结的 Bentall phenotype-only 主模型。",
        f"- 每个 leave-one-surgeon-out 外层训练集重新执行 {N_IMPUTATIONS} 次 MICE 和训练内 OOF 校准。",
        "- 2023–2024 情景把各预设观察变量的测试缺失率提高到真实后期队列水平。",
        "- worst-year 情景把缺失率提高到该变量在任一真实年份出现过的最高水平。",
        "- no-echo 情景将术前超声域全部设为不可用。",
        f"- 与正式主分析基线预测最大绝对差：{max_baseline_difference:.3e}。",
        "",
        "## Pooled cross-surgeon results",
        "",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"- {row.Scenario}: AUROC {row.AUROC:.3f}, AUPRC {row.AUPRC:.3f}, Brier {row.Brier:.3f}; "
            f"ΔAUROC {row.Delta_AUROC_vs_observed:+.3f} "
            f"(95% CI {row.Delta_AUROC_CI_lower:+.3f}–{row.Delta_AUROC_CI_upper:+.3f}); "
            f"mean |ΔP| {row.Mean_absolute_probability_shift:.3f}, "
            f"|ΔP|>0.10 in {row.Probability_shift_gt_010:.1%}."
        )
    lines.extend([
        "",
        "## Interpretation boundary",
        "",
        "压力测试衡量对可观测信息变化的敏感性，不证明真实未来环境一定按该方式变化。测试只增加缺失，不合成不存在的生物学信息。",
    ])
    (REPORTS / "JBHI_missingness_shift_stress_report_zh.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(LOGS / "missingness_shift_stress.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )
    predictions, fold_metrics, scenario_audit = run_stress_tests()
    max_baseline_difference = verify_baseline(predictions)
    summary = paired_environment_bootstrap(predictions)
    predictions.to_csv(PREDICTIONS / "missingness_shift_stress_predictions.csv", index=False)
    fold_metrics.to_csv(TABLES / "missingness_shift_stress_fold_metrics.csv", index=False)
    scenario_audit.to_csv(TABLES / "missingness_shift_scenario_audit.csv", index=False)
    summary.to_csv(TABLES / "missingness_shift_stress_summary.csv", index=False)
    make_plot(summary)
    write_report(summary, max_baseline_difference)
    logging.info("Completed missingness stress tests; max baseline difference %.3e", max_baseline_difference)


if __name__ == "__main__":
    main()
