#!/usr/bin/env python3
"""Supportive algorithm benchmark under the frozen LOSO/20-MI design.

The confirmatory model remains the prespecified L2 logistic regression. This
post-freeze benchmark tests whether its transportability finding depends on a
particular learner. Every learner receives the same fold-specific features and
the same independently fitted stochastic imputations.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/ataad_jbhi_matplotlib")

import lightgbm
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge, LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


REPOSITORY = Path(__file__).resolve().parents[1]
PROJECT = Path(os.environ.get("ATAAD_POP_OUTPUT_ROOT", REPOSITORY))
FORMAL = PROJECT / "artifacts" / "formal_validation" / "formal"
SCRIPT_DIR = PROJECT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import v2_jbhi_formal_validation as frozen  # noqa: E402


SEED = frozen.SEED
N_BOOTSTRAP = 2000
ALGORITHM_SPECS = {
    "L2_logistic_reproduction": "L2 logistic regression, C=0.01",
    "Elastic_net_logistic": "elastic-net logistic regression, C=0.01, l1_ratio=0.5",
    "Random_forest": "random forest, 150 trees, min_samples_leaf=10",
    "LightGBM": "LightGBM, 150 trees, learning_rate=0.025, num_leaves=15",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("quick", "formal"), default="formal")
    return parser.parse_args()


def setup(mode: str) -> dict[str, Path]:
    base = FORMAL / "algorithm_benchmark" / mode
    paths = {
        "base": base,
        "predictions": base / "predictions",
        "tables": base / "tables",
        "plots": base / "plots",
        "reports": base / "reports",
        "logs": base / "logs",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(paths["logs"] / "algorithm_benchmark.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )
    return paths


def make_estimator(name: str, seed: int, tree_estimators: int):
    if name == "L2_logistic_reproduction":
        return LogisticRegression(
            C=0.01,
            penalty="l2",
            solver="liblinear",
            max_iter=5000,
            random_state=SEED,
        )
    if name == "Elastic_net_logistic":
        return LogisticRegression(
            C=0.01,
            penalty="elasticnet",
            l1_ratio=0.5,
            solver="saga",
            max_iter=5000,
            tol=1e-4,
            random_state=seed,
        )
    if name == "Random_forest":
        return RandomForestClassifier(
            n_estimators=tree_estimators,
            min_samples_leaf=10,
            max_features="sqrt",
            bootstrap=True,
            random_state=seed,
            n_jobs=2,
        )
    if name == "LightGBM":
        return LGBMClassifier(
            objective="binary",
            n_estimators=tree_estimators,
            learning_rate=0.025,
            num_leaves=15,
            max_depth=5,
            min_child_samples=30,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.7,
            reg_alpha=1.0,
            reg_lambda=5.0,
            random_state=seed,
            n_jobs=2,
            verbosity=-1,
        )
    raise ValueError(name)


def uses_scaling(name: str) -> bool:
    return name in {"L2_logistic_reproduction", "Elastic_net_logistic"}


def fit_predict(
    name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    seed: int,
    tree_estimators: int,
) -> np.ndarray:
    if uses_scaling(name):
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
    estimator = make_estimator(name, seed, tree_estimators)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", message="X does not have valid feature names.*")
        estimator.fit(x_train, y_train)
        return estimator.predict_proba(x_test)[:, 1]


def inner_oof(
    name: str,
    x: np.ndarray,
    y: np.ndarray,
    seed: int,
    tree_estimators: int,
) -> np.ndarray:
    output = np.full(len(y), np.nan, dtype=float)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for inner_number, (train_idx, validation_idx) in enumerate(splitter.split(x, y), start=1):
        output[validation_idx] = fit_predict(
            name,
            x[train_idx],
            y[train_idx],
            x[validation_idx],
            seed + inner_number * 17,
            tree_estimators,
        )
    if np.isnan(output).any():
        raise RuntimeError(f"Missing inner OOF predictions for {name}")
    return output


def run_benchmark(
    raw: pd.DataFrame,
    candidate_meta: pd.DataFrame,
    splits: list[dict[str, object]],
    n_imputations: int,
    mice_rounds: int,
    tree_estimators: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prediction_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    diagnostic_rows: list[dict[str, object]] = []
    algorithms = list(ALGORITHM_SPECS)

    for split_number, split in enumerate(splits, start=1):
        fold_start = time.time()
        train = raw.iloc[np.asarray(split["Train_index"], dtype=int)].copy().reset_index(drop=True)
        test = raw.iloc[np.asarray(split["Test_index"], dtype=int)].copy().reset_index(drop=True)
        y_train = train["D1_Bentall"].astype(int).to_numpy()
        y_test = test["D1_Bentall"].astype(int).to_numpy()
        retained, _ = frozen.select_fold_features(
            train,
            candidate_meta,
            str(split["Scheme"]),
            int(split["Fold"]),
            str(split["Environment"]),
        )
        train_x = train[retained].apply(pd.to_numeric, errors="coerce")
        test_x = test[retained].apply(pd.to_numeric, errors="coerce")
        discrete, _, support = frozen.infer_variable_types(train_x)
        lower, upper = frozen.build_bounds(train_x, discrete, support)
        nearest = min(frozen.N_NEAREST_FEATURES, len(retained))
        outer_predictions = {name: [] for name in algorithms}
        train_oof_predictions = {name: [] for name in algorithms}

        logging.info(
            "Fold %d/%d %s | train=%d test=%d retained=%d",
            split_number,
            len(splits),
            split["Environment"],
            len(train),
            len(test),
            len(retained),
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
            completed_train = frozen.restore_observed_and_project(
                train_array, train_x, discrete, support
            ).to_numpy(dtype=float)
            completed_test = frozen.restore_observed_and_project(
                test_array, test_x, discrete, support
            ).to_numpy(dtype=float)

            for algorithm_number, name in enumerate(algorithms, start=1):
                model_seed = seed + algorithm_number * 101
                outer_predictions[name].append(
                    fit_predict(
                        name,
                        completed_train,
                        y_train,
                        completed_test,
                        model_seed,
                        tree_estimators,
                    )
                )
                train_oof_predictions[name].append(
                    inner_oof(
                        name,
                        completed_train,
                        y_train,
                        seed + 31 if name == "L2_logistic_reproduction" else model_seed + 31,
                        tree_estimators,
                    )
                )
            diagnostic_rows.append({
                "Fold": int(split["Fold"]),
                "Environment": str(split["Environment"]),
                "Imputation": imputation,
                "Seed": seed,
                "MICE_rounds_completed": int(imputer.n_iter_),
                "Retained_features": len(retained),
                "Runtime_seconds": time.time() - start,
            })
            logging.info(
                "  MI %02d/%02d completed in %.1fs",
                imputation,
                n_imputations,
                time.time() - start,
            )

        for name in algorithms:
            matrix = np.vstack(outer_predictions[name])
            raw_probability = matrix.mean(axis=0)
            pooled_train_oof = np.vstack(train_oof_predictions[name]).mean(axis=0)
            intercept, slope = frozen.calibration_intercept_slope(y_train, pooled_train_oof)
            if not np.isfinite(intercept) or not np.isfinite(slope) or slope <= 0:
                logging.warning("Invalid calibrator for %s/%s; using identity", split["Environment"], name)
                intercept, slope = 0.0, 1.0
            probability = frozen.apply_logistic_recalibration(raw_probability, intercept, slope)
            local_metrics = frozen.extended_metrics(y_test, probability)
            fold_rows.append({
                "Fold": int(split["Fold"]),
                "Environment": str(split["Environment"]),
                "Algorithm": name,
                "Train_N": len(train),
                "Test_N": len(test),
                "Test_events": int(y_test.sum()),
                "Retained_features": len(retained),
                "Training_calibration_intercept": intercept,
                "Training_calibration_slope": slope,
                **local_metrics,
            })
            between_mi_sd = matrix.std(axis=0, ddof=1) if n_imputations > 1 else np.zeros(len(test))
            for local_row, truth, p, p_raw, mi_sd in zip(
                test.itertuples(index=False),
                y_test,
                probability,
                raw_probability,
                between_mi_sd,
            ):
                prediction_rows.append({
                    "Fold": int(split["Fold"]),
                    "Environment": str(split["Environment"]),
                    "Algorithm": name,
                    "Row_ID": int(local_row.Row_ID),
                    "Y": int(truth),
                    "P": float(p),
                    "P_raw": float(p_raw),
                    "Between_MI_SD": float(mi_sd),
                    "Training_calibration_intercept": intercept,
                    "Training_calibration_slope": slope,
                })
        logging.info("Fold %s completed in %.1fs", split["Environment"], time.time() - fold_start)
    return pd.DataFrame(prediction_rows), pd.DataFrame(fold_rows), pd.DataFrame(diagnostic_rows)


def metric_values(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(p, dtype=float), 1e-8, 1 - 1e-8)
    return {
        "AUROC": float(roc_auc_score(y, p)),
        "AUPRC": float(average_precision_score(y, p)),
        "Brier": float(brier_score_loss(y, p)),
        "LogLoss": float(log_loss(y, p, labels=[0, 1])),
    }


def resample_by_environment(frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    sampled = []
    for _, group in frame.groupby("Environment", sort=False):
        index = group.index.to_numpy()
        sampled.append(rng.choice(index, len(index), replace=True))
    return frame.loc[np.concatenate(sampled)]


def summarize_predictions(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    algorithms = predictions["Algorithm"].drop_duplicates().tolist()
    point = {
        name: metric_values(group["Y"].to_numpy(), group["P"].to_numpy())
        for name, group in predictions.groupby("Algorithm", sort=False)
    }
    samples = {name: {metric: [] for metric in point[name]} for name in algorithms}
    wide = predictions.pivot(
        index=["Environment", "Row_ID", "Y"], columns="Algorithm", values="P"
    ).reset_index()
    delta_samples = {
        name: {metric: [] for metric in point[name]}
        for name in algorithms
        if name != "L2_logistic_reproduction"
    }
    rng = np.random.default_rng(SEED + 60000)
    for _ in range(N_BOOTSTRAP):
        sample = resample_by_environment(wide, rng)
        if sample["Y"].nunique() < 2:
            continue
        result = {
            name: metric_values(sample["Y"].to_numpy(), sample[name].to_numpy())
            for name in algorithms
        }
        for name in algorithms:
            for metric, value in result[name].items():
                samples[name][metric].append(value)
        reference = result["L2_logistic_reproduction"]
        for name in delta_samples:
            for metric in reference:
                delta_samples[name][metric].append(result[name][metric] - reference[metric])

    metric_rows: list[dict[str, object]] = []
    for name in algorithms:
        group = predictions.loc[predictions["Algorithm"].eq(name)]
        intercept, slope = frozen.calibration_intercept_slope(group["Y"], group["P"])
        for metric in point[name]:
            values = np.asarray(samples[name][metric], dtype=float)
            metric_rows.append({
                "Algorithm": name,
                "Metric": metric,
                "Estimate": point[name][metric],
                "CI_lower": np.percentile(values, 2.5),
                "CI_upper": np.percentile(values, 97.5),
                "Bootstrap_valid": len(values),
                "N": len(group),
                "Events": int(group["Y"].sum()),
                "Calibration_intercept": intercept,
                "Calibration_slope": slope,
            })

    delta_rows: list[dict[str, object]] = []
    for name, metric_data in delta_samples.items():
        for metric, values_list in metric_data.items():
            values = np.asarray(values_list, dtype=float)
            delta_rows.append({
                "Reference_algorithm": "L2_logistic_reproduction",
                "Comparison_algorithm": name,
                "Metric": metric,
                "Reference_estimate": point["L2_logistic_reproduction"][metric],
                "Comparison_estimate": point[name][metric],
                "Paired_delta_comparison_minus_reference": point[name][metric] - point["L2_logistic_reproduction"][metric],
                "Delta_CI_lower": np.percentile(values, 2.5),
                "Delta_CI_upper": np.percentile(values, 97.5),
                "Bootstrap_valid": len(values),
            })
    return pd.DataFrame(metric_rows), pd.DataFrame(delta_rows)


def verify_reproduction(predictions: pd.DataFrame) -> pd.DataFrame:
    formal = pd.read_csv(FORMAL / "predictions" / "pooled_outer_predictions.csv")
    formal = formal.loc[
        formal["Scheme"].eq("LEAVE_ONE_SURGEON_OUT")
        & formal["Model"].eq("Phenotype_only"),
        ["Row_ID", "P", "P_raw_before_training_only_calibration"],
    ].rename(columns={
        "P": "P_formal",
        "P_raw_before_training_only_calibration": "P_raw_formal",
    })
    reproduction = predictions.loc[
        predictions["Algorithm"].eq("L2_logistic_reproduction"),
        ["Row_ID", "P", "P_raw"],
    ].rename(columns={"P": "P_reproduction", "P_raw": "P_raw_reproduction"})
    merged = formal.merge(reproduction, on="Row_ID", validate="one_to_one")
    merged["Absolute_difference_calibrated"] = (
        merged["P_formal"] - merged["P_reproduction"]
    ).abs()
    merged["Absolute_difference_raw"] = (
        merged["P_raw_formal"] - merged["P_raw_reproduction"]
    ).abs()
    return merged


def make_plot(metrics_table: pd.DataFrame, paths: dict[str, Path]) -> None:
    display = metrics_table.loc[metrics_table["Metric"].eq("AUROC")].copy()
    display = display.sort_values("Estimate")
    labels = {
        "L2_logistic_reproduction": "L2 logistic (confirmatory)",
        "Elastic_net_logistic": "Elastic-net logistic",
        "Random_forest": "Random forest",
        "LightGBM": "LightGBM",
    }
    fig, ax = plt.subplots(figsize=(8.3, 4.7))
    y = np.arange(len(display))
    ax.errorbar(
        display["Estimate"],
        y,
        xerr=np.vstack([
            display["Estimate"] - display["CI_lower"],
            display["CI_upper"] - display["Estimate"],
        ]),
        fmt="o",
        color="#2563EB",
        ecolor="#93C5FD",
        capsize=3,
    )
    ax.set_yticks(y, [labels[name] for name in display["Algorithm"]])
    ax.axvline(0.5, color="#9CA3AF", linestyle="--", linewidth=1)
    ax.set_xlim(0.55, 0.88)
    ax.set_xlabel("Pooled leave-one-surgeon-out AUROC (95% CI)")
    ax.set_title("Supportive fixed-hyperparameter algorithm benchmark")
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(paths["plots"] / "formal_algorithm_benchmark.png", dpi=220)
    plt.close(fig)


def fmt(value: float) -> str:
    return "NA" if not np.isfinite(value) else f"{value:.3f}"


def write_report(
    metrics_table: pd.DataFrame,
    delta_table: pd.DataFrame,
    reproduction: pd.DataFrame,
    paths: dict[str, Path],
    mode: str,
    n_imputations: int,
) -> None:
    labels = {
        "L2_logistic_reproduction": "L2 logistic（确认性模型复现）",
        "Elastic_net_logistic": "Elastic-net logistic",
        "Random_forest": "Random forest",
        "LightGBM": "LightGBM",
    }
    lines = [
        "# JBHI 固定超参数算法支持性对照",
        "",
        "## 定位",
        "",
        "- L2 logistic 是冻结方案中的确认性主模型；其余算法为方案冻结后的支持性敏感性分析。",
        f"- 所有算法使用相同的留一术者外分割、相同的每折特征过滤、{n_imputations} 次独立 MICE 数据集及训练内 OOF 重校准。",
        "- 超参数固定，不根据外层测试结果挑选，因此本分析检验结论是否依赖特定学习器，而不是寻找最好分数。",
        "",
        "## 汇总结果",
        "",
        "| 算法 | AUROC (95% CI) | AUPRC | Brier | 校准截距 | 校准斜率 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for algorithm in ALGORITHM_SPECS:
        local = metrics_table.loc[metrics_table["Algorithm"].eq(algorithm)].set_index("Metric")
        auroc = local.loc["AUROC"]
        lines.append(
            f"| {labels[algorithm]} | {fmt(auroc.Estimate)} ({fmt(auroc.CI_lower)}–{fmt(auroc.CI_upper)}) | "
            f"{fmt(local.loc['AUPRC'].Estimate)} | {fmt(local.loc['Brier'].Estimate)} | "
            f"{fmt(auroc.Calibration_intercept)} | {fmt(auroc.Calibration_slope)} |"
        )
    lines += ["", "## 相对确认性 L2 logistic 的配对差异", ""]
    for row in delta_table.loc[delta_table["Metric"].eq("AUROC")].itertuples(index=False):
        lines.append(
            f"- {labels[row.Comparison_algorithm]}：AUROC Δ={fmt(row.Paired_delta_comparison_minus_reference)} "
            f"(95% CI {fmt(row.Delta_CI_lower)} 至 {fmt(row.Delta_CI_upper)})。"
        )
    max_cal = reproduction["Absolute_difference_calibrated"].max()
    max_raw = reproduction["Absolute_difference_raw"].max()
    if mode == "formal":
        sentinel_note = (
            "复现哨兵通过：正式20重插补复跑与主流程逐患者一致。"
            if max(max_cal, max_raw) <= 1e-12
            else "复现哨兵未通过：正式结果不得进入正文，需检查方案一致性。"
        )
    else:
        sentinel_note = "快速烟雾测试仅用2重插补，与正式20重插补预测不同属于预期现象。"
    lines += [
        "",
        "## 可复现性哨兵",
        "",
        f"- 独立复跑的 L2 logistic 与主流程逐患者预测最大绝对差：校准后 {max_cal:.3e}，校准前 {max_raw:.3e}。",
        f"- {sentinel_note}",
        "",
        "## 解释边界",
        "",
        "算法对照不能代替外部中心验证。若非线性模型没有稳定提高跨术者性能，论文将强调验证设计、捷径审计与不确定性治理，而不是声称新分类器本身具有创新性。",
    ]
    (paths["reports"] / "JBHI_algorithm_benchmark_report_zh.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    paths = setup(args.mode)
    if args.mode == "formal":
        # Match the frozen confirmatory pipeline exactly.  The reproduction
        # sentinel is only meaningful when both the number of imputations and
        # the maximum MICE iterations are identical to the primary analysis.
        n_imputations, mice_rounds, tree_estimators = 20, 20, 150
    else:
        n_imputations, mice_rounds, tree_estimators = 2, 5, 30
    raw, candidate_meta, _ = frozen.load_data()
    splits = [
        split for split in frozen.make_outer_splits(raw, "loso")
        if split["Scheme"] == "LEAVE_ONE_SURGEON_OUT"
    ]
    predictions, fold_metrics, diagnostics = run_benchmark(
        raw,
        candidate_meta,
        splits,
        n_imputations,
        mice_rounds,
        tree_estimators,
    )
    metrics_table, delta_table = summarize_predictions(predictions)
    reproduction = verify_reproduction(predictions)

    predictions.to_csv(paths["predictions"] / "algorithm_outer_predictions.csv", index=False)
    fold_metrics.to_csv(paths["tables"] / "algorithm_fold_metrics.csv", index=False)
    diagnostics.to_csv(paths["tables"] / "algorithm_mice_diagnostics.csv", index=False)
    metrics_table.to_csv(paths["tables"] / "algorithm_pooled_metrics.csv", index=False)
    delta_table.to_csv(paths["tables"] / "algorithm_paired_deltas_vs_l2.csv", index=False)
    reproduction.to_csv(paths["tables"] / "l2_reproduction_check.csv", index=False)
    make_plot(metrics_table, paths)
    write_report(metrics_table, delta_table, reproduction, paths, args.mode, n_imputations)

    manifest = {
        "mode": args.mode,
        "analysis_status": "supportive post-freeze fixed-hyperparameter benchmark",
        "n_imputations_per_outer_fold": n_imputations,
        "mice_rounds": mice_rounds,
        "outer_scheme": "leave-one-surgeon-out",
        "outer_folds": len(splits),
        "algorithms": ALGORITHM_SPECS,
        "n_environment_stratified_bootstrap": N_BOOTSTRAP,
        "seed": SEED,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "lightgbm": lightgbm.__version__,
        },
    }
    (paths["base"] / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Completed {args.mode} algorithm benchmark: {len(predictions)} prediction rows.")


if __name__ == "__main__":
    main()
