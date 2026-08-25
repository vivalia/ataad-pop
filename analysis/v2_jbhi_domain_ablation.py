#!/usr/bin/env python3
"""Supportive clinical-domain ablation for the ATAAD POP/JBHI manuscript.

The analysis uses leave-one-surgeon-out validation. Every ablation refits
feature filtering, 20 stochastic MICE completions, scaling, L2 logistic
regression, and training-only probability recalibration inside each outer
training environment. The frozen full phenotype predictions are used as the
paired reference and are not refitted.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/ataad_jbhi_domain_ablation_matplotlib")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.exceptions import ConvergenceWarning
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

import v2_jbhi_formal_validation as core


REPOSITORY = Path(__file__).resolve().parents[1]
PROJECT = Path(os.environ.get("ATAAD_POP_OUTPUT_ROOT", REPOSITORY))
FORMAL = PROJECT / "artifacts" / "formal_validation" / "formal"
OUT = PROJECT / "artifacts" / "domain_ablation" / "formal"
TABLES = OUT / "tables"
PREDICTIONS = OUT / "predictions"
PLOTS = OUT / "plots"
REPORTS = OUT / "reports"
LOGS = OUT / "logs"
RAW_XLSX = Path(os.environ.get("ATAAD_POP_DATA", REPOSITORY / "data" / "TAAD_new1.xlsx"))

N_IMPUTATIONS = 20
MICE_ROUNDS = 20
N_BOOTSTRAP = 2000
MAX_WORKERS = 4
SEED = int(os.environ.get("ATAAD_POP_SEED", "20260814"))

DOMAIN_LABELS = {
    "Demographics_and_timing": "Demographics and timing",
    "Clinical_symptoms_and_status": "Symptoms and clinical status",
    "Comorbidities_and_history": "Comorbidities and history",
    "Laboratory_tests": "Laboratory tests",
    "Echocardiography_and_valve_status": "Echocardiography and valve status",
    "Preoperative_pathology_and_malperfusion": "Pathology and malperfusion",
    "Strict_preoperative_CTA": "CTA anatomy",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def variant_specs(all_domains: list[str]) -> list[dict[str, object]]:
    specs: list[dict[str, object]] = []
    for domain in all_domains:
        specs.append(
            {
                "Variant": f"Without__{domain}",
                "Family": "Leave_one_domain_out",
                "Display": f"Without {DOMAIN_LABELS[domain]}",
                "Domains": [item for item in all_domains if item != domain],
            }
        )
    imaging = {"Echocardiography_and_valve_status", "Strict_preoperative_CTA"}
    specs.append(
        {
            "Variant": "Without__all_preoperative_imaging",
            "Family": "Leave_group_out",
            "Display": "Without echocardiography and CTA",
            "Domains": [item for item in all_domains if item not in imaging],
        }
    )
    for domain in all_domains:
        specs.append(
            {
                "Variant": f"Only__{domain}",
                "Family": "Domain_only",
                "Display": f"{DOMAIN_LABELS[domain]} only",
                "Domains": [domain],
            }
        )
    anatomy_domains = [
        "Echocardiography_and_valve_status",
        "Preoperative_pathology_and_malperfusion",
        "Strict_preoperative_CTA",
    ]
    specs.append(
        {
            "Variant": "Only__anatomy_focused",
            "Family": "Domain_group_only",
            "Display": "Anatomy-focused domains only",
            "Domains": anatomy_domains,
        }
    )
    return specs


def run_variant(spec: dict[str, object]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw, candidate_meta, _ = core.load_data()
    candidate_meta = candidate_meta.loc[candidate_meta["Domain"].isin(spec["Domains"])].copy()
    outer_splits = core.make_outer_splits(raw, "loso")
    prediction_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    diagnostic_rows: list[dict[str, object]] = []

    for split_number, split in enumerate(outer_splits, start=1):
        started = time.time()
        train = raw.iloc[np.asarray(split["Train_index"], dtype=int)].copy().reset_index(drop=True)
        test = raw.iloc[np.asarray(split["Test_index"], dtype=int)].copy().reset_index(drop=True)
        y_train = train["D1_Bentall"].astype(int).to_numpy()
        y_test = test["D1_Bentall"].astype(int).to_numpy()
        retained, _ = core.select_fold_features(
            train,
            candidate_meta,
            str(split["Scheme"]),
            int(split["Fold"]),
            str(split["Environment"]),
        )
        if not retained:
            raise RuntimeError(f"{spec['Variant']} retained no features in {split['Environment']}")

        train_x = train[retained].apply(pd.to_numeric, errors="coerce")
        test_x = test[retained].apply(pd.to_numeric, errors="coerce")
        discrete, _, support = core.infer_variable_types(train_x)
        lower, upper = core.build_bounds(train_x, discrete, support)
        nearest = min(core.N_NEAREST_FEATURES, len(retained))
        test_probabilities: list[np.ndarray] = []
        train_oof_probabilities: list[np.ndarray] = []
        imputation_iterations: list[int] = []

        for imputation in range(1, N_IMPUTATIONS + 1):
            seed = SEED + split_number * 100_000 + imputation * 1009
            imputer = IterativeImputer(
                estimator=BayesianRidge(),
                sample_posterior=True,
                max_iter=MICE_ROUNDS,
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
            train_completed = core.restore_observed_and_project(
                train_array, train_x, discrete, support
            ).to_numpy(dtype=float)
            test_completed = core.restore_observed_and_project(
                test_array, test_x, discrete, support
            ).to_numpy(dtype=float)
            probability, _ = core.fit_scaled_logistic(train_completed, y_train, test_completed)
            test_probabilities.append(probability)
            train_oof_probabilities.append(
                core.inner_oof_probabilities(train_completed, y_train, seed + 31)
            )
            imputation_iterations.append(int(imputer.n_iter_))
            if imputation % 5 == 0:
                print(
                    f"[{spec['Display']}] {split['Environment']} "
                    f"MI {imputation:02d}/{N_IMPUTATIONS}",
                    flush=True,
                )

        matrix = np.vstack(test_probabilities)
        pooled_raw = matrix.mean(axis=0)
        pooled_train_oof = np.vstack(train_oof_probabilities).mean(axis=0)
        cal_intercept, cal_slope = core.calibration_intercept_slope(y_train, pooled_train_oof)
        if not np.isfinite(cal_intercept) or not np.isfinite(cal_slope) or cal_slope <= 0:
            cal_intercept, cal_slope = 0.0, 1.0
        pooled = core.apply_logistic_recalibration(pooled_raw, cal_intercept, cal_slope)
        mi_sd = matrix.std(axis=0, ddof=1)
        fold_metric = core.extended_metrics(y_test, pooled)

        fold_rows.append(
            {
                "Variant": spec["Variant"],
                "Family": spec["Family"],
                "Display": spec["Display"],
                "Environment": split["Environment"],
                "Fold": split["Fold"],
                "N": len(test),
                "Events": int(y_test.sum()),
                "Retained_features": len(retained),
                "Training_calibration_intercept": cal_intercept,
                "Training_calibration_slope": cal_slope,
                **fold_metric,
            }
        )
        for row, truth, probability, probability_raw, uncertainty in zip(
            test.itertuples(index=False), y_test, pooled, pooled_raw, mi_sd
        ):
            prediction_rows.append(
                {
                    "Variant": spec["Variant"],
                    "Family": spec["Family"],
                    "Display": spec["Display"],
                    "Environment": split["Environment"],
                    "Fold": split["Fold"],
                    "Row_ID": int(row.Row_ID),
                    "Y": int(truth),
                    "P": float(probability),
                    "P_raw_before_training_only_calibration": float(probability_raw),
                    "Between_MI_SD": float(uncertainty),
                }
            )
        diagnostic_rows.append(
            {
                "Variant": spec["Variant"],
                "Family": spec["Family"],
                "Display": spec["Display"],
                "Environment": split["Environment"],
                "Retained_features": len(retained),
                "MICE_iterations_median": float(np.median(imputation_iterations)),
                "Runtime_seconds": time.time() - started,
            }
        )
        print(
            f"[{spec['Display']}] {split['Environment']} complete; "
            f"features={len(retained)} AUROC={fold_metric['AUROC']:.3f}",
            flush=True,
        )

    return pd.DataFrame(prediction_rows), pd.DataFrame(fold_rows), pd.DataFrame(diagnostic_rows)


def compact_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(p, dtype=float), 1e-8, 1 - 1e-8)
    return {
        "AUROC": float(roc_auc_score(y, p)),
        "AUPRC": float(average_precision_score(y, p)),
        "Brier": float(brier_score_loss(y, p)),
        "LogLoss": float(log_loss(y, p, labels=[0, 1])),
    }


def environment_indices(frame: pd.DataFrame) -> list[np.ndarray]:
    return [group.index.to_numpy() for _, group in frame.groupby("Environment", sort=False)]


def bootstrap_metrics(frame: pd.DataFrame, seed: int) -> dict[str, tuple[float, float, int]]:
    rng = np.random.default_rng(seed)
    samples = {name: [] for name in compact_metrics(frame["Y"], frame["P"])}
    groups = environment_indices(frame)
    for _ in range(N_BOOTSTRAP):
        sampled = np.concatenate([rng.choice(index, size=len(index), replace=True) for index in groups])
        local = frame.loc[sampled]
        if local["Y"].nunique() < 2:
            continue
        values = compact_metrics(local["Y"].to_numpy(), local["P"].to_numpy())
        for name, value in values.items():
            samples[name].append(value)
    return {
        name: (
            float(np.percentile(values, 2.5)),
            float(np.percentile(values, 97.5)),
            len(values),
        )
        for name, values in samples.items()
    }


def pooled_summary(predictions: pd.DataFrame, folds: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for number, (variant, group) in enumerate(predictions.groupby("Variant", sort=False), start=1):
        point = core.extended_metrics(group["Y"], group["P"])
        interval = bootstrap_metrics(group.reset_index(drop=True), SEED + number * 1000)
        meta = group.iloc[0]
        feature_counts = folds.loc[folds["Variant"].eq(variant), "Retained_features"]
        row: dict[str, object] = {
            "Variant": variant,
            "Family": meta["Family"],
            "Display": meta["Display"],
            "N": len(group),
            "Events": int(group["Y"].sum()),
            "Retained_features_median": float(feature_counts.median()),
            "Retained_features_min": int(feature_counts.min()),
            "Retained_features_max": int(feature_counts.max()),
            **point,
        }
        for metric, (lower, upper, valid) in interval.items():
            row[f"{metric}_CI_lower"] = lower
            row[f"{metric}_CI_upper"] = upper
            row[f"{metric}_bootstrap_valid"] = valid
        rows.append(row)
    return pd.DataFrame(rows)


def paired_deltas(full: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    full = full[["Environment", "Row_ID", "Y", "P"]].rename(columns={"P": "P_full"})
    for number, (variant, group) in enumerate(predictions.groupby("Variant", sort=False), start=1):
        wide = group.merge(full, on=["Environment", "Row_ID", "Y"], validate="one_to_one")
        wide = wide.reset_index(drop=True)
        reference = compact_metrics(wide["Y"].to_numpy(), wide["P_full"].to_numpy())
        comparison = compact_metrics(wide["Y"].to_numpy(), wide["P"].to_numpy())
        samples = {name: [] for name in reference}
        rng = np.random.default_rng(SEED + 100_000 + number * 1000)
        groups = environment_indices(wide)
        for _ in range(N_BOOTSTRAP):
            sampled = np.concatenate([rng.choice(index, size=len(index), replace=True) for index in groups])
            local = wide.loc[sampled]
            if local["Y"].nunique() < 2:
                continue
            a = compact_metrics(local["Y"].to_numpy(), local["P_full"].to_numpy())
            b = compact_metrics(local["Y"].to_numpy(), local["P"].to_numpy())
            for name in samples:
                samples[name].append(b[name] - a[name])
        for metric in reference:
            values = np.asarray(samples[metric], dtype=float)
            rows.append(
                {
                    "Variant": variant,
                    "Family": group.iloc[0]["Family"],
                    "Display": group.iloc[0]["Display"],
                    "Metric": metric,
                    "Full_estimate": reference[metric],
                    "Ablated_estimate": comparison[metric],
                    "Paired_delta_ablation_minus_full": comparison[metric] - reference[metric],
                    "Delta_CI_lower": float(np.percentile(values, 2.5)),
                    "Delta_CI_upper": float(np.percentile(values, 97.5)),
                    "Bootstrap_valid": len(values),
                }
            )
    return pd.DataFrame(rows)


def make_figure(summary: pd.DataFrame, deltas: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid", context="paper")
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 6.2), gridspec_kw={"wspace": 0.48})

    left = deltas.loc[
        deltas["Metric"].eq("AUROC")
        & deltas["Family"].isin(["Leave_one_domain_out", "Leave_group_out"])
    ].copy()
    left = left.sort_values("Paired_delta_ablation_minus_full")
    y = np.arange(len(left))
    axes[0].errorbar(
        left["Paired_delta_ablation_minus_full"],
        y,
        xerr=np.vstack(
            [
                left["Paired_delta_ablation_minus_full"] - left["Delta_CI_lower"],
                left["Delta_CI_upper"] - left["Paired_delta_ablation_minus_full"],
            ]
        ),
        fmt="o",
        color="#B91C1C",
        ecolor="#FCA5A5",
        capsize=3,
    )
    axes[0].axvline(0, color="#475569", linestyle="--", linewidth=1)
    axes[0].set_yticks(y, left["Display"])
    axes[0].set_xlabel("Paired ΔAUROC versus full phenotype model")
    axes[0].set_title("A. Leave-domain-out dependence")

    right = summary.loc[
        summary["Family"].isin(["Domain_only", "Domain_group_only"])
    ].copy().sort_values("AUROC")
    y2 = np.arange(len(right))
    axes[1].errorbar(
        right["AUROC"],
        y2,
        xerr=np.vstack(
            [right["AUROC"] - right["AUROC_CI_lower"], right["AUROC_CI_upper"] - right["AUROC"]]
        ),
        fmt="o",
        color="#1D4ED8",
        ecolor="#93C5FD",
        capsize=3,
    )
    axes[1].axvline(0.5, color="#64748B", linestyle=":", linewidth=1)
    axes[1].axvline(0.8110214441443493, color="#059669", linestyle="--", linewidth=1)
    axes[1].set_yticks(y2, right["Display"])
    axes[1].set_xlim(0.45, 0.86)
    axes[1].set_xlabel("Leave-one-surgeon-out AUROC (95% CI)")
    axes[1].set_title("B. Information available within each domain")

    for ax in axes:
        ax.grid(axis="y", visible=False)
        sns.despine(ax=ax)
    fig.suptitle("Clinical-domain ablation of the Bentall phenotype model", y=1.01, fontsize=14)
    fig.tight_layout()
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(PLOTS / f"Figure6_domain_ablation.{suffix}", dpi=240, bbox_inches="tight")
    plt.close(fig)


def fmt(value: float) -> str:
    return "NA" if not np.isfinite(value) else f"{value:.3f}"


def write_report(summary: pd.DataFrame, deltas: pd.DataFrame) -> None:
    lodo = deltas.loc[
        deltas["Metric"].eq("AUROC")
        & deltas["Family"].isin(["Leave_one_domain_out", "Leave_group_out"])
    ].copy().sort_values("Paired_delta_ablation_minus_full")
    domain_only = summary.loc[
        summary["Family"].isin(["Domain_only", "Domain_group_only"])
    ].copy().sort_values("AUROC", ascending=False)
    lines = [
        "# JBHI 支持性临床域消融实验",
        "",
        "## 设计",
        "",
        "- 仅评估冻结的 Bentall 表型任务；使用留一术者外验证。",
        "- 每个消融变体均在每个外层训练折内重新进行特征过滤、20次随机MICE、标准化、L2逻辑回归和训练内重校准。",
        "- 主要消融为逐一移除七个临床信息域；另测试同时移除术前超声与CTA。",
        "- 单域模型用于回答每个信息域单独能提供多少跨术者信息，不作为替代主模型。",
        "- 与完整表型模型的差值按同一患者配对，并在每个术者环境内重采样2,000次。",
        "- 本分析在确认性主模型完成后开展，属于支持性、方案冻结后分析。",
        "",
        "## Leave-domain-out AUROC",
        "",
        "| 变体 | AUROC | 配对ΔAUROC vs完整模型 (95% CI) |",
        "|---|---:|---:|",
    ]
    for row in lodo.itertuples(index=False):
        lines.append(
            f"| {row.Display} | {fmt(row.Ablated_estimate)} | "
            f"{fmt(row.Paired_delta_ablation_minus_full)} "
            f"({fmt(row.Delta_CI_lower)} to {fmt(row.Delta_CI_upper)}) |"
        )
    lines += [
        "",
        "## Domain-only AUROC",
        "",
        "| 变体 | 保留特征中位数 | AUROC (95% CI) | AUPRC | Brier |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in domain_only.itertuples(index=False):
        lines.append(
            f"| {row.Display} | {row.Retained_features_median:.0f} | {fmt(row.AUROC)} "
            f"({fmt(row.AUROC_CI_lower)}–{fmt(row.AUROC_CI_upper)}) | "
            f"{fmt(row.AUPRC)} | {fmt(row.Brier)} |"
        )
    (REPORTS / "JBHI_domain_ablation_report_zh.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    for path in (OUT, TABLES, PREDICTIONS, PLOTS, REPORTS, LOGS):
        path.mkdir(parents=True, exist_ok=True)
    raw, candidate_meta, _ = core.load_data()
    domains = candidate_meta["Domain"].drop_duplicates().tolist()
    specs = variant_specs(domains)
    print(f"Starting {len(specs)} ablation variants with {MAX_WORKERS} workers", flush=True)

    prediction_parts: list[pd.DataFrame] = []
    fold_parts: list[pd.DataFrame] = []
    diagnostic_parts: list[pd.DataFrame] = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(run_variant, spec): spec for spec in specs}
        for future in as_completed(futures):
            spec = futures[future]
            predictions, folds, diagnostics = future.result()
            prediction_parts.append(predictions)
            fold_parts.append(folds)
            diagnostic_parts.append(diagnostics)
            print(f"VARIANT COMPLETE: {spec['Display']}", flush=True)

    predictions = pd.concat(prediction_parts, ignore_index=True)
    folds = pd.concat(fold_parts, ignore_index=True)
    diagnostics = pd.concat(diagnostic_parts, ignore_index=True)
    order = {spec["Variant"]: number for number, spec in enumerate(specs)}
    predictions["_order"] = predictions["Variant"].map(order)
    folds["_order"] = folds["Variant"].map(order)
    diagnostics["_order"] = diagnostics["Variant"].map(order)
    predictions = predictions.sort_values(["_order", "Fold", "Row_ID"]).drop(columns="_order")
    folds = folds.sort_values(["_order", "Fold"]).drop(columns="_order")
    diagnostics = diagnostics.sort_values(["_order", "Environment"]).drop(columns="_order")

    full = pd.read_csv(FORMAL / "predictions" / "pooled_outer_predictions.csv")
    full = full.loc[
        full["Scheme"].eq("LEAVE_ONE_SURGEON_OUT") & full["Model"].eq("Phenotype_only")
    ].copy()
    summary = pooled_summary(predictions, folds)
    deltas = paired_deltas(full, predictions)

    predictions.to_csv(PREDICTIONS / "domain_ablation_predictions.csv", index=False)
    folds.to_csv(TABLES / "domain_ablation_fold_metrics.csv", index=False)
    diagnostics.to_csv(TABLES / "domain_ablation_diagnostics.csv", index=False)
    summary.to_csv(TABLES / "domain_ablation_pooled_metrics.csv", index=False)
    deltas.to_csv(TABLES / "domain_ablation_paired_deltas_vs_full.csv", index=False)
    make_figure(summary, deltas)
    write_report(summary, deltas)

    manifest = {
        "created_utc": pd.Timestamp.utcnow().isoformat(),
        "analysis_status": "post-freeze supportive clinical-domain ablation",
        "source_file": str(RAW_XLSX),
        "source_sha256": sha256(RAW_XLSX),
        "cohort_N": len(raw),
        "events": int(raw["D1_Bentall"].sum()),
        "outer_validation": "leave-one-surgeon-out",
        "variants": specs,
        "n_imputations": N_IMPUTATIONS,
        "mice_rounds": MICE_ROUNDS,
        "regularization_C": core.FIXED_C,
        "bootstrap_repetitions": N_BOOTSTRAP,
        "workers": MAX_WORKERS,
        "interpretation": "supportive dependence and information-content analysis; not a new confirmatory endpoint",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Completed domain ablation: {OUT}", flush=True)


if __name__ == "__main__":
    main()
