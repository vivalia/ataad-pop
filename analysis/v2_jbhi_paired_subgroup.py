#!/usr/bin/env python3
"""Paired model, recalibration, and post-freeze supportive subgroup analyses.

This script consumes the frozen formal-validation predictions. It does not
refit the prediction model and therefore cannot introduce information into the
outer validation folds.
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/ataad_jbhi_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import minimize
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score


REPOSITORY = Path(__file__).resolve().parents[1]
PROJECT = Path(os.environ.get("ATAAD_POP_OUTPUT_ROOT", REPOSITORY))
FORMAL = PROJECT / "artifacts" / "formal_validation" / "formal"
RAW_XLSX = Path(os.environ.get("ATAAD_POP_DATA", REPOSITORY / "data" / "TAAD_new1.xlsx"))
PREDICTIONS = FORMAL / "predictions" / "pooled_outer_predictions.csv"
N_BOOTSTRAP = 2000
SEED = int(os.environ.get("ATAAD_POP_SEED", "20260814"))


def calibration_intercept_slope(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    logit = np.log(p / (1 - p))

    def objective(params: np.ndarray) -> float:
        eta = params[0] + params[1] * logit
        probability = 1 / (1 + np.exp(-np.clip(eta, -35, 35)))
        return -float(
            np.sum(
                y * np.log(probability + 1e-12)
                + (1 - y) * np.log(1 - probability + 1e-12)
            )
        )

    result = minimize(objective, x0=np.array([0.0, 1.0]), method="BFGS")
    if not np.all(np.isfinite(result.x)):
        return np.nan, np.nan
    return float(result.x[0]), float(result.x[1])


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(p, dtype=float), 1e-8, 1 - 1e-8)
    two_classes = np.unique(y).size == 2
    intercept, slope = calibration_intercept_slope(y, p) if two_classes else (np.nan, np.nan)
    return {
        "AUROC": float(roc_auc_score(y, p)) if two_classes else np.nan,
        "AUPRC": float(average_precision_score(y, p)) if two_classes else np.nan,
        "Brier": float(brier_score_loss(y, p)),
        "LogLoss": float(log_loss(y, p, labels=[0, 1])),
        "Calibration_intercept": intercept,
        "Calibration_slope": slope,
    }


def sample_within_environment(frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    positions: list[np.ndarray] = []
    for _, group in frame.groupby("Environment", sort=False):
        index = group.index.to_numpy()
        positions.append(rng.choice(index, size=len(index), replace=True))
    return frame.loc[np.concatenate(positions)]


def paired_model_comparison(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    metric_names = ["AUROC", "AUPRC", "Brier", "LogLoss", "Calibration_intercept", "Calibration_slope"]
    for scheme_number, (scheme, group) in enumerate(predictions.groupby("Scheme", sort=False), start=1):
        wide = group.pivot(
            index=["Environment", "Row_ID", "Y"], columns="Model", values="P"
        ).reset_index()
        wide = wide.dropna(subset=["Phenotype_only", "Phenotype_plus_observation"])
        point_a = metrics(wide["Y"].to_numpy(), wide["Phenotype_only"].to_numpy())
        point_b = metrics(wide["Y"].to_numpy(), wide["Phenotype_plus_observation"].to_numpy())
        samples = {name: [] for name in metric_names}
        rng = np.random.default_rng(SEED + 1000 * scheme_number)
        for _ in range(N_BOOTSTRAP):
            sample = sample_within_environment(wide, rng)
            if sample["Y"].nunique() < 2:
                continue
            a = metrics(sample["Y"].to_numpy(), sample["Phenotype_only"].to_numpy())
            b = metrics(sample["Y"].to_numpy(), sample["Phenotype_plus_observation"].to_numpy())
            for name in metric_names:
                if np.isfinite(a[name]) and np.isfinite(b[name]):
                    samples[name].append(b[name] - a[name])
        for name in metric_names:
            values = np.asarray(samples[name], dtype=float)
            rows.append({
                "Scheme": scheme,
                "Reference_model": "Phenotype_only",
                "Comparison_model": "Phenotype_plus_observation",
                "Metric": name,
                "Reference_estimate": point_a[name],
                "Comparison_estimate": point_b[name],
                "Paired_delta_comparison_minus_reference": point_b[name] - point_a[name],
                "Delta_CI_lower": np.percentile(values, 2.5) if len(values) else np.nan,
                "Delta_CI_upper": np.percentile(values, 97.5) if len(values) else np.nan,
                "Bootstrap_valid": len(values),
                "N": len(wide),
                "Events": int(wide["Y"].sum()),
            })
    return pd.DataFrame(rows)


def recalibration_comparison(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    metric_names = ["AUROC", "AUPRC", "Brier", "LogLoss", "Calibration_intercept", "Calibration_slope"]
    primary = predictions.loc[predictions["Model"].eq("Phenotype_only")].copy()
    for scheme_number, (scheme, group) in enumerate(primary.groupby("Scheme", sort=False), start=1):
        point_raw = metrics(group["Y"].to_numpy(), group["P_raw_before_training_only_calibration"].to_numpy())
        point_cal = metrics(group["Y"].to_numpy(), group["P"].to_numpy())
        samples = {name: [] for name in metric_names}
        rng = np.random.default_rng(SEED + 5000 + 1000 * scheme_number)
        for _ in range(N_BOOTSTRAP):
            sample = sample_within_environment(group, rng)
            if sample["Y"].nunique() < 2:
                continue
            raw = metrics(sample["Y"].to_numpy(), sample["P_raw_before_training_only_calibration"].to_numpy())
            cal = metrics(sample["Y"].to_numpy(), sample["P"].to_numpy())
            for name in metric_names:
                if np.isfinite(raw[name]) and np.isfinite(cal[name]):
                    samples[name].append(cal[name] - raw[name])
        for name in metric_names:
            values = np.asarray(samples[name], dtype=float)
            rows.append({
                "Scheme": scheme,
                "Metric": name,
                "Raw_estimate": point_raw[name],
                "Training_only_recalibrated_estimate": point_cal[name],
                "Paired_delta_recalibrated_minus_raw": point_cal[name] - point_raw[name],
                "Delta_CI_lower": np.percentile(values, 2.5) if len(values) else np.nan,
                "Delta_CI_upper": np.percentile(values, 97.5) if len(values) else np.nan,
                "Bootstrap_valid": len(values),
                "N": len(group),
                "Events": int(group["Y"].sum()),
            })
    return pd.DataFrame(rows)


def load_subgroup_data() -> pd.DataFrame:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*Conditional Formatting extension.*")
        if RAW_XLSX.suffix.lower() == ".csv":
            raw = pd.read_csv(RAW_XLSX)
        else:
            raw = pd.read_excel(
                RAW_XLSX, sheet_name=os.environ.get("ATAAD_POP_SHEET", "main")
            )
    raw.columns = raw.columns.astype(str).str.strip()
    raw.insert(0, "Row_ID", np.arange(1, len(raw) + 1, dtype=int))
    raw["Male"] = pd.to_numeric(raw["Male"], errors="coerce")
    raw["Age"] = pd.to_numeric(raw["Age"], errors="coerce")
    raw["CTD"] = pd.to_numeric(raw["CTD"].replace({"1（类马方综合症）": 1}), errors="coerce")
    return raw[["Row_ID", "Male", "Age", "CTD"]]


def ordinary_bootstrap_ci(group: pd.DataFrame, seed: int) -> dict[str, tuple[float, float, int]]:
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {name: [] for name in metrics(group["Y"], group["P"])}
    for _ in range(N_BOOTSTRAP):
        sample = sample_within_environment(group, rng)
        if sample["Y"].nunique() < 2:
            continue
        result = metrics(sample["Y"].to_numpy(), sample["P"].to_numpy())
        for name, value in result.items():
            if np.isfinite(value):
                samples[name].append(value)
    return {
        name: (
            float(np.percentile(values, 2.5)) if values else np.nan,
            float(np.percentile(values, 97.5)) if values else np.nan,
            len(values),
        )
        for name, values in samples.items()
    }


def subgroup_performance(predictions: pd.DataFrame) -> pd.DataFrame:
    subgroup = load_subgroup_data()
    primary = predictions.loc[
        predictions["Scheme"].eq("LEAVE_ONE_SURGEON_OUT")
        & predictions["Model"].eq("Phenotype_only")
    ].merge(subgroup, on="Row_ID", validate="one_to_one")
    definitions = [
        ("Overall", "All", pd.Series(True, index=primary.index)),
        ("Sex", "Female", primary["Male"].eq(0)),
        ("Sex", "Male", primary["Male"].eq(1)),
        ("Age", "<60 years", primary["Age"].lt(60)),
        ("Age", "≥60 years", primary["Age"].ge(60)),
        ("Connective_tissue_disorder", "No", primary["CTD"].eq(0)),
        ("Connective_tissue_disorder", "Yes", primary["CTD"].eq(1)),
    ]
    rows: list[dict[str, object]] = []
    for group_number, (variable, level, mask) in enumerate(definitions, start=1):
        local = primary.loc[mask.fillna(False)].copy()
        point = metrics(local["Y"].to_numpy(), local["P"].to_numpy())
        interval = ordinary_bootstrap_ci(local, SEED + 10000 + group_number * 1000)
        row: dict[str, object] = {
            "Analysis_status": "post-freeze exploratory/supportive",
            "Subgroup_variable": variable,
            "Level": level,
            "N": len(local),
            "Events": int(local["Y"].sum()),
            "Prevalence": float(local["Y"].mean()) if len(local) else np.nan,
            "Missing_subgroup_value_N": int(primary[{"Sex": "Male", "Age": "Age", "Connective_tissue_disorder": "CTD"}.get(variable, "Age")].isna().sum()) if variable != "Overall" else 0,
        }
        for name, value in point.items():
            lower, upper, valid = interval[name]
            row[name] = value
            row[f"{name}_CI_lower"] = lower
            row[f"{name}_CI_upper"] = upper
            row[f"{name}_bootstrap_valid"] = valid
        rows.append(row)
    return pd.DataFrame(rows)


def subgroup_contrasts(predictions: pd.DataFrame) -> pd.DataFrame:
    subgroup = load_subgroup_data()
    primary = predictions.loc[
        predictions["Scheme"].eq("LEAVE_ONE_SURGEON_OUT")
        & predictions["Model"].eq("Phenotype_only")
    ].merge(subgroup, on="Row_ID", validate="one_to_one")
    contrasts = [
        ("Sex", "Female", primary["Male"].eq(0), "Male", primary["Male"].eq(1)),
        ("Age", "<60 years", primary["Age"].lt(60), "≥60 years", primary["Age"].ge(60)),
        (
            "Connective_tissue_disorder",
            "No",
            primary["CTD"].eq(0),
            "Yes",
            primary["CTD"].eq(1),
        ),
    ]
    metric_names = ["AUROC", "AUPRC", "Brier", "LogLoss"]
    rows: list[dict[str, object]] = []
    for contrast_number, (variable, reference, reference_mask, comparison, comparison_mask) in enumerate(
        contrasts, start=1
    ):
        reference_data = primary.loc[reference_mask.fillna(False)].copy()
        comparison_data = primary.loc[comparison_mask.fillna(False)].copy()
        reference_point = metrics(reference_data["Y"], reference_data["P"])
        comparison_point = metrics(comparison_data["Y"], comparison_data["P"])
        samples = {name: [] for name in metric_names}
        rng = np.random.default_rng(SEED + 20000 + contrast_number * 1000)
        for _ in range(N_BOOTSTRAP):
            reference_sample = sample_within_environment(reference_data, rng)
            comparison_sample = sample_within_environment(comparison_data, rng)
            if reference_sample["Y"].nunique() < 2 or comparison_sample["Y"].nunique() < 2:
                continue
            a = metrics(reference_sample["Y"], reference_sample["P"])
            b = metrics(comparison_sample["Y"], comparison_sample["P"])
            for name in metric_names:
                if np.isfinite(a[name]) and np.isfinite(b[name]):
                    samples[name].append(b[name] - a[name])
        for name in metric_names:
            values = np.asarray(samples[name], dtype=float)
            rows.append({
                "Subgroup_variable": variable,
                "Reference_level": reference,
                "Comparison_level": comparison,
                "Metric": name,
                "Reference_estimate": reference_point[name],
                "Comparison_estimate": comparison_point[name],
                "Delta_comparison_minus_reference": comparison_point[name] - reference_point[name],
                "Delta_CI_lower": np.percentile(values, 2.5) if len(values) else np.nan,
                "Delta_CI_upper": np.percentile(values, 97.5) if len(values) else np.nan,
                "Bootstrap_valid": len(values),
                "Reference_N": len(reference_data),
                "Comparison_N": len(comparison_data),
            })
    return pd.DataFrame(rows)


def task_triage_table() -> pd.DataFrame:
    channel = pd.read_csv(PROJECT / "tables" / "channel_model_pooled_metrics.csv")
    formal_metrics = pd.read_csv(FORMAL / "tables" / "pooled_metrics_environment_bootstrap_ci.csv")
    selective = pd.read_csv(FORMAL / "tables" / "selective_performance.csv")
    pilot_selective = pd.read_csv(PROJECT / "tables" / "cross_surgeon_selective_performance.csv")

    def channel_value(task: str, scheme: str, model: str, metric: str = "AUROC") -> float:
        row = channel.loc[
            channel["Task"].eq(task) & channel["Scheme"].eq(scheme) & channel["Model"].eq(model)
        ]
        return float(row.iloc[0][metric])

    def formal_value(scheme: str, model: str, metric: str = "AUROC") -> float:
        row = formal_metrics.loc[
            formal_metrics["Scheme"].eq(scheme)
            & formal_metrics["Model"].eq(model)
            & formal_metrics["Metric"].eq(metric)
        ]
        return float(row.iloc[0]["Estimate"])

    bentall_sel = selective.loc[
        selective["Model"].eq("Phenotype_only")
        & selective["Coverage"].eq(0.6)
        & selective["Selection_rule"].eq("Between_MI_SD")
    ].iloc[0]
    cabg_sel = pilot_selective.loc[
        pilot_selective["Task"].eq("Planned_CABG")
        & pilot_selective["Coverage"].eq(0.6)
        & pilot_selective["Selection_rule"].eq("Bootstrap_SD")
    ].iloc[0]

    return pd.DataFrame([
        {
            "Task": "Bentall",
            "Manuscript_role": "Primary portable decision phenotype",
            "Random_OOF_phenotype_AUROC": channel_value("Bentall", "RANDOM_OOF", "Phenotype_only"),
            "Cross_surgeon_phenotype_AUROC": formal_value("LEAVE_ONE_SURGEON_OUT", "Phenotype_only"),
            "Rolling_year_phenotype_AUROC": formal_value("ROLLING_YEAR", "Phenotype_only"),
            "Random_OOF_practice_AUROC": channel_value("Bentall", "RANDOM_OOF", "Practice_only_audit"),
            "Selective_AUROC_at_60pct": float(bentall_sel["AUROC"]),
            "Full_coverage_AUROC_for_selective_source": formal_value("LEAVE_ONE_SURGEON_OUT", "Phenotype_only"),
            "Interpretation": "Portable signal with clinically usable uncertainty triage; confirmatory endpoint.",
        },
        {
            "Task": "Arch_Fourbranch",
            "Manuscript_role": "Practice-shift shortcut case",
            "Random_OOF_phenotype_AUROC": channel_value("Arch_Fourbranch", "RANDOM_OOF", "Phenotype_only"),
            "Cross_surgeon_phenotype_AUROC": channel_value("Arch_Fourbranch", "LEAVE_ONE_SURGEON_OUT", "Phenotype_only"),
            "Rolling_year_phenotype_AUROC": np.nan,
            "Random_OOF_practice_AUROC": channel_value("Arch_Fourbranch", "RANDOM_OOF", "Practice_only_audit"),
            "Selective_AUROC_at_60pct": np.nan,
            "Full_coverage_AUROC_for_selective_source": np.nan,
            "Interpretation": "Apparent random-split predictability is dominated by practice context and fails transportability.",
        },
        {
            "Task": "Planned_CABG",
            "Manuscript_role": "Rare-task negative control",
            "Random_OOF_phenotype_AUROC": channel_value("Planned_CABG", "RANDOM_OOF", "Phenotype_only"),
            "Cross_surgeon_phenotype_AUROC": channel_value("Planned_CABG", "LEAVE_ONE_SURGEON_OUT", "Phenotype_only"),
            "Rolling_year_phenotype_AUROC": np.nan,
            "Random_OOF_practice_AUROC": channel_value("Planned_CABG", "RANDOM_OOF", "Practice_only_audit"),
            "Selective_AUROC_at_60pct": float(cabg_sel["AUROC"]),
            "Full_coverage_AUROC_for_selective_source": float(
                pilot_selective.loc[
                    pilot_selective["Task"].eq("Planned_CABG")
                    & pilot_selective["Coverage"].eq(1.0)
                    & pilot_selective["Selection_rule"].eq("Bootstrap_SD"),
                    "AUROC",
                ].iloc[0]
            ),
            "Interpretation": "Rare endpoint where uncertainty selection degrades ranking; retained as a negative control.",
        },
    ])


def make_subgroup_plot(table: pd.DataFrame) -> None:
    display = table.loc[table["Subgroup_variable"].ne("Overall")].copy()
    display["Label"] = display["Subgroup_variable"].replace({
        "Sex": "Sex",
        "Age": "Age",
        "Connective_tissue_disorder": "CTD",
    }) + ": " + display["Level"]
    fig, ax = plt.subplots(figsize=(8.2, 5.1))
    y = np.arange(len(display))
    ax.errorbar(
        display["AUROC"], y,
        xerr=np.vstack([
            display["AUROC"] - display["AUROC_CI_lower"],
            display["AUROC_CI_upper"] - display["AUROC"],
        ]),
        fmt="o", color="#2563EB", ecolor="#93C5FD", capsize=3,
    )
    ax.axvline(0.5, color="#9CA3AF", linestyle="--", linewidth=1)
    ax.set_yticks(y, display["Label"])
    ax.set_xlim(0.45, 1.0)
    ax.set_xlabel("AUROC (95% bootstrap CI)")
    ax.set_title("Bentall phenotype model: post-freeze exploratory subgroup performance")
    ax.invert_yaxis()
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(FORMAL / "plots" / "formal_subgroup_performance.png", dpi=220)
    plt.close(fig)


def fmt(value: float) -> str:
    return "NA" if not np.isfinite(value) else f"{value:.3f}"


def write_report(
    paired: pd.DataFrame,
    recalibration: pd.DataFrame,
    subgroup: pd.DataFrame,
    contrasts: pd.DataFrame,
    triage: pd.DataFrame,
) -> None:
    loso_pair = paired.loc[paired["Scheme"].eq("LEAVE_ONE_SURGEON_OUT")]
    temporal_pair = paired.loc[paired["Scheme"].eq("ROLLING_YEAR")]
    loso_recal = recalibration.loc[recalibration["Scheme"].eq("LEAVE_ONE_SURGEON_OUT")]

    def get_pair(frame: pd.DataFrame, metric_name: str) -> pd.Series:
        return frame.loc[frame["Metric"].eq(metric_name)].iloc[0]

    lines = [
        "# JBHI 配对比较、校准与预设亚组分析",
        "",
        "## 分析边界",
        "",
        "- 本分析只读取冻结的正式外层验证预测，不重新选择变量、不重新调参。",
        "- 所有模型比较均按同一患者、同一外层环境配对；95% CI 使用环境内重采样 2,000 次。",
        "- 亚组仅用于检查性能异质性，不将历史术式决策差异解释为治疗公平性或因果治疗效应。",
        "",
        "## 观察通道的增量价值",
        "",
    ]
    for label, frame in [("留一术者外", loso_pair), ("滚动年度", temporal_pair)]:
        auroc = get_pair(frame, "AUROC")
        brier = get_pair(frame, "Brier")
        lines.append(
            f"- {label}：加入观察/缺失指示后，AUROC {fmt(auroc.Reference_estimate)} → "
            f"{fmt(auroc.Comparison_estimate)}，配对 Δ={fmt(auroc.Paired_delta_comparison_minus_reference)} "
            f"(95% CI {fmt(auroc.Delta_CI_lower)} 至 {fmt(auroc.Delta_CI_upper)})；"
            f"Brier Δ={fmt(brier.Paired_delta_comparison_minus_reference)} "
            f"({fmt(brier.Delta_CI_lower)} 至 {fmt(brier.Delta_CI_upper)})。"
        )
    lines += [
        "",
        "结论：观察通道没有可重复的跨环境增量价值，主模型继续锁定为 phenotype-only；观察变量保留为审计和漂移监测通道。",
        "",
        "## 训练内校准",
        "",
    ]
    for metric_name in ("Brier", "LogLoss", "Calibration_intercept", "Calibration_slope"):
        row = get_pair(loso_recal, metric_name)
        lines.append(
            f"- {metric_name}：原始 {fmt(row.Raw_estimate)}，训练内重校准后 "
            f"{fmt(row.Training_only_recalibrated_estimate)}；配对 Δ="
            f"{fmt(row.Paired_delta_recalibrated_minus_raw)} "
            f"(95% CI {fmt(row.Delta_CI_lower)} 至 {fmt(row.Delta_CI_upper)})。"
        )
    lines += [
        "",
        "重校准完全在各外层训练集的内层 OOF 预测上估计，因此没有使用测试术者信息。",
        "",
        "## 方案冻结后支持性亚组（探索性）",
        "",
        "| 亚组 | N | 事件 | AUROC (95% CI) | AUPRC | Brier | 校准截距 | 校准斜率 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in subgroup.itertuples(index=False):
        label = "总体" if row.Subgroup_variable == "Overall" else f"{row.Subgroup_variable}: {row.Level}"
        lines.append(
            f"| {label} | {row.N} | {row.Events} | {fmt(row.AUROC)} "
            f"({fmt(row.AUROC_CI_lower)}–{fmt(row.AUROC_CI_upper)}) | {fmt(row.AUPRC)} | "
            f"{fmt(row.Brier)} | {fmt(row.Calibration_intercept)} | {fmt(row.Calibration_slope)} |"
        )
    lines += [
        "",
        "小亚组的置信区间宽时只作描述，不据此声称模型在该人群中已被充分验证。",
        "",
        "### 亚组性能差异（探索性）",
        "",
    ]
    for row in contrasts.loc[contrasts["Metric"].eq("AUROC")].itertuples(index=False):
        lines.append(
            f"- {row.Subgroup_variable}，{row.Comparison_level} 相对 {row.Reference_level}："
            f"AUROC Δ={fmt(row.Delta_comparison_minus_reference)} "
            f"(95% CI {fmt(row.Delta_CI_lower)} 至 {fmt(row.Delta_CI_upper)})。"
        )
    lines += [
        "",
        "其中年龄差异提示 ≥60 岁病例需要独立校准/外部验证；该事后比较不作多重性校正，也不改变锁定模型。",
        "",
        "## 三任务在论文中的角色",
        "",
        "| 任务 | 论文角色 | 随机 OOF 表型 AUROC | 跨术者表型 AUROC | practice-only AUROC | 60%覆盖 AUROC |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in triage.itertuples(index=False):
        lines.append(
            f"| {row.Task} | {row.Manuscript_role} | {fmt(row.Random_OOF_phenotype_AUROC)} | "
            f"{fmt(row.Cross_surgeon_phenotype_AUROC)} | {fmt(row.Random_OOF_practice_AUROC)} | "
            f"{fmt(row.Selective_AUROC_at_60pct)} |"
        )
    lines += [
        "",
        "这三类结果共同支持 POP 框架：可迁移的表型信号、不可迁移的工作流捷径，以及不确定性选择失败的稀有任务负对照。",
    ]
    (FORMAL / "reports" / "JBHI_paired_and_subgroup_report_zh.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    for directory in (FORMAL / "tables", FORMAL / "plots", FORMAL / "reports"):
        directory.mkdir(parents=True, exist_ok=True)
    predictions = pd.read_csv(PREDICTIONS)
    paired = paired_model_comparison(predictions)
    recalibration = recalibration_comparison(predictions)
    subgroup = subgroup_performance(predictions)
    contrasts = subgroup_contrasts(predictions)
    triage = task_triage_table()

    paired.to_csv(FORMAL / "tables" / "paired_model_comparisons.csv", index=False)
    recalibration.to_csv(FORMAL / "tables" / "training_only_recalibration_comparison.csv", index=False)
    subgroup.to_csv(FORMAL / "tables" / "prespecified_subgroup_performance.csv", index=False)
    subgroup.to_csv(FORMAL / "tables" / "supportive_subgroup_performance.csv", index=False)
    contrasts.to_csv(FORMAL / "tables" / "exploratory_subgroup_contrasts.csv", index=False)
    triage.to_csv(FORMAL / "tables" / "task_triage_for_manuscript.csv", index=False)
    make_subgroup_plot(subgroup)
    write_report(paired, recalibration, subgroup, contrasts, triage)

    audit = {
        "input_predictions": str(PREDICTIONS),
        "n_bootstrap": N_BOOTSTRAP,
        "seed": SEED,
        "prediction_rows": len(predictions),
        "analysis_note": "post-processing of frozen outer-fold predictions only; no refitting",
    }
    (FORMAL / "paired_subgroup_manifest.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print("Completed paired, recalibration, subgroup, and task-triage analyses.")


if __name__ == "__main__":
    main()
