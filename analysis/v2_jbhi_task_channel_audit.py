#!/usr/bin/env python3
"""Three-task POP channel audit used by the ATAAD/JBHI manuscript.

This analysis supplies the exploratory task-triage evidence that precedes the
formal Bentall-only validation.  It separates strictly preoperative phenotype
(X), observation/missingness indicators (M), and practice context (C) for:

* Bentall (candidate transportable decision phenotype),
* four-branch total-arch reconstruction (practice-shift shortcut case), and
* planned CABG (rare-task negative control).

All preprocessing and regularisation selection are fitted inside each outer
fold.  Surgeon and year are audit-only context variables and are never part of
the final patient-phenotype score.
"""

from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/ataad_jbhi_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

REPOSITORY = Path(__file__).resolve().parents[1]
RAW_XLSX = Path(os.environ.get("ATAAD_POP_DATA", REPOSITORY / "data" / "authorized_input.xlsx"))
FEATURE_METADATA = Path(
    os.environ.get("ATAAD_POP_FEATURE_METADATA", REPOSITORY / "config" / "feature_metadata.csv")
)
PROJECT = Path(os.environ.get("ATAAD_POP_OUTPUT_ROOT", REPOSITORY))
SHEET_NAME = os.environ.get("ATAAD_POP_SHEET", "main")
SEED = int(os.environ.get("ATAAD_POP_SEED", "20260814"))
AUDIT = PROJECT / "artifacts" / "task_channel_audit"
TABLES = AUDIT / "tables"
PLOTS = AUDIT / "plots"
PREDICTIONS = AUDIT / "predictions"
REPORTS = AUDIT / "reports"

RANDOM_FOLDS = 5
INNER_FOLDS = 3
CS = np.array([0.001, 0.01, 0.1, 1.0])

TASKS = {
    "Bentall": {"target": "D1_Bentall", "eligible": None},
    "Arch_Fourbranch": {"target": "D2_Fourbranch", "eligible": "D2_Arch_eligible"},
    "Planned_CABG": {"target": "D3_Planned_CABG", "eligible": "D3_CABG_eligible"},
}

CHANNEL_MODELS = {
    "Phenotype_only": ("phenotype",),
    "Observation_only": ("observation",),
    "Practice_only_audit": ("practice",),
    "Phenotype_plus_observation": ("phenotype", "observation"),
    "Phenotype_plus_practice_audit": ("phenotype", "practice"),
    "Full_three_channel_audit": ("phenotype", "observation", "practice"),
}


@dataclass
class TaskData:
    frame: pd.DataFrame
    y: np.ndarray
    groups: np.ndarray


def ensure_dirs() -> None:
    for path in (TABLES, PLOTS, REPORTS):
        path.mkdir(parents=True, exist_ok=True)


def numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def derive_targets(frame: pd.DataFrame) -> pd.DataFrame:
    """Create recorded-decision targets without using postoperative outcomes."""
    out = frame.copy()
    mechanical = numeric(out, "Bentall_mechanic_valve").fillna(0)
    biological = numeric(out, "Bentall_bio_valve").fillna(0)
    out["D1_Bentall"] = ((mechanical == 1) | (biological == 1)).astype(int)

    fourbranch = numeric(out, "Totalarch_fourbranch").fillna(0)
    island = numeric(out, "Totalarch_island").fillna(0)
    arch_eligible = (fourbranch == 1) | (island == 1)
    out["D2_Arch_eligible"] = arch_eligible.astype(int)
    out["D2_Fourbranch"] = np.where(arch_eligible, (fourbranch == 1).astype(int), np.nan)

    dissection = numeric(out, "CABG_for_dissection").fillna(0)
    cad = numeric(out, "CABG_for_CAD").fillna(0)
    salvage = numeric(out, "Salvage_CABG_after_declamp").fillna(0)
    any_cabg = numeric(out, "CABG").fillna(0)
    planned = (dissection == 1) | (cad == 1)
    salvage_only = (salvage == 1) & (~planned)
    ambiguous = (any_cabg == 1) & (~planned) & (salvage == 0)
    cabg_eligible = ~(salvage_only | ambiguous)
    out["D3_CABG_eligible"] = cabg_eligible.astype(int)
    out["D3_Planned_CABG"] = np.where(cabg_eligible, planned.astype(int), np.nan)
    return out


def load_data() -> tuple[pd.DataFrame, list[str], list[str], pd.DataFrame]:
    """Load the locked cohort and reproduce the 124-variable feasibility set."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*Conditional Formatting extension.*")
        if RAW_XLSX.suffix.lower() == ".csv":
            raw = pd.read_csv(RAW_XLSX)
        else:
            raw = pd.read_excel(RAW_XLSX, sheet_name=SHEET_NAME)
    raw.columns = raw.columns.astype(str).str.strip()
    raw.insert(0, "Row_ID", np.arange(1, len(raw) + 1, dtype=int))
    required_decisions = {"Bentall_mechanic_valve", "Bentall_bio_valve", "Totalarch_fourbranch",
                          "Totalarch_island", "CABG_for_dissection", "CABG_for_CAD",
                          "Salvage_CABG_after_declamp", "CABG"}
    absent = required_decisions.difference(raw.columns)
    if absent:
        raise ValueError(f"Missing decision columns: {sorted(absent)}")
    raw = derive_targets(raw)
    raw["Date_of_surgery"] = pd.to_datetime(raw["Date_of_surgery"], errors="coerce")
    raw["Year"] = raw["Date_of_surgery"].dt.year.astype("Int64")
    raw["Surgeon"] = numeric(raw, "Surgeon").astype("Int64")
    raw = raw.loc[numeric(raw, "Missingctimage").fillna(0).ne(1)].copy()
    raw = raw.loc[raw["Year"].notna() & raw["Surgeon"].notna()].reset_index(drop=True)

    metadata = pd.read_csv(FEATURE_METADATA)
    required = {"Variable", "Domain", "Imputation_type", "Missing_indicator_for_sensitivity"}
    missing_columns = sorted(required.difference(metadata.columns))
    if missing_columns:
        raise ValueError(f"Feature metadata missing columns: {missing_columns}")

    # The feasibility audit used the Stage-01 globally retained set.  Four
    # columns explicitly marked as fold-specific candidates are therefore not
    # included here; the formal validation reconsiders them inside each fold.
    retained = ~metadata["Imputation_type"].astype(str).str.startswith("Fold-specific")
    audit_metadata = metadata.loc[retained].drop_duplicates("Variable").reset_index(drop=True)
    phenotype_features = audit_metadata["Variable"].tolist()
    observation_features = audit_metadata.loc[
        audit_metadata["Missing_indicator_for_sensitivity"]
        .astype(str)
        .str.lower()
        .eq("true"),
        "Variable",
    ].tolist()
    missing_from_source = [column for column in phenotype_features if column not in raw]
    if missing_from_source:
        raise ValueError(f"Audit features absent from source workbook: {missing_from_source}")
    for column in phenotype_features:
        raw[column] = pd.to_numeric(raw[column], errors="coerce")
    indicators = pd.DataFrame(
        {f"{column}__MISSING": raw[column].isna().astype(int) for column in observation_features},
        index=raw.index,
    )
    raw = pd.concat([raw, indicators], axis=1)
    return raw, phenotype_features, observation_features, audit_metadata


def safe_metrics(y: np.ndarray | pd.Series, p: np.ndarray | pd.Series) -> dict[str, object]:
    y_array = np.asarray(y, dtype=int)
    p_array = np.clip(np.asarray(p, dtype=float), 1e-8, 1 - 1e-8)
    two_classes = np.unique(y_array).size == 2
    return {
        "N": int(len(y_array)),
        "Events": int(y_array.sum()),
        "Prevalence": float(y_array.mean()) if len(y_array) else np.nan,
        "AUROC": float(roc_auc_score(y_array, p_array)) if two_classes else np.nan,
        "AUPRC": float(average_precision_score(y_array, p_array)) if two_classes else np.nan,
        "Brier": float(brier_score_loss(y_array, p_array)) if len(y_array) else np.nan,
        "LogLoss": float(log_loss(y_array, p_array, labels=[0, 1])) if len(y_array) else np.nan,
        "Outcome_has_two_classes": bool(two_classes),
    }


def make_preprocessor(
    channels: tuple[str, ...],
    phenotype_features: list[str],
    missing_columns: list[str],
) -> ColumnTransformer:
    transformers: list[tuple] = []
    if "phenotype" in channels:
        transformers.append((
            "phenotype",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]),
            phenotype_features,
        ))
    if "observation" in channels:
        transformers.append((
            "observation",
            Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
                ("scaler", StandardScaler()),
            ]),
            missing_columns,
        ))
    if "practice" in channels:
        transformers.extend([
            (
                "year",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                ["Year"],
            ),
            (
                "surgeon",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first")),
                ]),
                ["Surgeon"],
            ),
        ])
    return ColumnTransformer(transformers=transformers, remainder="drop")


def fit_channel_model(
    train: pd.DataFrame,
    y: np.ndarray,
    channels: tuple[str, ...],
    phenotype_features: list[str],
    missing_columns: list[str],
    seed: int,
) -> Pipeline:
    """Fit one channel combination with inner-CV regularisation selection."""
    counts = np.bincount(np.asarray(y, dtype=int), minlength=2)
    possible_folds = int(min(INNER_FOLDS, counts.min()))
    if possible_folds >= 2:
        inner = StratifiedKFold(n_splits=possible_folds, shuffle=True, random_state=seed)
        classifier = LogisticRegressionCV(
            Cs=CS,
            cv=inner,
            scoring="neg_log_loss",
            solver="liblinear",
            penalty="l2",
            max_iter=4000,
            refit=True,
        )
    else:
        classifier = LogisticRegression(C=0.01, solver="liblinear", max_iter=4000)
    model = Pipeline([
        ("preprocess", make_preprocessor(channels, phenotype_features, missing_columns)),
        ("classifier", classifier),
    ])
    model.fit(train, y)
    return model


def chosen_c(model: Pipeline) -> float:
    classifier = model.named_steps["classifier"]
    if hasattr(classifier, "C_"):
        return float(np.ravel(classifier.C_)[0])
    return float(classifier.C)


def evaluate_scheme(
    task: str,
    task_data: TaskData,
    scheme: str,
    phenotype_features: list[str],
    missing_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame, y = task_data.frame, task_data.y
    if scheme == "RANDOM_OOF":
        offset = list(TASKS).index(task) * 100
        splitter = StratifiedKFold(
            n_splits=RANDOM_FOLDS,
            shuffle=True,
            random_state=SEED + offset,
        )
        splits = list(splitter.split(frame, y))
    elif scheme == "LEAVE_ONE_SURGEON_OUT":
        splits = list(LeaveOneGroupOut().split(frame, y, task_data.groups))
    else:
        raise ValueError(f"Unknown scheme: {scheme}")

    prediction_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    for fold, (train_index, test_index) in enumerate(splits, start=1):
        y_train, y_test = y[train_index], y[test_index]
        environment = (
            f"fold_{fold}"
            if scheme == "RANDOM_OOF"
            else f"surgeon_{task_data.groups[test_index][0]}"
        )
        baseline = np.repeat(np.clip(y_train.mean(), 1e-6, 1 - 1e-6), len(test_index))
        for local_index, probability in zip(test_index, baseline, strict=True):
            prediction_rows.append({
                "Task": task,
                "Scheme": scheme,
                "Model": "Prevalence_only",
                "Fold": fold,
                "Environment": environment,
                "Row_ID": int(frame.iloc[local_index]["Row_ID"]),
                "Year": int(frame.iloc[local_index]["Year"]),
                "Surgeon": int(frame.iloc[local_index]["Surgeon"]),
                "Y": int(y[local_index]),
                "P": float(probability),
                "Selected_C": np.nan,
            })
        fold_rows.append({
            "Task": task,
            "Scheme": scheme,
            "Model": "Prevalence_only",
            "Fold": fold,
            "Environment": environment,
            **safe_metrics(y_test, baseline),
            "Selected_C": np.nan,
        })

        for model_number, (model_name, channels) in enumerate(CHANNEL_MODELS.items()):
            model = fit_channel_model(
                frame.iloc[train_index],
                y_train,
                channels,
                phenotype_features,
                missing_columns,
                SEED + fold * 100 + model_number,
            )
            probability = model.predict_proba(frame.iloc[test_index])[:, 1]
            c_value = chosen_c(model)
            for local_index, value in zip(test_index, probability, strict=True):
                prediction_rows.append({
                    "Task": task,
                    "Scheme": scheme,
                    "Model": model_name,
                    "Fold": fold,
                    "Environment": environment,
                    "Row_ID": int(frame.iloc[local_index]["Row_ID"]),
                    "Year": int(frame.iloc[local_index]["Year"]),
                    "Surgeon": int(frame.iloc[local_index]["Surgeon"]),
                    "Y": int(y[local_index]),
                    "P": float(value),
                    "Selected_C": c_value,
                })
            fold_rows.append({
                "Task": task,
                "Scheme": scheme,
                "Model": model_name,
                "Fold": fold,
                "Environment": environment,
                **safe_metrics(y_test, probability),
                "Selected_C": c_value,
            })
    return pd.DataFrame(prediction_rows), pd.DataFrame(fold_rows)


def prevalence_tables(raw: pd.DataFrame) -> None:
    by_surgeon: list[dict[str, object]] = []
    by_year: list[dict[str, object]] = []
    for task, specification in TASKS.items():
        eligible = (
            pd.Series(True, index=raw.index)
            if specification["eligible"] is None
            else raw[specification["eligible"]].eq(1)
        )
        local = raw.loc[eligible]
        for surgeon, group in local.groupby("Surgeon"):
            by_surgeon.append({
                "Task": task,
                "Surgeon": int(surgeon),
                "N": len(group),
                "Events": int(group[specification["target"]].sum()),
                "Rate": float(group[specification["target"]].mean()),
            })
        for year, group in local.groupby("Year"):
            by_year.append({
                "Task": task,
                "Year": int(year),
                "N": len(group),
                "Events": int(group[specification["target"]].sum()),
                "Rate": float(group[specification["target"]].mean()),
            })
    pd.DataFrame(by_surgeon).to_csv(TABLES / "task_prevalence_by_surgeon.csv", index=False)
    pd.DataFrame(by_year).to_csv(TABLES / "task_prevalence_by_year.csv", index=False)


def channel_validation(
    raw: pd.DataFrame,
    phenotype_features: list[str],
    observation_features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    missing_columns = [f"{column}__MISSING" for column in observation_features]
    predictions: list[pd.DataFrame] = []
    fold_metrics: list[pd.DataFrame] = []
    for task, specification in TASKS.items():
        eligible = (
            pd.Series(True, index=raw.index)
            if specification["eligible"] is None
            else raw[specification["eligible"]].eq(1)
        )
        local = raw.loc[eligible].copy().reset_index(drop=True)
        task_data = TaskData(
            frame=local,
            y=local[specification["target"]].astype(int).to_numpy(),
            groups=local["Surgeon"].astype(int).to_numpy(),
        )
        for scheme in ("RANDOM_OOF", "LEAVE_ONE_SURGEON_OUT"):
            local_predictions, local_folds = evaluate_scheme(
                task,
                task_data,
                scheme,
                phenotype_features,
                missing_columns,
            )
            predictions.append(local_predictions)
            fold_metrics.append(local_folds)
    prediction_frame = pd.concat(predictions, ignore_index=True)
    fold_frame = pd.concat(fold_metrics, ignore_index=True)
    pooled_rows = []
    for keys, group in prediction_frame.groupby(["Task", "Scheme", "Model"], sort=False):
        pooled_rows.append({
            "Task": keys[0],
            "Scheme": keys[1],
            "Model": keys[2],
            **safe_metrics(group["Y"], group["P"]),
        })
    pooled = pd.DataFrame(pooled_rows)
    if os.environ.get("ATAAD_POP_SAVE_CASE_PREDICTIONS") == "1":
        PREDICTIONS.mkdir(parents=True, exist_ok=True)
        prediction_frame.to_csv(PREDICTIONS / "channel_model_out_of_fold_predictions.csv", index=False)
    fold_frame.to_csv(TABLES / "channel_model_fold_metrics.csv", index=False)
    pooled.to_csv(TABLES / "channel_model_pooled_metrics.csv", index=False)
    return prediction_frame, fold_frame, pooled


def make_plot(pooled: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    local = pooled.loc[pooled["Model"].ne("Prevalence_only")].copy()
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
    for axis, scheme in zip(axes, ("RANDOM_OOF", "LEAVE_ONE_SURGEON_OUT"), strict=True):
        subset = local.loc[local["Scheme"].eq(scheme)]
        sns.barplot(data=subset, x="Task", y="AUROC", hue="Model", ax=axis)
        axis.set_title(scheme.replace("_", " ").title())
        axis.set_ylim(0, 1)
        axis.tick_params(axis="x", rotation=10)
        if scheme != "RANDOM_OOF":
            legend = axis.get_legend()
            if legend is not None:
                legend.remove()
    axes[0].legend(fontsize=7, loc="lower right")
    fig.suptitle("POP channel audit: random versus cross-surgeon discrimination")
    fig.tight_layout()
    fig.savefig(PLOTS / "channel_models_random_vs_cross_surgeon.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_report(raw: pd.DataFrame, pooled: pd.DataFrame) -> None:
    lines = [
        "# Three-task POP channel audit",
        "",
        f"Eligible CTA cohort: {len(raw):,}; surgeons: {raw['Surgeon'].nunique()}; "
        f"years: {int(raw['Year'].min())}–{int(raw['Year'].max())}.",
        "",
        "All imputation, scaling, encoding, and regularisation selection were fitted inside each outer fold.",
        "",
        "| Task | Validation | Phenotype AUROC | Observation AUROC | Practice AUROC |",
        "|---|---|---:|---:|---:|",
    ]
    for task in TASKS:
        for scheme in ("RANDOM_OOF", "LEAVE_ONE_SURGEON_OUT"):
            local = pooled.loc[pooled["Task"].eq(task) & pooled["Scheme"].eq(scheme)]
            value = {
                row.Model: row.AUROC for row in local.itertuples(index=False)
            }
            lines.append(
                f"| {task} | {scheme} | {value['Phenotype_only']:.3f} | "
                f"{value['Observation_only']:.3f} | {value['Practice_only_audit']:.3f} |"
            )
    lines += [
        "",
        "Interpretation is task-specific: Bentall proceeds to formal transport validation; "
        "four-branch reconstruction is retained as a practice-shortcut case; planned CABG "
        "is retained as a rare-task negative control.",
    ]
    (REPORTS / "task_channel_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_dirs()
    raw, phenotype_features, observation_features, metadata = load_data()
    metadata.to_csv(TABLES / "task_channel_feature_metadata.csv", index=False)
    prevalence_tables(raw)
    _, folds, pooled = channel_validation(raw, phenotype_features, observation_features)
    make_plot(pooled)
    write_report(raw, pooled)
    manifest = {
        "created_utc": datetime.now(UTC).isoformat(),
        "eligible_rows": len(raw),
        "phenotype_features": len(phenotype_features),
        "observation_indicators": len(observation_features),
        "tasks": list(TASKS),
        "channel_models": list(CHANNEL_MODELS),
        "validation": ["RANDOM_OOF", "LEAVE_ONE_SURGEON_OUT"],
        "outer_fold_metrics_rows": len(folds),
        "seed": SEED,
    }
    (AUDIT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "eligible_rows": len(raw),
        "pooled_metric_rows": len(pooled),
        "output": str(AUDIT),
    }, ensure_ascii=False))


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Found unknown categories in columns.*during transform.*",
            category=UserWarning,
        )
        main()
