#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import subprocess
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Literal, Sequence

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

import mamut
from mamut import Mamut
from mamut.evidence import (
    default_baseline_estimators,
    detect_leakage_risks,
    evaluate_estimators_on_split,
)

CompetitionName = Literal["spaceship-titanic"]
RecipeName = Literal[
    "raw",
    "spaceship_inductive",
    "spaceship_cohort",
    "spaceship_basic",
    "spaceship_inductive_v2",
    "spaceship_cohort_v2",
    "spaceship_competition_v3",
]
ValidationProtocol = Literal["grouped", "row"]
GroupScope = Literal["passenger", "household_component"]
BenchmarkStage = Literal["diagnostic", "development", "confirmation"]
OutputFormat = Literal["markdown", "json", "csv"]

DEFAULT_CACHE_DIR = Path(".cache/mamut/kaggle")
DEFAULT_OUTPUT_DIR = Path(".cache/mamut/benchmark-results")
SPACESHIP_FILES = ("train.csv", "test.csv", "sample_submission.csv")
RAW_DROP_COLUMNS = ("Transported",)
SPACESHIP_BASIC_DROP_COLUMNS = (
    "Transported",
    "PassengerId",
    "Name",
    "Cabin",
    "PassengerGroup",
)
DEFAULT_EXCLUDED_MODELS = ()
DEFAULT_CONFIRMATION_SEED = 20260524
FEATURE_RECIPE_VERSION = "3"

warnings.filterwarnings(
    "ignore",
    message="Found unknown categories in columns .* during transform.*",
    module="sklearn\\.preprocessing\\._encoders",
)
warnings.filterwarnings(
    "ignore",
    message="l1_ratio parameter is only used when penalty is 'elasticnet'.*",
    module="sklearn\\.linear_model\\._logistic",
)


@dataclass(frozen=True)
class BenchmarkConfig:
    competition: CompetitionName
    recipe: RecipeName
    runs: int
    n_iterations: int
    random_state: int
    holdout_size: float
    validation_protocol: ValidationProtocol
    score_metric: str
    search_profile: str
    selection_strategy: str
    selection_cv_splits: int
    selection_cv_repeats: int
    selection_practical_margin: float
    preprocessing_profile: str
    optimization_method: str
    excluded_models: tuple[str, ...]
    included_models: tuple[str, ...] | None = None
    group_scope: GroupScope = "passenger"
    stage: BenchmarkStage = "diagnostic"
    confirmation_size: float = 0.2
    confirmation_seed: int = DEFAULT_CONFIRMATION_SEED
    campaign_id: str = "exploration"
    max_runtime_seconds: float | None = None


@dataclass(frozen=True)
class BenchmarkMetadata:
    generated_at_utc: str
    mamut_version: str
    git_commit: str
    train_sha256: str
    test_sha256: str
    sample_submission_sha256: str
    git_branch: str
    git_dirty: bool
    python_version: str
    feature_recipe_version: str


def ensure_competition_data(
    competition: CompetitionName,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    *,
    force_download: bool = False,
) -> Path:
    if competition != "spaceship-titanic":
        raise ValueError("Only spaceship-titanic is supported in benchmark v1.")

    data_dir = cache_dir / competition
    data_dir.mkdir(parents=True, exist_ok=True)

    missing_files = [
        file_name
        for file_name in SPACESHIP_FILES
        if not (data_dir / file_name).exists()
    ]
    if missing_files or force_download:
        for file_name in SPACESHIP_FILES:
            command = [
                "kaggle",
                "competitions",
                "download",
                "-c",
                competition,
                "-f",
                file_name,
                "-p",
                str(data_dir),
                "-q",
            ]
            if force_download:
                command.append("--force")
            subprocess.run(command, check=True)

    missing_after_download = [
        file_name
        for file_name in SPACESHIP_FILES
        if not (data_dir / file_name).exists()
    ]
    if missing_after_download:
        raise FileNotFoundError(
            "Missing Kaggle competition files after download: "
            f"{missing_after_download}. Check Kaggle CLI authentication."
        )

    return data_dir


def load_spaceship_titanic(
    data_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    sample_submission = pd.read_csv(data_dir / "sample_submission.csv")

    required_train_columns = {"PassengerId", "Transported"}
    required_test_columns = {"PassengerId"}
    required_submission_columns = ["PassengerId", "Transported"]
    if not required_train_columns.issubset(train.columns):
        raise ValueError(f"train.csv must contain {sorted(required_train_columns)}.")
    if not required_test_columns.issubset(test.columns):
        raise ValueError(f"test.csv must contain {sorted(required_test_columns)}.")
    if list(sample_submission.columns) != required_submission_columns:
        raise ValueError(
            "sample_submission.csv must have columns: "
            f"{required_submission_columns}."
        )

    return train, test, sample_submission


def prepare_spaceship_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    recipe: RecipeName,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    y = train["Transported"].astype(bool).reset_index(drop=True)
    passenger_ids = test["PassengerId"].copy().reset_index(drop=True)

    if recipe == "raw":
        X = train.drop(columns=list(RAW_DROP_COLUMNS)).reset_index(drop=True)
        X_test = test[X.columns].copy().reset_index(drop=True)
        return X, y, X_test, passenger_ids

    if recipe in {
        "spaceship_inductive",
        "spaceship_cohort",
        "spaceship_basic",
        "spaceship_inductive_v2",
        "spaceship_cohort_v2",
        "spaceship_competition_v3",
    }:
        X = _spaceship_features_for_recipe(train, recipe).reset_index(drop=True)
        X_test = _spaceship_features_for_recipe(test, recipe).reset_index(drop=True)
        X_test = X_test.reindex(columns=X.columns)
        return X, y, X_test, passenger_ids

    raise ValueError(
        "recipe must be one of: raw, spaceship_inductive, spaceship_cohort, "
        "spaceship_basic, spaceship_inductive_v2, spaceship_cohort_v2, "
        "spaceship_competition_v3."
    )


def prepare_spaceship_modeling_split(
    modeling: pd.DataFrame,
    holdout: pd.DataFrame,
    *,
    recipe: RecipeName,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    y_modeling = modeling["Transported"].astype(bool).reset_index(drop=True)
    y_holdout = holdout["Transported"].astype(bool).reset_index(drop=True)
    X_modeling = _spaceship_features_for_recipe(modeling, recipe).reset_index(drop=True)
    X_holdout = _spaceship_features_for_recipe(holdout, recipe).reset_index(drop=True)
    X_holdout = X_holdout.reindex(columns=X_modeling.columns)
    return X_modeling, y_modeling, X_holdout, y_holdout


def _spaceship_features_for_recipe(
    frame: pd.DataFrame, recipe: RecipeName
) -> pd.DataFrame:
    if recipe == "raw":
        return frame.drop(columns=list(RAW_DROP_COLUMNS), errors="ignore")
    if recipe == "spaceship_inductive":
        return _spaceship_features(frame, include_cohort_features=False)
    if recipe in {"spaceship_cohort", "spaceship_basic"}:
        return _spaceship_features(frame, include_cohort_features=True)
    if recipe == "spaceship_inductive_v2":
        return _spaceship_features_v2(frame, include_cohort_features=False)
    if recipe == "spaceship_cohort_v2":
        return _spaceship_features_v2(frame, include_cohort_features=True)
    if recipe == "spaceship_competition_v3":
        return _spaceship_features_v2(frame, include_cohort_features=True)
    raise ValueError(
        "recipe must be one of: raw, spaceship_inductive, spaceship_cohort, "
        "spaceship_basic, spaceship_inductive_v2, spaceship_cohort_v2, "
        "spaceship_competition_v3."
    )


def _spaceship_features(
    frame: pd.DataFrame, *, include_cohort_features: bool
) -> pd.DataFrame:
    result = frame.copy()
    passenger_parts = result["PassengerId"].astype("string").str.split("_", expand=True)
    passenger_parts = passenger_parts.reindex(columns=range(2))
    result["PassengerGroup"] = passenger_parts[0].fillna("unknown")
    result["PassengerNumber"] = pd.to_numeric(passenger_parts[1], errors="coerce")
    if include_cohort_features:
        group_sizes = result.groupby("PassengerGroup")["PassengerGroup"].transform(
            "size"
        )
        result["PassengerGroupSize"] = group_sizes
        result["PassengerGroupIsSolo"] = group_sizes.eq(1)

    cabin_parts = result["Cabin"].astype("string").str.split("/", expand=True)
    cabin_parts = cabin_parts.reindex(columns=range(3))
    result["CabinDeck"] = cabin_parts[0].fillna("Unknown")
    result["CabinNumber"] = pd.to_numeric(cabin_parts[1], errors="coerce")
    result["CabinSide"] = cabin_parts[2].fillna("Unknown")

    spending_columns = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    existing_spending_columns = [
        column for column in spending_columns if column in result.columns
    ]
    spending = result[existing_spending_columns].apply(pd.to_numeric, errors="coerce")
    result["SpendingTotal"] = spending.sum(axis=1, min_count=1)
    result["SpendingAny"] = result["SpendingTotal"].gt(0)
    result["SpendingMissingCount"] = spending.isna().sum(axis=1)

    return result.drop(
        columns=[
            column
            for column in SPACESHIP_BASIC_DROP_COLUMNS
            if column in result.columns
        ],
        errors="ignore",
    )


def _spaceship_features_v2(
    frame: pd.DataFrame, *, include_cohort_features: bool
) -> pd.DataFrame:
    """Produce target-free, documented Spaceship Titanic domain features."""
    result = _spaceship_features(frame, include_cohort_features=include_cohort_features)
    source = frame.reset_index(drop=True)
    result = result.reset_index(drop=True)

    names = source["Name"].astype("string")
    result["FamilyName"] = (
        names.str.rsplit(n=1).str[-1].fillna("Unknown").str.strip().str.lower()
    )
    result["NameMissing"] = names.isna()
    if include_cohort_features:
        family_sizes = result.groupby("FamilyName")["FamilyName"].transform("size")
        result["FamilyBatchSize"] = family_sizes
        result["FamilyBatchShared"] = family_sizes.gt(1)

    age = pd.to_numeric(source["Age"], errors="coerce")
    result["AgeMissing"] = age.isna()
    result["IsChild"] = age.lt(13)
    result["AgeBand"] = (
        pd.cut(
            age,
            bins=[-np.inf, 12, 17, 25, 39, 59, np.inf],
            labels=["child", "teen", "young_adult", "adult", "middle_age", "senior"],
        )
        .astype("string")
        .fillna("Unknown")
    )

    cabin_number = pd.to_numeric(result["CabinNumber"], errors="coerce")
    result["CabinMissing"] = source["Cabin"].isna()
    result["CabinNumberBand"] = (
        (cabin_number // 100).astype("Int64").astype("string").fillna("Unknown")
    )

    spending_columns = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    spending = source[spending_columns].apply(pd.to_numeric, errors="coerce")
    observed_spending = spending.fillna(0)
    result["NoSpend"] = observed_spending.sum(axis=1).eq(0)
    result["AmenitiesUsed"] = observed_spending.gt(0).sum(axis=1)
    result["ServiceSpend"] = observed_spending[["RoomService", "FoodCourt"]].sum(axis=1)
    result["LuxurySpend"] = observed_spending[["ShoppingMall", "Spa", "VRDeck"]].sum(
        axis=1
    )
    result["LogSpendingTotal"] = np.log1p(observed_spending.sum(axis=1))
    cryo_sleep = source["CryoSleep"].astype("boolean").fillna(False)
    result["CryoSpendConflict"] = cryo_sleep & ~result["NoSpend"]

    return result


def passenger_groups(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["PassengerId"]
        .astype("string")
        .str.split("_", n=1, expand=True)[0]
        .fillna("unknown")
    )


def family_names(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["Name"]
        .astype("string")
        .str.rsplit(n=1)
        .str[-1]
        .str.strip()
        .str.lower()
        .fillna("unknown")
    )


def household_components(frame: pd.DataFrame) -> pd.Series:
    """Join passenger groups sharing a known family name for sensitivity testing."""
    passenger = passenger_groups(frame).reset_index(drop=True)
    family = family_names(frame).reset_index(drop=True)
    parents: dict[str, str] = {}

    def find(value: str) -> str:
        parents.setdefault(value, value)
        if parents[value] != value:
            parents[value] = find(parents[value])
        return parents[value]

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    for group, surname in zip(passenger, family):
        group_key = f"group:{group}"
        find(group_key)
        if pd.notna(surname) and surname not in {"", "unknown"}:
            union(group_key, f"family:{surname}")

    return passenger.map(lambda group: find(f"group:{group}"))


def validation_groups(frame: pd.DataFrame, group_scope: GroupScope) -> pd.Series:
    if group_scope == "passenger":
        return passenger_groups(frame)
    if group_scope == "household_component":
        return household_components(frame)
    raise ValueError("group_scope must be one of: passenger, household_component.")


def validate_recipe_scope(recipe: RecipeName, group_scope: GroupScope) -> None:
    is_cohort_recipe = recipe == "spaceship_cohort_v2"
    has_safe_scope = group_scope == "household_component"
    invalid_cohort_scope = is_cohort_recipe and not has_safe_scope
    if invalid_cohort_scope:
        raise ValueError(
            "spaceship_cohort_v2 requires group_scope='household_component' so "
            "batch-level family features remain fold-disjoint."
        )


def validation_estimand(recipe: RecipeName, group_scope: GroupScope) -> str:
    if group_scope == "household_component":
        return "generalization to held-out surname-linked household components"
    if recipe == "spaceship_competition_v3":
        return (
            "competition-aligned batch prediction with target-free relational "
            "features and surname categories allowed to recur across folds"
        )
    return (
        "passenger-group-disjoint prediction; surname categories may recur across "
        "folds when present in the selected recipe"
    )


def relational_overlap_audit(
    modeling: pd.DataFrame, evaluation: pd.DataFrame
) -> list[dict]:
    """Describe observable identifier recurrence between fitted and scored batches."""
    key_series = {
        "PassengerGroup": (passenger_groups(modeling), passenger_groups(evaluation)),
        "FamilyName": (family_names(modeling), family_names(evaluation)),
        "Cabin": (
            modeling["Cabin"].astype("string").str.lower().fillna("unknown"),
            evaluation["Cabin"].astype("string").str.lower().fillna("unknown"),
        ),
    }
    rows = []
    for feature, (left, right) in key_series.items():
        ignored = {"", "unknown"}
        shared = (set(left) - ignored) & (set(right) - ignored)
        evaluation_shared = right.isin(shared)
        rows.append(
            {
                "feature": feature,
                "shared_values": len(shared),
                "evaluation_rows_with_seen_value": int(evaluation_shared.sum()),
                "evaluation_fraction_with_seen_value": float(evaluation_shared.mean()),
            }
        )
    return rows


def split_spaceship_validation(
    train: pd.DataFrame,
    *,
    validation_protocol: ValidationProtocol,
    holdout_size: float,
    random_state: int,
    run: int = 0,
    group_scope: GroupScope = "passenger",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series | None, pd.Series | None]:
    y = train["Transported"].astype(bool).reset_index(drop=True)
    groups = validation_groups(train, group_scope).reset_index(drop=True)
    if validation_protocol == "grouped":
        n_splits = max(2, int(round(1 / holdout_size)))
        n_splits = min(n_splits, int(groups.nunique()))
        if n_splits < 2:
            raise ValueError("Grouped validation requires at least two groups.")
        splitter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state + (run // n_splits),
        )
        folds = list(splitter.split(train, y, groups))
        modeling_idx, holdout_idx = folds[run % len(folds)]
        return (
            train.iloc[modeling_idx],
            train.iloc[holdout_idx],
            groups.iloc[modeling_idx].reset_index(drop=True),
            groups.iloc[holdout_idx].reset_index(drop=True),
        )
    if validation_protocol == "row":
        modeling_raw, holdout_raw = train_test_split(
            train,
            test_size=holdout_size,
            stratify=y,
            random_state=random_state + run,
        )
        return modeling_raw, holdout_raw, None, None
    raise ValueError("validation_protocol must be one of: grouped, row.")


def reserve_confirmation_partition(
    train: pd.DataFrame,
    *,
    validation_protocol: ValidationProtocol,
    group_scope: GroupScope,
    confirmation_size: float,
    confirmation_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series | None, pd.Series | None]:
    return split_spaceship_validation(
        train,
        validation_protocol=validation_protocol,
        holdout_size=confirmation_size,
        random_state=confirmation_seed,
        group_scope=group_scope,
    )


def run_spaceship_benchmark(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    config: BenchmarkConfig,
) -> tuple[pd.DataFrame, dict, list[dict]]:
    rows = []
    prediction_rows = []
    diagnostics = []
    validate_recipe_scope(config.recipe, config.group_scope)
    benchmark_start = time.perf_counter()

    for run in range(config.runs):
        run_random_state = config.random_state + run
        modeling_raw, holdout_raw, groups_modeling, groups_holdout = (
            split_spaceship_validation(
                train,
                validation_protocol=config.validation_protocol,
                holdout_size=config.holdout_size,
                random_state=config.random_state,
                run=run,
                group_scope=config.group_scope,
            )
        )
        X_modeling, y_modeling, X_holdout, y_holdout = prepare_spaceship_modeling_split(
            modeling_raw,
            holdout_raw,
            recipe=config.recipe,
        )
        start_time = time.perf_counter()
        model = _make_mamut(config, random_state=run_random_state, final_refit=True)
        model.fit(
            X_modeling,
            y_modeling,
            X_holdout=X_holdout,
            y_holdout=y_holdout,
            groups=groups_modeling,
            groups_holdout=groups_holdout,
        )
        baseline_comparison, leakage_checks, group_overlap = benchmark_integrity_audit(
            model,
            X_modeling=X_modeling,
            y_modeling=y_modeling,
            X_evaluation=X_holdout,
            y_evaluation=y_holdout,
            groups_modeling=groups_modeling,
            groups_evaluation=groups_holdout,
        )
        duration_seconds = time.perf_counter() - start_time

        selected_model = model.selected_estimator_.__class__.__name__
        audit_candidates = model.holdout_summary_.sort_values(
            "accuracy_score", ascending=False
        )
        best_audit_candidate = audit_candidates.iloc[0]
        best_baseline = _best_baseline_row(baseline_comparison)
        leakage_warnings = int(
            leakage_checks["severity"].isin(["warning", "critical"]).sum()
        )

        selected_score = float(model.holdout_score_)
        best_audit_score = float(best_audit_candidate["accuracy_score"])
        baseline_score = _safe_float(best_baseline.get("score"))
        baseline_available = pd.notna(baseline_score)
        exceeds_margin = (
            baseline_score - selected_score > config.selection_practical_margin
        )
        baseline_challenge = baseline_available and exceeds_margin
        selected_predictions = np.asarray(model.predict(X_holdout)).astype(bool)
        holdout_truth = y_holdout.to_numpy(dtype=bool)
        repeat = run // max(2, int(round(1 / config.holdout_size)))
        for group, truth, prediction in zip(
            groups_holdout if groups_holdout is not None else range(len(X_holdout)),
            holdout_truth,
            selected_predictions,
        ):
            prediction_rows.append(
                {
                    "repeat": repeat,
                    "group": str(group),
                    "correct": bool(truth == prediction),
                }
            )
        rows.append(
            {
                "competition": config.competition,
                "recipe": config.recipe,
                "validation_protocol": config.validation_protocol,
                "group_scope": config.group_scope,
                "stage": config.stage,
                "run": run,
                "random_state": run_random_state,
                "train_rows": len(X_modeling),
                "holdout_rows": len(X_holdout),
                "features": X_modeling.shape[1],
                "selected_model": selected_model,
                "selected_holdout_score": selected_score,
                "best_audit_candidate_model": best_audit_candidate["model"],
                "best_audit_candidate_holdout_score": best_audit_score,
                "audit_candidate_delta": best_audit_score - selected_score,
                "best_baseline_model": best_baseline.get("model", ""),
                "best_baseline_score": baseline_score,
                "baseline_uplift": (
                    selected_score - baseline_score
                    if pd.notna(baseline_score)
                    else np.nan
                ),
                "guidance_status": (
                    "challenged_by_fixed_baseline"
                    if baseline_challenge
                    else "confirmed_against_fixed_baselines"
                ),
                "guidance_recommended_model": selected_model,
                "leakage_warnings": leakage_warnings,
                "evaluation_group_overlap": group_overlap,
                "duration_seconds": duration_seconds,
            }
        )
        diagnostics.append(
            {
                "run": run,
                "relational_overlap": relational_overlap_audit(
                    modeling_raw, holdout_raw
                ),
                "selection_summary": model.selection_summary_.to_dict(orient="records"),
                "candidate_holdout_audit_only": audit_candidates.to_dict(
                    orient="records"
                ),
                "fixed_baseline_comparison": baseline_comparison.to_dict(
                    orient="records"
                ),
                "leakage_checks": leakage_checks.to_dict(orient="records"),
            }
        )
        elapsed_seconds = time.perf_counter() - benchmark_start
        print(
            (
                f"[benchmark] completed development run {run + 1}/{config.runs}: "
                f"selected={selected_model} score={selected_score:.4f} "
                f"elapsed={elapsed_seconds:.1f}s"
            ),
            file=sys.stderr,
            flush=True,
        )
        budget_configured = config.max_runtime_seconds is not None
        budget_seconds = float(config.max_runtime_seconds or 0)
        budget_reached = budget_configured and elapsed_seconds >= budget_seconds
        runs_remaining = run + 1 < config.runs
        if budget_reached and runs_remaining:
            print(
                (
                    "[benchmark] stopping between completed runs because the soft "
                    f"runtime budget of {config.max_runtime_seconds:.1f}s was reached."
                ),
                file=sys.stderr,
                flush=True,
            )
            break

    run_results = pd.DataFrame(rows)
    aggregate = summarize_runs(
        run_results,
        score_column="selected_holdout_score",
        predictions=pd.DataFrame(prediction_rows),
        random_state=config.random_state,
    )
    aggregate.update(
        {
            "requested_runs": config.runs,
            "completed_runs": len(run_results),
            "runtime_budget_exhausted": len(run_results) < config.runs,
        }
    )
    return run_results, aggregate, diagnostics


def summarize_runs(
    run_results: pd.DataFrame,
    *,
    score_column: str,
    predictions: pd.DataFrame | None = None,
    random_state: int = 42,
) -> dict:
    scores = run_results[score_column].dropna().astype(float)
    audit_deltas = run_results["audit_candidate_delta"].dropna().astype(float)
    baseline_uplifts = run_results["baseline_uplift"].dropna().astype(float)
    stability_low = _safe_float(scores.min())
    stability_high = _safe_float(scores.max())
    bootstrap_low, bootstrap_high = group_bootstrap_accuracy_interval(
        predictions, random_state=random_state
    )

    return {
        "runs": int(len(run_results)),
        "mean_score": _safe_float(scores.mean()),
        "std_score": _safe_float(scores.std(ddof=1)) if len(scores) > 1 else 0.0,
        "stability_low": stability_low,
        "stability_high": stability_high,
        "group_bootstrap_low": bootstrap_low,
        "group_bootstrap_high": bootstrap_high,
        "interval_method": "group bootstrap of recorded outer predictions",
        "best_run_score": _safe_float(scores.max()),
        "worst_run_score": _safe_float(scores.min()),
        "mean_audit_candidate_delta": _safe_float(audit_deltas.mean()),
        "median_audit_candidate_delta": _safe_float(audit_deltas.median()),
        "p90_audit_candidate_delta": _safe_float(audit_deltas.quantile(0.90)),
        "mean_baseline_uplift": _safe_float(baseline_uplifts.mean()),
        "challenged_rate": _safe_float(
            run_results["guidance_status"].str.contains("challenged").mean()
        ),
        "total_duration_seconds": _safe_float(run_results["duration_seconds"].sum()),
    }


def group_bootstrap_accuracy_interval(
    predictions: pd.DataFrame | None,
    *,
    random_state: int,
    n_resamples: int = 2000,
    confidence_level: float = 0.95,
) -> tuple[float, float]:
    if predictions is None or predictions.empty:
        return np.nan, np.nan
    rng = np.random.default_rng(random_state)
    grouped_repeats = []
    for _, frame in predictions.groupby("repeat", sort=True):
        grouped = (
            frame.groupby("group", sort=False)["correct"]
            .agg(["sum", "count"])
            .to_numpy(dtype=float)
        )
        grouped_repeats.append(grouped)
    samples = np.empty(n_resamples, dtype=float)
    for _ in range(n_resamples):
        correct = 0.0
        count = 0.0
        for grouped in grouped_repeats:
            selected = rng.integers(0, len(grouped), size=len(grouped))
            correct += grouped[selected, 0].sum()
            count += grouped[selected, 1].sum()
        samples[_] = correct / count
    alpha = (1 - confidence_level) / 2
    return tuple(float(value) for value in np.quantile(samples, [alpha, 1 - alpha]))


def benchmark_integrity_audit(
    model: Mamut,
    *,
    X_modeling: pd.DataFrame,
    y_modeling: pd.Series,
    X_evaluation: pd.DataFrame,
    y_evaluation: pd.Series,
    groups_modeling: pd.Series | None,
    groups_evaluation: pd.Series | None,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    baseline_comparison = evaluate_estimators_on_split(
        estimators=default_baseline_estimators(random_state=model.random_state),
        X_train=X_modeling,
        y_train=y_modeling,
        X_evaluation=X_evaluation,
        y_evaluation=y_evaluation,
        metric_name=model.score_metric_name,
        binary=model.binary,
        preprocessor_factory=model._make_model_preprocessor,
    )
    leakage_checks = detect_leakage_risks(X_modeling, y_modeling)
    if groups_modeling is None or groups_evaluation is None:
        group_overlap = np.nan
    else:
        group_overlap = float(
            len(set(groups_modeling).intersection(set(groups_evaluation)))
        )
    return baseline_comparison, leakage_checks, group_overlap


def run_confirmation_benchmark(
    development: pd.DataFrame,
    confirmation: pd.DataFrame,
    development_groups: pd.Series | None,
    confirmation_groups: pd.Series | None,
    *,
    config: BenchmarkConfig,
) -> tuple[pd.DataFrame, dict, list[dict], Mamut]:
    validate_recipe_scope(config.recipe, config.group_scope)
    X_development, y_development, X_confirmation, y_confirmation = (
        prepare_spaceship_modeling_split(
            development,
            confirmation,
            recipe=config.recipe,
        )
    )
    start_time = time.perf_counter()
    model = _make_mamut(config, random_state=config.random_state, final_refit=True)
    model.fit(
        X_development,
        y_development,
        groups=development_groups,
    )
    baseline_comparison, leakage_checks, group_overlap = benchmark_integrity_audit(
        model,
        X_modeling=X_development,
        y_modeling=y_development,
        X_evaluation=X_confirmation,
        y_evaluation=y_confirmation,
        groups_modeling=development_groups,
        groups_evaluation=confirmation_groups,
    )
    duration_seconds = time.perf_counter() - start_time
    selected_model = model.selected_estimator_.__class__.__name__
    best_baseline = _best_baseline_row(baseline_comparison)
    predictions = np.asarray(model.predict(X_confirmation)).astype(bool)
    selected_score = float(model.score_metric(y_confirmation, predictions))
    baseline_score = _safe_float(best_baseline.get("score"))
    truth = y_confirmation.to_numpy(dtype=bool)
    recorded_predictions = pd.DataFrame(
        {
            "repeat": 0,
            "group": [
                str(value)
                for value in (
                    confirmation_groups
                    if confirmation_groups is not None
                    else range(len(X_confirmation))
                )
            ],
            "correct": truth == predictions,
        }
    )
    run_results = pd.DataFrame(
        [
            {
                "competition": config.competition,
                "recipe": config.recipe,
                "validation_protocol": config.validation_protocol,
                "group_scope": config.group_scope,
                "stage": "confirmation",
                "run": 0,
                "random_state": config.random_state,
                "train_rows": len(X_development),
                "holdout_rows": len(X_confirmation),
                "features": X_development.shape[1],
                "selected_model": selected_model,
                "selected_holdout_score": selected_score,
                "best_audit_candidate_model": "not evaluated for confirmation",
                "best_audit_candidate_holdout_score": np.nan,
                "audit_candidate_delta": np.nan,
                "best_baseline_model": best_baseline.get("model", ""),
                "best_baseline_score": baseline_score,
                "baseline_uplift": (
                    selected_score - baseline_score
                    if pd.notna(baseline_score)
                    else np.nan
                ),
                "guidance_status": "confirmation_observation_only",
                "guidance_recommended_model": selected_model,
                "leakage_warnings": int(
                    leakage_checks["severity"].isin(["warning", "critical"]).sum()
                ),
                "evaluation_group_overlap": group_overlap,
                "duration_seconds": duration_seconds,
            }
        ]
    )
    aggregate = summarize_runs(
        run_results,
        score_column="selected_holdout_score",
        predictions=recorded_predictions,
        random_state=config.random_state,
    )
    aggregate["confirmation_observation_only"] = True
    diagnostics = [
        {
            "run": 0,
            "relational_overlap": relational_overlap_audit(development, confirmation),
            "selection_summary": model.selection_summary_.to_dict(orient="records"),
            "fixed_baseline_comparison": baseline_comparison.to_dict(orient="records"),
            "leakage_checks": leakage_checks.to_dict(orient="records"),
            "candidate_holdout_audit_only": "not evaluated during confirmation",
        }
    ]
    print(
        (
            "[benchmark] completed locked confirmation: "
            f"selected={selected_model} score={selected_score:.4f} "
            f"elapsed={duration_seconds:.1f}s"
        ),
        file=sys.stderr,
        flush=True,
    )
    return run_results, aggregate, diagnostics, model


def fit_submission_model(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    config: BenchmarkConfig,
    frozen_model: Mamut | None = None,
) -> tuple[pd.DataFrame, Mamut]:
    X, y, X_test, passenger_ids = prepare_spaceship_features(
        train, test, recipe=config.recipe
    )
    if frozen_model is not None:
        selected_model = frozen_model.selected_estimator_.__class__.__name__
        estimator = clone(frozen_model.selected_estimator_)
        preprocessor = frozen_model._make_model_preprocessor(selected_model)
        if preprocessor is not None:
            X_fitted, y_fitted = preprocessor.fit_transform(X.copy(), y.copy())
            X_predict = preprocessor.transform(X_test.copy())
        else:
            X_fitted, y_fitted = X, y
            X_predict = X_test
        estimator.fit(X_fitted, y_fitted)
        predictions = _coerce_bool_predictions(estimator.predict(X_predict))
        submission = pd.DataFrame(
            {"PassengerId": passenger_ids, "Transported": predictions}
        )
        return submission, frozen_model

    model = _make_mamut(config, random_state=config.random_state, final_refit=True)
    groups = (
        validation_groups(train, config.group_scope)
        if config.validation_protocol == "grouped"
        else None
    )
    model.fit(X, y, groups=groups)
    predictions = _coerce_bool_predictions(model.predict(X_test))
    submission = pd.DataFrame(
        {"PassengerId": passenger_ids, "Transported": predictions}
    )
    return submission, model


def selected_hyperparameters(model: Mamut) -> dict:
    selected_model = model.selected_estimator_.__class__.__name__
    study = model.optuna_studies_.get(selected_model)
    return dict(study.best_params) if study is not None else {}


def write_submission(submission: pd.DataFrame, path: Path) -> Path:
    if list(submission.columns) != ["PassengerId", "Transported"]:
        raise ValueError("Submission must contain columns: PassengerId, Transported.")
    path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(path, index=False)
    return path


def submit_to_kaggle(
    competition: CompetitionName, submission_path: Path, message: str
) -> None:
    subprocess.run(
        [
            "kaggle",
            "competitions",
            "submit",
            "-c",
            competition,
            "-f",
            str(submission_path),
            "-m",
            message,
        ],
        check=True,
    )


def build_metadata(data_dir: Path) -> BenchmarkMetadata:
    return BenchmarkMetadata(
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        mamut_version=mamut.__version__,
        git_commit=_git_commit(),
        train_sha256=_sha256(data_dir / "train.csv"),
        test_sha256=_sha256(data_dir / "test.csv"),
        sample_submission_sha256=_sha256(data_dir / "sample_submission.csv"),
        git_branch=_git_branch(),
        git_dirty=_git_dirty(),
        python_version=platform.python_version(),
        feature_recipe_version=FEATURE_RECIPE_VERSION,
    )


def format_results(payload: dict, output_format: OutputFormat) -> str:
    run_results = pd.DataFrame(payload["runs"])
    aggregate = pd.DataFrame([payload["aggregate"]])
    if output_format == "json":
        return json.dumps(_json_ready(payload), indent=2, allow_nan=False)
    if output_format == "csv":
        return run_results.to_csv(index=False, float_format="%.6f")
    if output_format == "markdown":
        return "\n".join(
            [
                "## Aggregate",
                _to_markdown(_display_scores(aggregate)),
                "",
                "## Runs",
                _to_markdown(_display_scores(run_results)),
            ]
        )
    raise ValueError("Unsupported output format.")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run external Kaggle benchmarks for MAMUT."
    )
    parser.add_argument("competition", choices=["spaceship-titanic"])
    parser.add_argument(
        "--recipe",
        choices=[
            "raw",
            "spaceship_inductive",
            "spaceship_cohort",
            "spaceship_basic",
            "spaceship_inductive_v2",
            "spaceship_cohort_v2",
            "spaceship_competition_v3",
        ],
        default="spaceship_competition_v3",
    )
    parser.add_argument(
        "--stage",
        choices=["diagnostic", "development", "confirmation"],
        default="development",
        help="Reserve a locked confirmation partition unless running legacy diagnostics.",
    )
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--n-iterations", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--holdout-size", type=float, default=0.2)
    parser.add_argument(
        "--validation-protocol", choices=["grouped", "row"], default="grouped"
    )
    parser.add_argument(
        "--group-scope",
        choices=["passenger", "household_component"],
        default="passenger",
    )
    parser.add_argument("--confirmation-size", type=float, default=0.2)
    parser.add_argument(
        "--confirmation-seed", type=int, default=DEFAULT_CONFIRMATION_SEED
    )
    parser.add_argument(
        "--campaign-id",
        default="exploration",
        help="Stable identifier used to isolate development and confirmation records.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional immutable run directory name; generated automatically by default.",
    )
    parser.add_argument(
        "--max-runtime-seconds",
        type=float,
        default=None,
        help="Soft development budget checked between completed outer runs.",
    )
    parser.add_argument(
        "--score-metric",
        choices=["accuracy", "balanced_accuracy", "f1"],
        default="accuracy",
    )
    parser.add_argument(
        "--search-profile",
        choices=["quick", "balanced", "thorough"],
        default="quick",
    )
    parser.add_argument(
        "--selection-strategy",
        choices=["single_split", "nested_cv", "repeated_cv"],
        default="nested_cv",
    )
    parser.add_argument("--selection-cv-splits", type=int, default=5)
    parser.add_argument("--selection-cv-repeats", type=int, default=2)
    parser.add_argument("--selection-practical-margin", type=float, default=0.005)
    parser.add_argument(
        "--preprocessing-profile",
        choices=["auto", "generic_ohe"],
        default="auto",
    )
    parser.add_argument(
        "--optimization-method",
        choices=["random_search", "bayes"],
        default="random_search",
    )
    parser.add_argument(
        "--exclude-models", nargs="*", default=list(DEFAULT_EXCLUDED_MODELS)
    )
    parser.add_argument("--include-models", nargs="*", default=None)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--format", choices=["markdown", "json", "csv"], default="markdown"
    )
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--write-submission", action="store_true")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--submission-message", default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.runs < 1:
        raise ValueError("--runs must be at least 1.")
    if args.n_iterations < 1:
        raise ValueError("--n-iterations must be at least 1.")
    if args.selection_cv_splits < 2:
        raise ValueError("--selection-cv-splits must be at least 2.")
    if args.selection_cv_repeats < 1:
        raise ValueError("--selection-cv-repeats must be at least 1.")
    if args.selection_practical_margin < 0:
        raise ValueError("--selection-practical-margin must be non-negative.")
    if args.submit and not args.write_submission:
        raise ValueError("--submit requires --write-submission.")
    if args.write_submission and args.stage != "confirmation":
        raise ValueError("--write-submission requires --stage confirmation.")
    if args.include_models and args.exclude_models:
        raise ValueError("Use --include-models or --exclude-models, not both.")
    if not 0 < args.confirmation_size < 1:
        raise ValueError("--confirmation-size must be greater than 0 and less than 1.")
    if args.max_runtime_seconds is not None and args.max_runtime_seconds <= 0:
        raise ValueError("--max-runtime-seconds must be greater than zero.")
    validate_identifier(args.campaign_id, name="campaign-id")
    if args.run_id is not None:
        validate_identifier(args.run_id, name="run-id")
    validate_recipe_scope(args.recipe, args.group_scope)

    config = BenchmarkConfig(
        competition=args.competition,
        recipe=args.recipe,
        runs=args.runs,
        n_iterations=args.n_iterations,
        random_state=args.random_state,
        holdout_size=args.holdout_size,
        validation_protocol=args.validation_protocol,
        score_metric=args.score_metric,
        search_profile=args.search_profile,
        selection_strategy=args.selection_strategy,
        selection_cv_splits=args.selection_cv_splits,
        selection_cv_repeats=args.selection_cv_repeats,
        selection_practical_margin=args.selection_practical_margin,
        preprocessing_profile=args.preprocessing_profile,
        optimization_method=args.optimization_method,
        excluded_models=tuple(args.exclude_models),
        included_models=tuple(args.include_models) if args.include_models else None,
        group_scope=args.group_scope,
        stage=args.stage,
        confirmation_size=args.confirmation_size,
        confirmation_seed=args.confirmation_seed,
        campaign_id=args.campaign_id,
        max_runtime_seconds=args.max_runtime_seconds,
    )

    data_dir = ensure_competition_data(
        config.competition,
        args.cache_dir,
        force_download=args.force_download,
    )
    train, test, _ = load_spaceship_titanic(data_dir)
    metadata = build_metadata(data_dir)
    if args.submit and metadata.git_dirty:
        raise RuntimeError(
            "Official Kaggle submission requires a clean git working tree so the "
            "uploaded artifact is reproducible from a recorded commit."
        )
    campaign_dir = args.output_dir / config.competition / config.campaign_id
    run_id = args.run_id or build_run_id(metadata, config.stage)
    output_dir = campaign_dir / config.recipe / run_id
    if output_dir.exists():
        raise RuntimeError(
            f"Run directory already exists: {output_dir}. Use a new --run-id."
        )
    output_dir.mkdir(parents=True)
    confirmation_marker = campaign_dir / "confirmation_consumed.json"
    if args.stage == "confirmation" and confirmation_marker.exists():
        raise RuntimeError(
            "The reserved confirmation partition has already been evaluated for this "
            "campaign. Start a separately documented campaign instead of reusing it."
        )

    partitions = {
        "official_train_rows": len(train),
        "official_test_rows": len(test),
        "confirmation_reserved": args.stage != "diagnostic",
        "official_train_test_relational_overlap": relational_overlap_audit(train, test),
    }
    confirmation_model = None
    if args.stage == "diagnostic":
        run_results, aggregate, diagnostics = run_spaceship_benchmark(
            train, test, config=config
        )
    else:
        development, confirmation, development_groups, confirmation_groups = (
            reserve_confirmation_partition(
                train,
                validation_protocol=config.validation_protocol,
                group_scope=config.group_scope,
                confirmation_size=config.confirmation_size,
                confirmation_seed=config.confirmation_seed,
            )
        )
        partitions.update(
            {
                "development_rows": len(development),
                "confirmation_rows": len(confirmation),
                "confirmation_seed": config.confirmation_seed,
                "confirmation_size": config.confirmation_size,
            }
        )
        if args.stage == "development":
            run_results, aggregate, diagnostics = run_spaceship_benchmark(
                development, test, config=config
            )
        else:
            (
                run_results,
                aggregate,
                diagnostics,
                confirmation_model,
            ) = run_confirmation_benchmark(
                development,
                confirmation,
                development_groups,
                confirmation_groups,
                config=config,
            )

    payload = {
        "metadata": asdict(metadata),
        "config": asdict(config),
        "partitions": partitions,
        "protocol": {
            "primary_metric": config.score_metric,
            "grouped_validation": config.validation_protocol == "grouped",
            "validation_estimand": validation_estimand(
                config.recipe, config.group_scope
            ),
            "development_interval": (
                "Group bootstrap interval of recorded outer predictions; it "
                "does not represent post-selection confirmation."
            ),
            "confirmation_policy": (
                "Confirmation scores are observations of a frozen candidate and "
                "must not be used to promote an alternative model."
            ),
        },
        "aggregate": aggregate,
        "runs": run_results.to_dict(orient="records"),
        "diagnostics": diagnostics,
        "submission": {"generated": False, "uploaded": False},
        "campaign_id": config.campaign_id,
        "run_id": run_id,
    }
    results_path = output_dir / "results.json"
    manifest_path = output_dir / "manifest.json"

    write_campaign_artifacts(payload, results_path, manifest_path)
    if args.stage == "confirmation":
        confirmation_marker.write_text(
            json.dumps(
                {
                    "generated_at_utc": metadata.generated_at_utc,
                    "git_commit": metadata.git_commit,
                    "recipe": config.recipe,
                    "config": asdict(config),
                    "result_path": str(results_path),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    if args.write_submission:
        submission, submission_model = fit_submission_model(
            train,
            test,
            config=config,
            frozen_model=confirmation_model,
        )
        submission_path = output_dir / "submission.csv"
        write_submission(submission, submission_path)
        payload["submission"] = {
            "generated": True,
            "uploaded": False,
            "path": str(submission_path),
            "fit_policy": "fixed_confirmation_parameters_refit_on_full_training",
            "selected_hyperparameters": selected_hyperparameters(submission_model),
        }
        write_campaign_artifacts(payload, results_path, manifest_path)
        if args.submit:
            message = args.submission_message or _default_submission_message(
                config, metadata
            )
            submit_to_kaggle(config.competition, submission_path, message)
            payload["submission"] = {
                "generated": True,
                "uploaded": True,
                "path": str(submission_path),
                "message": message,
                "fit_policy": "fixed_confirmation_parameters_refit_on_full_training",
                "selected_hyperparameters": selected_hyperparameters(submission_model),
            }
            write_campaign_artifacts(payload, results_path, manifest_path)
    print(format_results(payload, args.format))
    return 0


def write_campaign_artifacts(
    payload: dict, results_path: Path, manifest_path: Path
) -> None:
    manifest_path.write_text(
        json.dumps(
            _json_ready(
                {
                    "metadata": payload["metadata"],
                    "config": payload["config"],
                    "partitions": payload["partitions"],
                    "protocol": payload["protocol"],
                    "submission": payload["submission"],
                    "campaign_id": payload["campaign_id"],
                    "run_id": payload["run_id"],
                }
            ),
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    payload["manifest_path"] = str(manifest_path)
    results_path.write_text(
        json.dumps(_json_ready(payload), indent=2, allow_nan=False),
        encoding="utf-8",
    )


def _make_mamut(
    config: BenchmarkConfig, *, random_state: int, final_refit: bool
) -> Mamut:
    return Mamut(
        score_metric=config.score_metric,
        search_profile=config.search_profile,
        selection_strategy=config.selection_strategy,
        selection_cv_splits=config.selection_cv_splits,
        selection_cv_repeats=config.selection_cv_repeats,
        selection_practical_margin=config.selection_practical_margin,
        preprocessing_profile=config.preprocessing_profile,
        optimization_method=config.optimization_method,
        n_iterations=config.n_iterations,
        random_state=random_state,
        exclude_models=list(config.excluded_models),
        include_models=(
            list(config.included_models) if config.included_models is not None else None
        ),
        refit_final_model=final_refit,
        num_imputation="mean",
        evidence_cv_splits=3,
        evidence_cv_repeats=1,
    )


def _metric_column(score_metric: str) -> str:
    return {
        "accuracy": "accuracy_score",
        "balanced_accuracy": "balanced_accuracy_score",
        "f1": "f1_score",
    }[score_metric]


def _best_baseline_row(baseline_comparison: pd.DataFrame) -> pd.Series:
    baselines = baseline_comparison.loc[
        ~baseline_comparison["model"].str.startswith("MAMUT ")
    ].dropna(subset=["score"])
    if baselines.empty:
        return pd.Series({"model": "", "score": np.nan})
    return baselines.sort_values("score", ascending=False).iloc[0]


def _safe_float(value) -> float:
    if pd.isna(value):
        return np.nan
    return float(value)


def _coerce_bool_predictions(predictions: Iterable) -> list[bool]:
    result = []
    for prediction in predictions:
        if isinstance(prediction, (bool, np.bool_)):
            result.append(bool(prediction))
        elif isinstance(prediction, str):
            normalized = prediction.strip().lower()
            if normalized not in {"true", "false"}:
                raise ValueError(f"Cannot coerce prediction {prediction!r} to bool.")
            result.append(normalized == "true")
        else:
            result.append(bool(prediction))
    return result


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _git_branch() -> str:
    try:
        return subprocess.check_output(
            ["git", "branch", "--show-current"], text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _git_dirty() -> bool:
    try:
        return bool(
            subprocess.check_output(
                ["git", "status", "--porcelain", "--untracked-files=all"],
                text=True,
            ).strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return True


def validate_identifier(value: str, *, name: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value):
        raise ValueError(
            f"--{name} must start with an alphanumeric character and contain "
            "only letters, numbers, '.', '_' or '-'."
        )


def build_run_id(metadata: BenchmarkMetadata, stage: BenchmarkStage) -> str:
    timestamp = datetime.fromisoformat(metadata.generated_at_utc).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )
    return f"{stage}-{timestamp}-{metadata.git_commit}"


def _default_submission_message(
    config: BenchmarkConfig, metadata: BenchmarkMetadata
) -> str:
    return (
        f"MAMUT {metadata.mamut_version} {config.recipe} "
        f"runs={config.runs} n_iter={config.n_iterations} commit={metadata.git_commit}"
    )


def _json_ready(value):
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _display_scores(frame: pd.DataFrame) -> pd.DataFrame:
    display = frame.copy()
    score_like = [
        column
        for column in display.columns
        if any(
            token in column
            for token in (
                "score",
                "regret",
                "uplift",
                "rate",
                "ci_",
                "stability",
                "bootstrap",
            )
        )
    ]
    for column in score_like:
        display[column] = display[column].map(_format_score)
    if "duration_seconds" in display.columns:
        display["duration_seconds"] = display["duration_seconds"].map(
            lambda value: f"{float(value):.1f}"
        )
    if "total_duration_seconds" in display.columns:
        display["total_duration_seconds"] = display["total_duration_seconds"].map(
            lambda value: f"{float(value):.1f}"
        )
    return display


def _format_score(value) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{float(value):.4f}"


def _to_markdown(table: pd.DataFrame) -> str:
    headers = list(table.columns)
    rows = [[str(value) for value in row] for row in table.to_numpy()]
    widths = [
        max(len(str(header)), max((len(row[index]) for row in rows), default=0))
        for index, header in enumerate(headers)
    ]
    header_line = _format_markdown_row([str(header) for header in headers], widths)
    separator_line = _format_markdown_row(["-" * width for width in widths], widths)
    row_lines = [_format_markdown_row(row, widths) for row in rows]
    return "\n".join([header_line, separator_line, *row_lines])


def _format_markdown_row(values: list[str], widths: list[int]) -> str:
    padded_values = [value.ljust(widths[index]) for index, value in enumerate(values)]
    return f"| {' | '.join(padded_values)} |"


if __name__ == "__main__":
    raise SystemExit(main())
