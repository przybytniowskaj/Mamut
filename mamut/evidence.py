from typing import Callable, Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    jaccard_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedGroupKFold

from mamut.model_selection import fit_estimator, make_preprocessor


def default_baseline_estimators(random_state: Optional[int] = 42) -> dict:
    return {
        "Dummy Most Frequent": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=random_state,
            solver="liblinear",
        ),
        "Random Forest": RandomForestClassifier(
            class_weight="balanced_subsample",
            n_estimators=200,
            n_jobs=1,
            random_state=random_state,
        ),
    }


def detect_leakage_risks(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    y = pd.Series(y).reset_index(drop=True)
    X = pd.DataFrame(X).reset_index(drop=True)
    issues = []

    if y.isna().any():
        issues.append(
            {
                "severity": "critical",
                "check": "target_missing_values",
                "message": "Target contains missing values.",
            }
        )

    n_classes = y.nunique(dropna=True)
    if n_classes < 2:
        issues.append(
            {
                "severity": "critical",
                "check": "target_class_count",
                "message": "Target has fewer than two observed classes.",
            }
        )

    class_share = y.value_counts(normalize=True, dropna=True)
    if not class_share.empty and class_share.min() < 0.05:
        issues.append(
            {
                "severity": "warning",
                "check": "class_imbalance",
                "message": f"Minority class share is {class_share.min():.3f}.",
            }
        )

    if y.name is not None and y.name in X.columns:
        issues.append(
            {
                "severity": "critical",
                "check": "target_column_name_present",
                "message": f"Feature matrix contains a column named like the target: {y.name}.",
            }
        )

    target_terms = ("target", "label", "class", "outcome", "response")
    for column in X.columns:
        column_name = str(column).lower()
        if any(term in column_name for term in target_terms):
            issues.append(
                {
                    "severity": "warning",
                    "check": "target_like_feature_name",
                    "message": f"Feature '{column}' has a target-like name.",
                }
            )

        feature = X[column].reset_index(drop=True)
        if feature.equals(y):
            issues.append(
                {
                    "severity": "critical",
                    "check": "feature_equals_target",
                    "message": f"Feature '{column}' exactly matches the target.",
                }
            )

        unique_ratio = feature.nunique(dropna=True) / max(len(feature), 1)
        id_like_name = any(term in column_name for term in ("id", "uuid", "key"))
        if unique_ratio >= 0.95 and id_like_name:
            issues.append(
                {
                    "severity": "warning",
                    "check": "id_like_high_cardinality_feature",
                    "message": f"Feature '{column}' looks like an identifier and is nearly unique.",
                }
            )

        if 1 < feature.nunique(dropna=True) <= max(20, len(feature) * 0.2):
            grouped_target_counts = y.groupby(feature, dropna=False).nunique(
                dropna=True
            )
            if not grouped_target_counts.empty and grouped_target_counts.max() == 1:
                issues.append(
                    {
                        "severity": "warning",
                        "check": "single_feature_perfect_mapping",
                        "message": f"Feature '{column}' maps deterministically to the target in this dataset.",
                    }
                )

    duplicated_features = X.duplicated(keep=False)
    if duplicated_features.any():
        duplicate_frame = X.loc[duplicated_features].copy()
        duplicate_frame["_target"] = y.loc[duplicated_features].to_numpy()
        conflicting_groups = duplicate_frame.groupby(list(X.columns), dropna=False)[
            "_target"
        ].nunique()
        n_conflicting = int((conflicting_groups > 1).sum())
        if n_conflicting:
            issues.append(
                {
                    "severity": "info",
                    "check": "duplicate_features_conflicting_targets",
                    "message": (
                        f"{n_conflicting} duplicated feature pattern(s) have conflicting "
                        "targets; this is outcome ambiguity, not evidence of leakage."
                    ),
                }
            )
        else:
            issues.append(
                {
                    "severity": "info",
                    "check": "duplicate_features",
                    "message": f"{int(duplicated_features.sum())} rows duplicate another feature row.",
                }
            )

    if not issues:
        issues.append(
            {
                "severity": "pass",
                "check": "basic_leakage_screen",
                "message": "No high-signal leakage risks were detected by the basic checks.",
            }
        )

    return pd.DataFrame(issues)


def build_evidence_report(
    X: pd.DataFrame,
    y: pd.Series,
    y_leakage: Optional[pd.Series],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_evaluation: pd.DataFrame,
    y_evaluation: pd.Series,
    selected_estimator,
    metric_name: str,
    binary: bool,
    preprocessor_factory: Callable,
    evaluation_dataset: str,
    holdout_available: bool,
    groups: Optional[pd.Series] = None,
    groups_train: Optional[pd.Series] = None,
    groups_evaluation: Optional[pd.Series] = None,
    cv_splits: int = 5,
    cv_repeats: int = 3,
    confidence_level: float = 0.95,
    random_state: Optional[int] = 42,
    practical_margin: float = 0.01,
    candidate_estimators: Optional[dict] = None,
) -> dict:
    selected_model_label = f"MAMUT Selected ({selected_estimator.__class__.__name__})"
    candidate_estimators = candidate_estimators or {}
    candidate_comparison_estimators = {
        f"MAMUT Candidate ({model_name})": estimator
        for model_name, estimator in candidate_estimators.items()
        if estimator.__class__.__name__ != selected_estimator.__class__.__name__
    }
    estimators = {
        selected_model_label: selected_estimator,
        **candidate_comparison_estimators,
        **default_baseline_estimators(random_state=random_state),
    }

    baseline_comparison = evaluate_estimators_on_split(
        estimators=estimators,
        X_train=X_train,
        y_train=y_train,
        X_evaluation=X_evaluation,
        y_evaluation=y_evaluation,
        metric_name=metric_name,
        binary=binary,
        preprocessor_factory=preprocessor_factory,
    )
    score_stability = repeated_stratified_cv_scores(
        estimators=estimators,
        X=X,
        y=y,
        metric_name=metric_name,
        binary=binary,
        preprocessor_factory=preprocessor_factory,
        cv_splits=cv_splits,
        cv_repeats=cv_repeats,
        confidence_level=confidence_level,
        random_state=random_state,
        groups=groups,
    )
    leakage_checks = detect_leakage_risks(X, y if y_leakage is None else y_leakage)
    group_overlap = np.nan
    if groups_train is not None and groups_evaluation is not None:
        group_overlap = len(set(groups_train).intersection(groups_evaluation))
    validation_integrity = pd.DataFrame(
        [
            {
                "evaluation_dataset": evaluation_dataset,
                "holdout_available": holdout_available,
                "cv_strategy": (
                    "RepeatedStratifiedGroupKFold"
                    if groups is not None
                    else "RepeatedStratifiedKFold"
                ),
                "cv_splits": score_stability.attrs.get("cv_splits", np.nan),
                "cv_repeats": cv_repeats,
                "confidence_level": confidence_level,
                "interval_interpretation": (
                    "Descriptive stability interval from dependent resampling scores; "
                    "not a confirmatory confidence interval."
                ),
                "grouped_validation": groups is not None,
                "n_groups": (
                    int(pd.Series(groups).nunique()) if groups is not None else np.nan
                ),
                "evaluation_group_overlap": group_overlap,
                "n_leakage_warnings": int(
                    leakage_checks["severity"].isin(["warning", "critical"]).sum()
                ),
            }
        ]
    )
    selection_guidance = build_selection_guidance(
        baseline_comparison=baseline_comparison,
        score_stability=score_stability,
        leakage_checks=leakage_checks,
        selected_model_label=selected_model_label,
        evaluation_dataset=evaluation_dataset,
        practical_margin=practical_margin,
    )

    return {
        "validation_integrity": validation_integrity,
        "leakage_checks": leakage_checks,
        "baseline_comparison": baseline_comparison,
        "score_stability": score_stability,
        "selection_guidance": selection_guidance,
    }


def build_selection_guidance(
    baseline_comparison: pd.DataFrame,
    score_stability: pd.DataFrame,
    leakage_checks: pd.DataFrame,
    selected_model_label: str,
    evaluation_dataset: str,
    practical_margin: float = 0.01,
) -> pd.DataFrame:
    critical_leakage = leakage_checks["severity"].eq("critical").any()
    split_row = _row_for_model(baseline_comparison, selected_model_label, "model")
    stability_row = _row_for_model(score_stability, selected_model_label, "model")
    best_split = _best_row(baseline_comparison, "score")
    best_stability = _best_row(score_stability, "mean_score")

    selected_split_score = _value_or_nan(split_row, "score")
    selected_stability_mean = _value_or_nan(stability_row, "mean_score")
    selected_stability_std = _value_or_nan(stability_row, "std_score")
    selected_ci_high = _value_or_nan(stability_row, "ci_high")

    best_split_score = _value_or_nan(best_split, "score")
    best_stability_mean = _value_or_nan(best_stability, "mean_score")
    best_stability_std = _value_or_nan(best_stability, "std_score")
    best_ci_low = _value_or_nan(best_stability, "ci_low")

    split_delta = best_split_score - selected_split_score
    stability_delta = best_stability_mean - selected_stability_mean
    split_challenger = _value_or_empty(best_split, "model")
    stability_challenger = _value_or_empty(best_stability, "model")

    split_has_challenger = split_challenger != selected_model_label
    split_delta_available = pd.notna(split_delta)
    split_challenge = all(
        [split_has_challenger, split_delta_available, split_delta > practical_margin]
    )

    stability_has_challenger = stability_challenger != selected_model_label
    stability_delta_available = pd.notna(stability_delta)
    stability_challenge = all(
        [
            stability_has_challenger,
            stability_delta_available,
            stability_delta > practical_margin,
        ]
    )

    ci_available = pd.notna(best_ci_low) and pd.notna(selected_ci_high)
    ci_separated = ci_available and best_ci_low > selected_ci_high

    stability_std_available = pd.notna(best_stability_std) and pd.notna(
        selected_stability_std
    )
    stability_similar_mean = all(
        [stability_delta_available, abs(stability_delta) <= practical_margin]
    )
    stability_lower_variance = all(
        [
            stability_std_available,
            best_stability_std + practical_margin < selected_stability_std,
        ]
    )
    stability_caution = all(
        [
            stability_has_challenger,
            not stability_challenge,
            stability_similar_mean,
            stability_lower_variance,
        ]
    )

    if critical_leakage:
        status = "blocked"
        recommended_model = selected_model_label
        review_candidate = selected_model_label
        action = "Fix critical leakage risks before trusting model-selection evidence."
        reason = "At least one critical leakage risk was detected."
    elif split_challenge:
        status = "challenged"
        review_candidate = split_challenger
        recommended_model = (
            selected_model_label
            if evaluation_dataset == "holdout"
            else split_challenger
        )
        action = _challenge_action(evaluation_dataset)
        reason = (
            f"{split_challenger} outperformed the selected model on the "
            f"{evaluation_dataset} split by {split_delta:.4f}."
        )
    elif stability_challenge:
        status = "challenged_strong" if ci_separated else "challenged"
        review_candidate = stability_challenger
        recommended_model = (
            selected_model_label
            if evaluation_dataset == "holdout"
            else stability_challenger
        )
        action = (
            "Prefer the challenger for review and rerun selection with repeated "
            "validation before changing the production candidate."
        )
        reason = (
            f"{stability_challenger} had a higher repeated-validation mean by "
            f"{stability_delta:.4f}."
        )
        if ci_separated:
            reason += " Its stability interval is separated above the selected model."
    elif stability_caution:
        status = "confirmed_with_caution"
        recommended_model = selected_model_label
        review_candidate = stability_challenger
        action = (
            "The selected model remains competitive, but review the lower-variance "
            "alternative before deployment."
        )
        reason = (
            f"{stability_challenger} had comparable mean performance with lower "
            "fold-to-fold variability."
        )
    elif pd.isna(selected_split_score) or pd.isna(selected_stability_mean):
        status = "inconclusive"
        recommended_model = selected_model_label
        review_candidate = selected_model_label
        action = "Evidence was incomplete; inspect failed baseline or stability rows."
        reason = "Selected model evidence contains missing scores."
    else:
        status = "confirmed"
        recommended_model = selected_model_label
        review_candidate = selected_model_label
        action = "Selected model is competitive with evidence baselines."
        reason = (
            "No evidence baseline exceeded the selected model by the practical margin."
        )

    return pd.DataFrame(
        [
            {
                "status": status,
                "selected_model": selected_model_label,
                "recommended_model": recommended_model,
                "review_candidate": review_candidate,
                "reason": reason,
                "action": action,
                "evaluation_dataset": evaluation_dataset,
                "practical_margin": practical_margin,
                "selected_split_score": selected_split_score,
                "best_split_model": split_challenger,
                "best_split_score": best_split_score,
                "split_delta": split_delta,
                "selected_stability_mean": selected_stability_mean,
                "selected_stability_std": selected_stability_std,
                "best_stability_model": stability_challenger,
                "best_stability_mean": best_stability_mean,
                "best_stability_std": best_stability_std,
                "stability_delta": stability_delta,
                "ci_separated": ci_separated,
            }
        ]
    )


def _challenge_action(evaluation_dataset: str) -> str:
    if evaluation_dataset == "holdout":
        return (
            "Do not silently promote the challenger from final holdout evidence. "
            "Rerun model selection with the challenger included or reserve a new "
            "final holdout before deployment."
        )
    return (
        "Treat the challenger as the current candidate and confirm it with a final "
        "holdout or repeated validation before deployment."
    )


def _row_for_model(table: pd.DataFrame, model_name: str, column: str):
    match = table.loc[table[column] == model_name]
    if match.empty:
        return None
    return match.iloc[0]


def _best_row(table: pd.DataFrame, score_column: str):
    valid_rows = table.dropna(subset=[score_column])
    if valid_rows.empty:
        return None
    return valid_rows.sort_values(by=score_column, ascending=False).iloc[0]


def _value_or_nan(row, column: str):
    if row is None:
        return np.nan
    return row[column]


def _value_or_empty(row, column: str) -> str:
    if row is None:
        return ""
    return str(row[column])


def evaluate_estimators_on_split(
    estimators: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_evaluation: pd.DataFrame,
    y_evaluation: pd.Series,
    metric_name: str,
    binary: bool,
    preprocessor_factory: Callable,
) -> pd.DataFrame:
    rows = []
    for model_name, estimator in estimators.items():
        try:
            fitted_estimator, X_eval_transformed = _fit_estimator_with_preprocessing(
                model_name=model_name,
                estimator=estimator,
                X_train=X_train,
                y_train=y_train,
                X_evaluation=X_evaluation,
                preprocessor_factory=preprocessor_factory,
            )
            score = score_estimator(
                fitted_estimator,
                X_eval_transformed,
                y_evaluation,
                metric_name=metric_name,
                binary=binary,
            )
            status = "ok"
        except Exception as exc:  # pragma: no cover - exercised by user data edge cases
            score = np.nan
            status = f"failed: {exc.__class__.__name__}"

        rows.append(
            {
                "model": model_name,
                "metric": metric_name,
                "score": score,
                "status": status,
            }
        )

    return pd.DataFrame(rows).sort_values(
        by="score", ascending=False, na_position="last"
    )


def repeated_stratified_cv_scores(
    estimators: dict,
    X: pd.DataFrame,
    y: pd.Series,
    metric_name: str,
    binary: bool,
    preprocessor_factory: Callable,
    cv_splits: int = 5,
    cv_repeats: int = 3,
    confidence_level: float = 0.95,
    random_state: Optional[int] = 42,
    groups: Optional[pd.Series] = None,
) -> pd.DataFrame:
    X = pd.DataFrame(X).reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True)
    groups = pd.Series(groups).reset_index(drop=True) if groups is not None else None
    min_class_count = int(y.value_counts().min())
    effective_splits = min(cv_splits, min_class_count)
    if groups is not None:
        effective_splits = min(effective_splits, int(groups.nunique()))

    if effective_splits < 2:
        rows = [
            {
                "model": model_name,
                "metric": metric_name,
                "mean_score": np.nan,
                "std_score": np.nan,
                "ci_low": np.nan,
                "ci_high": np.nan,
                "n_scores": 0,
                "status": "skipped: fewer than two samples in at least one class",
            }
            for model_name in estimators
        ]
        result = pd.DataFrame(rows)
        result.attrs["cv_splits"] = effective_splits
        return result

    if groups is None:
        splitter = RepeatedStratifiedKFold(
            n_splits=effective_splits,
            n_repeats=cv_repeats,
            random_state=random_state,
        )
        folds = list(splitter.split(X, y))
    else:
        folds = []
        for repeat in range(cv_repeats):
            repeat_seed = None if random_state is None else random_state + repeat
            splitter = StratifiedGroupKFold(
                n_splits=effective_splits,
                shuffle=True,
                random_state=repeat_seed,
            )
            folds.extend(splitter.split(X, y, groups))
    rows = []

    for model_name, estimator in estimators.items():
        scores = []
        status = "ok"
        for train_idx, eval_idx in folds:
            try:
                fitted_estimator, X_eval_transformed = (
                    _fit_estimator_with_preprocessing(
                        estimator=estimator,
                        model_name=model_name,
                        X_train=X.iloc[train_idx],
                        y_train=y.iloc[train_idx],
                        X_evaluation=X.iloc[eval_idx],
                        preprocessor_factory=preprocessor_factory,
                    )
                )
                scores.append(
                    score_estimator(
                        fitted_estimator,
                        X_eval_transformed,
                        y.iloc[eval_idx],
                        metric_name=metric_name,
                        binary=binary,
                    )
                )
            except (
                Exception
            ) as exc:  # pragma: no cover - exercised by user data edge cases
                status = f"failed: {exc.__class__.__name__}"
                break

        summary = summarize_scores(scores, confidence_level=confidence_level)
        rows.append(
            {
                "model": model_name,
                "metric": metric_name,
                **summary,
                "status": status,
            }
        )

    result = pd.DataFrame(rows).sort_values(
        by="mean_score", ascending=False, na_position="last"
    )
    result.attrs["cv_splits"] = effective_splits
    return result


def summarize_scores(scores, confidence_level: float = 0.95) -> dict:
    score_array = pd.Series(scores, dtype="float64").dropna().to_numpy()
    n_scores = len(score_array)

    if n_scores == 0:
        return {
            "mean_score": np.nan,
            "std_score": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "n_scores": 0,
        }

    mean_score = float(np.mean(score_array))
    std_score = float(np.std(score_array, ddof=1)) if n_scores > 1 else 0.0
    if n_scores > 1:
        critical_value = stats.t.ppf((1 + confidence_level) / 2, df=n_scores - 1)
        margin = critical_value * std_score / np.sqrt(n_scores)
        ci_low = max(0.0, mean_score - margin)
        ci_high = min(1.0, mean_score + margin)
    else:
        ci_low = np.nan
        ci_high = np.nan

    return {
        "mean_score": mean_score,
        "std_score": std_score,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n_scores": n_scores,
    }


def score_estimator(estimator, X, y, metric_name: str, binary: bool) -> float:
    y = np.asarray(y)
    if metric_name == "roc_auc_score":
        if not hasattr(estimator, "predict_proba"):
            return np.nan
        y_score = estimator.predict_proba(X)
        if binary:
            y_score = y_score[:, 1]
            return roc_auc_score(y, y_score)
        return roc_auc_score(y, y_score, multi_class="ovr", average="weighted")

    y_pred = estimator.predict(X)
    if metric_name == "accuracy_score":
        return accuracy_score(y, y_pred)
    if metric_name == "balanced_accuracy_score":
        return balanced_accuracy_score(y, y_pred)
    if metric_name == "precision_score":
        return precision_score(y, y_pred, average="weighted", zero_division=0)
    if metric_name == "recall_score":
        return recall_score(y, y_pred, average="weighted", zero_division=0)
    if metric_name == "f1_score":
        return f1_score(y, y_pred, average="weighted", zero_division=0)
    if metric_name == "jaccard_score":
        return jaccard_score(y, y_pred, average="weighted", zero_division=0)

    raise ValueError(f"Unsupported metric: {metric_name}")


def _fit_estimator_with_preprocessing(
    model_name: str,
    estimator,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_evaluation: pd.DataFrame,
    preprocessor_factory: Callable,
):
    preprocessor = make_preprocessor(preprocessor_factory, model_name)
    y_train = pd.Series(y_train, index=X_train.index)

    if preprocessor is not None:
        X_train_transformed, y_train_transformed = preprocessor.fit_transform(
            X_train.copy(), y_train.copy()
        )
        X_eval_transformed = preprocessor.transform(X_evaluation.copy())
    else:
        X_train_transformed = (
            X_train.to_numpy() if hasattr(X_train, "to_numpy") else X_train
        )
        y_train_transformed = y_train.to_numpy()
        X_eval_transformed = (
            X_evaluation.to_numpy()
            if hasattr(X_evaluation, "to_numpy")
            else X_evaluation
        )

    fitted_estimator = clone(estimator)
    fit_estimator(fitted_estimator, X_train_transformed, y_train_transformed)
    return fitted_estimator, X_eval_transformed
