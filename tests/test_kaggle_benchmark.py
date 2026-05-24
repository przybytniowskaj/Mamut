import math
from pathlib import Path

import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

import scripts.benchmark_kaggle as benchmark_kaggle
from scripts.benchmark_kaggle import (
    BenchmarkConfig,
    _coerce_bool_predictions,
    _make_mamut,
    estimated_candidate_fit_upper_bound,
    fit_submission_model,
    format_results,
    group_bootstrap_accuracy_interval,
    household_components,
    passenger_groups,
    prepare_spaceship_features,
    prepare_spaceship_modeling_split,
    relational_overlap_audit,
    reserve_confirmation_partition,
    run_confirmation_benchmark,
    split_spaceship_validation,
    summarize_runs,
    validate_recipe_scope,
    validation_estimand,
    write_submission,
)


def _spaceship_train() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "PassengerId": ["0001_01", "0002_01", "0003_01", "0003_02"],
            "HomePlanet": ["Europa", "Earth", "Mars", "Mars"],
            "CryoSleep": [False, True, False, False],
            "Cabin": ["B/0/P", "F/1/S", "A/2/S", None],
            "Destination": [
                "TRAPPIST-1e",
                "TRAPPIST-1e",
                "55 Cancri e",
                "PSO J318.5-22",
            ],
            "Age": [39.0, 24.0, 58.0, 33.0],
            "VIP": [False, False, True, False],
            "RoomService": [0.0, 109.0, 43.0, None],
            "FoodCourt": [0.0, 9.0, 3576.0, 0.0],
            "ShoppingMall": [0.0, 25.0, 0.0, 0.0],
            "Spa": [0.0, 549.0, 6715.0, 0.0],
            "VRDeck": [0.0, 44.0, 49.0, 0.0],
            "Name": ["A B", "C D", "E F", "G H"],
            "Transported": [False, True, False, True],
        }
    )


def _spaceship_test() -> pd.DataFrame:
    return (
        _spaceship_train()
        .drop(columns=["Transported"])
        .assign(PassengerId=["9001_01", "9002_01", "9003_01", "9003_02"])
    )


def test_spaceship_basic_recipe_creates_expected_feature_contract():
    X, y, X_test, passenger_ids = prepare_spaceship_features(
        _spaceship_train(),
        _spaceship_test(),
        recipe="spaceship_basic",
    )

    assert list(X.columns) == list(X_test.columns)
    assert len(X) == len(y) == 4
    assert passenger_ids.tolist() == ["9001_01", "9002_01", "9003_01", "9003_02"]
    assert {
        "PassengerNumber",
        "PassengerGroupSize",
        "PassengerGroupIsSolo",
        "CabinDeck",
        "CabinNumber",
        "CabinSide",
    }.issubset(X.columns)
    assert {"SpendingTotal", "SpendingAny", "SpendingMissingCount"}.issubset(X.columns)
    assert {"PassengerId", "PassengerGroup", "Name", "Cabin", "Transported"}.isdisjoint(
        X.columns
    )
    assert y.dtype == bool


def test_spaceship_inductive_recipe_excludes_batch_cohort_features():
    X, _, X_test, _ = prepare_spaceship_features(
        _spaceship_train(),
        _spaceship_test(),
        recipe="spaceship_inductive",
    )

    assert list(X.columns) == list(X_test.columns)
    assert "PassengerNumber" in X.columns
    assert "PassengerGroupSize" not in X.columns
    assert "PassengerGroupIsSolo" not in X.columns


def test_spaceship_v2_recipes_add_domain_features_and_separate_batch_features():
    inductive, _, _, _ = prepare_spaceship_features(
        _spaceship_train(), _spaceship_test(), recipe="spaceship_inductive_v2"
    )
    cohort, _, _, _ = prepare_spaceship_features(
        _spaceship_train(), _spaceship_test(), recipe="spaceship_cohort_v2"
    )

    assert {
        "FamilyName",
        "AgeBand",
        "NoSpend",
        "AmenitiesUsed",
        "CryoSpendConflict",
        "CabinNumberBand",
    }.issubset(inductive.columns)
    assert {"FamilyBatchSize", "FamilyBatchShared"}.isdisjoint(inductive.columns)
    assert {"FamilyBatchSize", "FamilyBatchShared"}.issubset(cohort.columns)


def test_cohort_v2_requires_household_component_scope():
    with pytest.raises(ValueError, match="household_component"):
        validate_recipe_scope("spaceship_cohort_v2", "passenger")


def test_competition_v3_allows_target_free_batch_features_under_passenger_scope():
    validate_recipe_scope("spaceship_competition_v3", "passenger")
    features, _, _, _ = prepare_spaceship_features(
        _spaceship_train(), _spaceship_test(), recipe="spaceship_competition_v3"
    )

    assert {"PassengerGroupSize", "FamilyBatchSize", "FamilyBatchShared"}.issubset(
        features.columns
    )
    assert "competition-aligned" in validation_estimand(
        "spaceship_competition_v3", "passenger"
    )


def test_candidate_fit_estimate_makes_nested_search_cost_explicit():
    fixed = BenchmarkConfig(
        competition="spaceship-titanic",
        recipe="spaceship_inductive_v2",
        runs=5,
        n_iterations=3,
        random_state=42,
        holdout_size=0.2,
        validation_protocol="grouped",
        score_metric="accuracy",
        search_profile="balanced",
        selection_strategy="single_split",
        selection_cv_splits=3,
        selection_cv_repeats=1,
        selection_practical_margin=0.005,
        preprocessing_profile="auto",
        optimization_method="random_search",
        excluded_models=(),
        included_models=("LGBMClassifier",),
    )
    broad_nested = BenchmarkConfig(
        **{
            **fixed.__dict__,
            "runs": 3,
            "selection_strategy": "nested_cv",
            "included_models": (
                "RandomForestClassifier",
                "ExtraTreesClassifier",
                "LGBMClassifier",
                "CatBoostClassifier",
                "XGBClassifier",
            ),
        }
    )

    assert estimated_candidate_fit_upper_bound(fixed) == 85
    assert estimated_candidate_fit_upper_bound(broad_nested) == 1008


def test_benchmark_threads_are_forwarded_to_mamut():
    config = BenchmarkConfig(
        competition="spaceship-titanic",
        recipe="spaceship_inductive_v2",
        runs=1,
        n_iterations=1,
        random_state=42,
        holdout_size=0.2,
        validation_protocol="grouped",
        score_metric="accuracy",
        search_profile="quick",
        selection_strategy="single_split",
        selection_cv_splits=2,
        selection_cv_repeats=1,
        selection_practical_margin=0.005,
        preprocessing_profile="auto",
        optimization_method="random_search",
        excluded_models=(),
        included_models=("LGBMClassifier",),
        n_jobs=-1,
    )

    assert _make_mamut(config, random_state=42, final_refit=True).n_jobs == -1


def test_passenger_groups_are_derived_from_passenger_identifier_prefix():
    groups = passenger_groups(_spaceship_train())

    assert groups.tolist() == ["0001", "0002", "0003", "0003"]


def test_household_components_join_distinct_passenger_groups_with_family_name():
    frame = _spaceship_train().copy()
    frame.loc[0, "Name"] = "A Shared"
    frame.loc[1, "Name"] = "B Shared"

    groups = household_components(frame)

    assert groups.iloc[0] == groups.iloc[1]
    assert groups.iloc[0] != groups.iloc[2]


def test_grouped_validation_protocol_has_no_passenger_group_overlap():
    modeling, holdout, modeling_groups, holdout_groups = split_spaceship_validation(
        _spaceship_train(),
        validation_protocol="grouped",
        holdout_size=0.5,
        random_state=42,
    )

    assert len(modeling) + len(holdout) == len(_spaceship_train())
    assert set(modeling_groups).isdisjoint(holdout_groups)


def test_reserved_confirmation_partition_is_group_disjoint():
    development, confirmation, development_groups, confirmation_groups = (
        reserve_confirmation_partition(
            _spaceship_train(),
            validation_protocol="grouped",
            group_scope="passenger",
            confirmation_size=0.5,
            confirmation_seed=123,
        )
    )

    assert len(development) + len(confirmation) == len(_spaceship_train())
    assert set(development_groups).isdisjoint(confirmation_groups)


def test_relational_overlap_audit_reports_recurring_family_names_separately():
    modeling = _spaceship_train().iloc[:2].copy()
    evaluation = _spaceship_train().iloc[2:].copy()
    modeling.loc[0, "Name"] = "A Shared"
    evaluation.loc[0, "Name"] = "B Shared"

    rows = pd.DataFrame(relational_overlap_audit(modeling, evaluation)).set_index(
        "feature"
    )

    assert rows.loc["PassengerGroup", "shared_values"] == 0
    assert rows.loc["FamilyName", "shared_values"] == 1
    assert rows.loc["FamilyName", "evaluation_rows_with_seen_value"] == 1


def test_confirmation_scores_only_selected_prediction_without_passing_holdout_to_fit(
    monkeypatch,
):
    captured = {}

    class DummyModel:
        random_state = 42
        selected_estimator_ = type("ChosenEstimator", (), {})()
        selection_summary_ = pd.DataFrame([{"model": "ChosenEstimator"}])
        binary = True
        score_metric_name = "accuracy_score"

        @staticmethod
        def score_metric(y_true, y_pred):
            return float((pd.Series(y_true).to_numpy() == y_pred).mean())

        def fit(self, X, y, **kwargs):
            captured.update(kwargs)

        @staticmethod
        def predict(X):
            return [True] * len(X)

    monkeypatch.setattr(
        benchmark_kaggle,
        "_make_mamut",
        lambda config, random_state, final_refit: DummyModel(),
    )
    monkeypatch.setattr(
        benchmark_kaggle,
        "benchmark_integrity_audit",
        lambda *args, **kwargs: (
            pd.DataFrame([{"model": "Random Forest", "score": 0.5}]),
            pd.DataFrame([{"severity": "pass"}]),
            0.0,
        ),
    )
    config = BenchmarkConfig(
        competition="spaceship-titanic",
        recipe="spaceship_inductive_v2",
        runs=1,
        n_iterations=1,
        random_state=42,
        holdout_size=0.5,
        validation_protocol="grouped",
        score_metric="accuracy",
        search_profile="quick",
        selection_strategy="nested_cv",
        selection_cv_splits=2,
        selection_cv_repeats=1,
        selection_practical_margin=0.005,
        preprocessing_profile="auto",
        optimization_method="random_search",
        excluded_models=(),
    )
    development = _spaceship_train().iloc[:2]
    confirmation = _spaceship_train().iloc[2:]

    run_confirmation_benchmark(
        development,
        confirmation,
        pd.Series(["0001", "0002"]),
        pd.Series(["0003", "0003"]),
        config=config,
    )

    assert "X_holdout" not in captured
    assert "y_holdout" not in captured
    assert "groups_holdout" not in captured


def test_submission_refits_frozen_confirmation_parameters_without_retuning(
    monkeypatch,
):
    class FrozenModel:
        selected_estimator_ = DummyClassifier(strategy="most_frequent")

        @staticmethod
        def _make_model_preprocessor(model_name):
            return None

    monkeypatch.setattr(
        benchmark_kaggle,
        "_make_mamut",
        lambda *args, **kwargs: pytest.fail("Frozen submission must not retune MAMUT."),
    )
    config = BenchmarkConfig(
        competition="spaceship-titanic",
        recipe="spaceship_inductive_v2",
        runs=1,
        n_iterations=1,
        random_state=42,
        holdout_size=0.5,
        validation_protocol="grouped",
        score_metric="accuracy",
        search_profile="quick",
        selection_strategy="single_split",
        selection_cv_splits=2,
        selection_cv_repeats=1,
        selection_practical_margin=0.005,
        preprocessing_profile="auto",
        optimization_method="random_search",
        excluded_models=(),
    )

    submission, returned_model = fit_submission_model(
        _spaceship_train(),
        _spaceship_test(),
        config=config,
        frozen_model=FrozenModel(),
    )

    assert list(submission.columns) == ["PassengerId", "Transported"]
    assert len(submission) == len(_spaceship_test())
    assert isinstance(returned_model, FrozenModel)


def test_raw_recipe_preserves_original_feature_columns_without_target():
    X, y, X_test, _ = prepare_spaceship_features(
        _spaceship_train(),
        _spaceship_test(),
        recipe="raw",
    )

    assert "Transported" not in X.columns
    assert "PassengerId" in X.columns
    assert list(X.columns) == list(X_test.columns)
    assert y.tolist() == [False, True, False, True]


def test_modeling_split_prepares_features_after_outer_split():
    modeling = _spaceship_train().iloc[:3]
    holdout = _spaceship_train().iloc[3:]

    X_modeling, y_modeling, X_holdout, y_holdout = prepare_spaceship_modeling_split(
        modeling,
        holdout,
        recipe="spaceship_basic",
    )

    assert list(X_modeling.columns) == list(X_holdout.columns)
    assert len(X_modeling) == len(y_modeling) == 3
    assert len(X_holdout) == len(y_holdout) == 1
    assert "Transported" not in X_modeling.columns


def test_summarize_runs_tracks_score_stability_and_audit_candidate_delta():
    runs = pd.DataFrame(
        [
            {
                "selected_holdout_score": 0.80,
                "audit_candidate_delta": 0.00,
                "baseline_uplift": 0.05,
                "guidance_status": "confirmed",
                "duration_seconds": 1.0,
            },
            {
                "selected_holdout_score": 0.82,
                "audit_candidate_delta": 0.02,
                "baseline_uplift": 0.03,
                "guidance_status": "challenged",
                "duration_seconds": 2.0,
            },
        ]
    )

    summary = summarize_runs(runs, score_column="selected_holdout_score")

    assert summary["runs"] == 2
    assert summary["mean_score"] == 0.81
    assert summary["mean_audit_candidate_delta"] == 0.01
    assert math.isclose(summary["p90_audit_candidate_delta"], 0.018)
    assert summary["challenged_rate"] == 0.5
    assert summary["stability_low"] == 0.80
    assert summary["stability_high"] == 0.82


def test_group_bootstrap_accuracy_interval_uses_recorded_group_predictions():
    predictions = pd.DataFrame(
        {
            "repeat": [0, 0, 0, 0],
            "group": ["a", "a", "b", "b"],
            "correct": [True, True, False, False],
        }
    )

    low, high = group_bootstrap_accuracy_interval(
        predictions, random_state=42, n_resamples=200
    )

    assert 0 <= low <= 0.5 <= high <= 1


def test_write_submission_matches_kaggle_schema(tmp_path: Path):
    submission = pd.DataFrame(
        {
            "PassengerId": ["9001_01", "9002_01"],
            "Transported": [True, False],
        }
    )

    path = write_submission(submission, tmp_path / "submission.csv")

    reloaded = pd.read_csv(path)
    assert list(reloaded.columns) == ["PassengerId", "Transported"]
    assert reloaded["PassengerId"].tolist() == ["9001_01", "9002_01"]
    assert reloaded["Transported"].tolist() == [True, False]


def test_format_results_markdown_contains_aggregate_and_run_tables():
    config = BenchmarkConfig(
        competition="spaceship-titanic",
        recipe="spaceship_basic",
        runs=1,
        n_iterations=1,
        random_state=42,
        holdout_size=0.2,
        validation_protocol="grouped",
        score_metric="accuracy",
        search_profile="quick",
        selection_strategy="single_split",
        selection_cv_splits=5,
        selection_cv_repeats=2,
        selection_practical_margin=0.005,
        preprocessing_profile="auto",
        optimization_method="random_search",
        excluded_models=("SVC",),
    )
    payload = {
        "metadata": {},
        "config": config.__dict__,
        "aggregate": {
            "runs": 1,
            "mean_score": 0.81234,
            "group_bootstrap_low": 0.74123,
        },
        "runs": [
            {
                "run": 0,
                "selected_model": "XGBClassifier",
                "selected_holdout_score": 0.81234,
            }
        ],
    }

    output = format_results(payload, "markdown")

    assert "## Aggregate" in output
    assert "## Runs" in output
    assert "XGBClassifier" in output
    assert "0.8123" in output
    assert "0.7412" in output


def test_format_results_json_converts_nan_to_null():
    payload = {
        "metadata": {},
        "config": {},
        "aggregate": {"runs": 1, "mean_score": float("nan")},
        "runs": [{"run": 0, "baseline_uplift": float("nan")}],
    }

    output = format_results(payload, "json")

    assert '"mean_score": null' in output
    assert '"baseline_uplift": null' in output


def test_bool_prediction_coercion_for_submission_output():
    assert _coerce_bool_predictions([True, False, "True", "false", 1, 0]) == [
        True,
        False,
        True,
        False,
        True,
        False,
    ]
