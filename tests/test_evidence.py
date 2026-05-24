import numpy as np
import pandas as pd

import mamut.model_selection as model_selection
from mamut.evidence import build_selection_guidance, detect_leakage_risks
from mamut.wrapper import Mamut


def _exclude_all_except(model_name: str) -> list[str]:
    return [name for name in model_selection.model_param_dict if name != model_name]


def _classification_frame(n_samples: int = 72) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(123)
    target = np.tile([0, 1], n_samples // 2)
    X = pd.DataFrame(
        {
            "signal": target + rng.normal(0, 0.2, n_samples),
            "noise": rng.normal(0, 1, n_samples),
            "segment": np.where(target == 1, "high", "low"),
        }
    )
    order = rng.permutation(n_samples)
    return X.iloc[order].reset_index(drop=True), pd.Series(target[order], name="target")


def test_detect_leakage_risks_flags_target_copy_and_target_like_name():
    y = pd.Series([0, 1, 0, 1], name="target")
    X = pd.DataFrame(
        {
            "target": y,
            "customer_id": [100, 101, 102, 103],
            "feature": [0.1, 0.2, 0.3, 0.4],
        }
    )

    checks = detect_leakage_risks(X, y)

    assert "feature_equals_target" in set(checks["check"])
    assert "target_column_name_present" in set(checks["check"])
    assert "id_like_high_cardinality_feature" in set(checks["check"])


def test_conflicting_duplicate_features_are_data_ambiguity_not_leakage_warning():
    X = pd.DataFrame({"signal": [1, 1, 2], "segment": ["a", "a", "b"]})
    y = pd.Series([0, 1, 1], name="target")

    checks = detect_leakage_risks(X, y)
    collision = checks.loc[
        checks["check"].eq("duplicate_features_conflicting_targets")
    ].iloc[0]

    assert collision["severity"] == "info"
    assert "not evidence of leakage" in collision["message"]


def test_generate_evidence_contains_baselines_and_score_intervals():
    X, y = _classification_frame()
    mamut = Mamut(
        exclude_models=_exclude_all_except("GaussianNB"),
        n_iterations=1,
        optimization_method="random_search",
        random_state=42,
        holdout_size=0.2,
        evidence_cv_splits=2,
        evidence_cv_repeats=1,
        num_imputation="mean",
    )
    mamut.fit(X, y)

    evidence = mamut.generate_evidence()

    assert set(evidence) == {
        "validation_integrity",
        "leakage_checks",
        "baseline_comparison",
        "score_stability",
        "selection_guidance",
    }
    assert "Dummy Most Frequent" in set(mamut.baseline_comparison_["model"])
    assert "Logistic Regression" in set(mamut.baseline_comparison_["model"])
    assert "Random Forest" in set(mamut.baseline_comparison_["model"])
    assert {"mean_score", "std_score", "ci_low", "ci_high", "n_scores"}.issubset(
        mamut.score_stability_.columns
    )
    assert mamut.score_stability_["n_scores"].min() == 2
    assert mamut.score_stability_["ci_low"].dropna().between(0, 1).all()
    assert mamut.score_stability_["ci_high"].dropna().between(0, 1).all()
    assert mamut.validation_integrity_.iloc[0]["evaluation_dataset"] == "holdout"
    assert mamut.selection_guidance_.iloc[0]["status"] in {
        "blocked",
        "challenged",
        "challenged_strong",
        "confirmed",
        "confirmed_with_caution",
        "inconclusive",
    }


def test_generate_evidence_uses_validation_when_holdout_is_absent():
    X, y = _classification_frame()
    mamut = Mamut(
        exclude_models=_exclude_all_except("GaussianNB"),
        n_iterations=1,
        optimization_method="random_search",
        random_state=42,
        evidence_cv_splits=2,
        evidence_cv_repeats=1,
        num_imputation="mean",
    )
    mamut.fit(X, y)

    mamut.generate_evidence()

    assert mamut.validation_integrity_.iloc[0]["evaluation_dataset"] == "validation"
    assert not bool(mamut.validation_integrity_.iloc[0]["holdout_available"])


def test_final_confirmation_evidence_can_omit_non_selected_candidates():
    X, y = _classification_frame()
    mamut = Mamut(
        include_models=["GaussianNB", "LogisticRegression"],
        n_iterations=1,
        optimization_method="random_search",
        holdout_size=0.2,
        evidence_cv_splits=2,
        evidence_cv_repeats=1,
    )
    mamut.fit(X, y)

    mamut.generate_evidence(dataset="holdout", include_candidate_comparison=False)

    labels = set(mamut.baseline_comparison_["model"])
    assert not any(label.startswith("MAMUT Candidate") for label in labels)
    assert any(label.startswith("MAMUT Selected") for label in labels)


def test_generate_evidence_reports_group_disjoint_validation():
    X, y = _classification_frame()
    groups = pd.Series(np.repeat(np.arange(36), 2))
    mamut = Mamut(
        exclude_models=_exclude_all_except("GaussianNB"),
        n_iterations=1,
        optimization_method="random_search",
        random_state=42,
        holdout_size=0.2,
        evidence_cv_splits=2,
        evidence_cv_repeats=1,
        num_imputation="mean",
    )
    mamut.fit(X, y, groups=groups)

    mamut.generate_evidence()
    row = mamut.validation_integrity_.iloc[0]

    assert bool(row["grouped_validation"])
    assert row["cv_strategy"] == "RepeatedStratifiedGroupKFold"
    assert row["evaluation_group_overlap"] == 0


def test_selection_guidance_challenges_holdout_winner_without_silent_promotion():
    selected = "MAMUT Selected (GaussianNB)"
    baseline_comparison = pd.DataFrame(
        [
            {"model": selected, "score": 0.80},
            {"model": "Random Forest", "score": 0.90},
        ]
    )
    score_stability = pd.DataFrame(
        [
            {
                "model": selected,
                "mean_score": 0.82,
                "std_score": 0.03,
                "ci_low": 0.78,
                "ci_high": 0.86,
            },
            {
                "model": "Random Forest",
                "mean_score": 0.88,
                "std_score": 0.02,
                "ci_low": 0.85,
                "ci_high": 0.91,
            },
        ]
    )
    leakage_checks = pd.DataFrame(
        [{"severity": "pass", "check": "basic_leakage_screen", "message": "ok"}]
    )

    guidance = build_selection_guidance(
        baseline_comparison=baseline_comparison,
        score_stability=score_stability,
        leakage_checks=leakage_checks,
        selected_model_label=selected,
        evaluation_dataset="holdout",
        practical_margin=0.01,
    )

    row = guidance.iloc[0]
    assert row["status"] == "challenged"
    assert row["recommended_model"] == selected
    assert row["review_candidate"] == "Random Forest"
    assert "Do not silently promote" in row["action"]


def test_selection_guidance_confirms_competitive_selected_model():
    selected = "MAMUT Selected (LogisticRegression)"
    baseline_comparison = pd.DataFrame(
        [
            {"model": selected, "score": 0.91},
            {"model": "Random Forest", "score": 0.905},
        ]
    )
    score_stability = pd.DataFrame(
        [
            {
                "model": selected,
                "mean_score": 0.90,
                "std_score": 0.02,
                "ci_low": 0.87,
                "ci_high": 0.93,
            },
            {
                "model": "Random Forest",
                "mean_score": 0.895,
                "std_score": 0.02,
                "ci_low": 0.86,
                "ci_high": 0.93,
            },
        ]
    )
    leakage_checks = pd.DataFrame(
        [{"severity": "pass", "check": "basic_leakage_screen", "message": "ok"}]
    )

    guidance = build_selection_guidance(
        baseline_comparison=baseline_comparison,
        score_stability=score_stability,
        leakage_checks=leakage_checks,
        selected_model_label=selected,
        evaluation_dataset="validation",
        practical_margin=0.01,
    )

    assert guidance.iloc[0]["status"] == "confirmed"
    assert guidance.iloc[0]["recommended_model"] == selected


def test_selection_guidance_blocks_critical_leakage():
    selected = "MAMUT Selected (LogisticRegression)"
    baseline_comparison = pd.DataFrame([{"model": selected, "score": 0.91}])
    score_stability = pd.DataFrame(
        [
            {
                "model": selected,
                "mean_score": 0.90,
                "std_score": 0.02,
                "ci_low": 0.87,
                "ci_high": 0.93,
            }
        ]
    )
    leakage_checks = pd.DataFrame(
        [{"severity": "critical", "check": "feature_equals_target", "message": "bad"}]
    )

    guidance = build_selection_guidance(
        baseline_comparison=baseline_comparison,
        score_stability=score_stability,
        leakage_checks=leakage_checks,
        selected_model_label=selected,
        evaluation_dataset="validation",
    )

    assert guidance.iloc[0]["status"] == "blocked"
